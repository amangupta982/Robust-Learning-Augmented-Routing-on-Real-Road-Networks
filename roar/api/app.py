"""Phase 7: the deployable routing API. Stateless request handling; the
graph and the trained predictor are loaded ONCE at process startup (see
`lifespan` below), never per-request.

## The ground-truth caveat (read before trusting this API's numbers)

RobustnessGuard (roar/routing/guard.py) needs a ground-truth cost function
to evaluate a candidate path before deciding whether to return it. In the
paper's offline replay experiments (Phases 4-6), that ground truth is
`OraclePredictor` -- the REAL realized METR-LA speeds, available because
every replayed query is historical. This live API has no such
foreknowledge (no oracle "what actually happened" exists for a route
starting now): it uses the SAME trained LightGBM predictor as both the
planning predictor AND the guard's reference. That makes the guard here a
**self-consistency guard**: it still protects against the search
returning a policy that's bad even by the predictor's OWN model (e.g. an
artifact of the blended heuristic's potential inadmissibility), and it
still reports a `robustness_bound` on that basis -- but it is NOT the
paper's stronger empirical guarantee (bounded degradation relative to
REALITY, verified in Phases 4-6 against real held-out speeds). Serving a
genuine real-world bound live would require a live ground-truth signal
(e.g. a real-time traffic feed used purely for verification) or an online
monitoring/replanning mechanism; both are out of scope for this phase (see
LIMITATIONS.md).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from roar.eval.metrics import resolve_model_dir
from roar.graph.config import load_config
from roar.graph.features import FEATURES_PATH
from roar.predictor.lightgbm import PREDICTOR_CONFIG_PATH, LightGBMPredictor
from roar.routing.baselines import predictor_cost_fn
from roar.routing.graph import RoutingGraph, load_road_graph
from roar.routing.robust_astar import RobustAStar

DEFAULT_ALPHA = 0.3
DEFAULT_LAMBDA_BASE = 1.0

_state: dict[str, Any] = {}


def _load_predictor(model_version: str = "latest") -> tuple[LightGBMPredictor, str]:
    data_cfg = load_config()
    predictor_cfg = load_config(PREDICTOR_CONFIG_PATH)
    predictor_cfg.setdefault("time_bucket_minutes", data_cfg["time_bucket_minutes"])
    features_df = pd.read_parquet(FEATURES_PATH)
    model_dir = resolve_model_dir(model_version)
    predictor = LightGBMPredictor.load(model_dir, features_df)
    return predictor, model_dir.name


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    graph = load_road_graph()
    predictor, model_version = _load_predictor()
    _state["graph"] = graph
    _state["predictor"] = predictor
    _state["model_version"] = model_version
    _state["loaded_at"] = dt.datetime.now(dt.UTC)
    yield
    _state.clear()


app = FastAPI(
    title="ROAR Routing API",
    description=(
        "Robust learning-augmented routing over the real LA drive network. "
        "See roar/api/app.py's module docstring for the ground-truth caveat "
        "on robustness_bound in this live deployment."
    ),
    version="0.0.1",
    lifespan=lifespan,
)


class RouteRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    origin: str = Field(description="Origin OSM node id, as a string")
    destination: str = Field(description="Destination OSM node id, as a string")
    depart_time: dt.datetime
    lambda_base: float = Field(
        default=DEFAULT_LAMBDA_BASE, alias="lambda", ge=0.0, le=1.0,
        description="Trust in the ML-estimate heuristic term, in [0, 1]",
    )
    alpha: float = Field(
        default=DEFAULT_ALPHA, ge=0.0,
        description="Guard bound: realized cost never exceeds (1+alpha) * classical cost",
    )
    confidence_modulated: bool = False


class RobustnessBoundResponse(BaseModel):
    alpha: float
    classical_cost: float
    realized_cost: float
    ratio: float
    guard_invoked: bool


class RouteResponse(BaseModel):
    path: list[str] | None
    cost: float
    eta: dt.datetime | None
    node_expansions: int
    latency_ms: float
    robustness_bound: RobustnessBoundResponse


def _get_graph() -> RoutingGraph:
    graph = _state.get("graph")
    if graph is None:
        raise HTTPException(status_code=503, detail="graph not loaded yet")
    return graph


@app.post("/route", response_model=RouteResponse)
def route(request: RouteRequest) -> RouteResponse:
    graph = _get_graph()
    predictor = _state.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail="predictor not loaded yet")

    if request.origin not in graph.nodes:
        raise HTTPException(status_code=422, detail=f"unknown origin node: {request.origin!r}")
    if request.destination not in graph.nodes:
        raise HTTPException(
            status_code=422, detail=f"unknown destination node: {request.destination!r}"
        )

    # Self-consistency guard (see module docstring): the same predictor
    # serves as both the planning cost function and the guard's reference,
    # since no independent ground truth exists for a route starting now.
    ground_truth_cost_fn = predictor_cost_fn(predictor)
    robust = RobustAStar(
        graph,
        predictor,
        ground_truth_cost_fn,
        alpha=request.alpha,
        lambda_base=request.lambda_base,
        confidence_modulated=request.confidence_modulated,
    )
    result = robust.search(request.origin, request.destination, request.depart_time)

    eta = None
    if result.path is not None and result.cost != float("inf"):
        eta = request.depart_time + dt.timedelta(seconds=result.cost)

    return RouteResponse(
        path=result.path,
        cost=result.cost,
        eta=eta,
        node_expansions=result.node_expansions,
        latency_ms=result.latency_ms,
        robustness_bound=RobustnessBoundResponse(
            alpha=result.robustness_bound.alpha,
            classical_cost=result.robustness_bound.classical_cost,
            realized_cost=result.robustness_bound.realized_cost,
            ratio=result.robustness_bound.ratio,
            guard_invoked=result.robustness_bound.guard_invoked,
        ),
    )


@app.get("/predictor/health")
def predictor_health() -> dict[str, Any]:
    predictor = _state.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail="predictor not loaded yet")
    return {
        "status": "ok",
        "predictor_type": "lightgbm",
        "model_version": _state.get("model_version"),
        "loaded_at": _state["loaded_at"].isoformat(),
    }
