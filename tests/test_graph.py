"""Graph connectivity check.

Requires the real processed graph (`make data` / `python -m roar.graph.load_graph`
must have run first) — skipped otherwise rather than faked, since a fabricated
graph would defeat the point of the test.
"""

import networkx as nx
import osmnx as ox
import pytest
from roar.graph.load_graph import GRAPH_PATH, largest_scc_fraction

pytestmark = pytest.mark.skipif(
    not GRAPH_PATH.exists(),
    reason="data/processed graph not built yet; run `make data` first",
)


def test_graph_loads_and_scc_is_documented():
    graph = ox.load_graphml(GRAPH_PATH)
    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0

    scc_size, n_nodes, fraction = largest_scc_fraction(graph)
    assert scc_size > 0
    assert 0.0 < fraction <= 1.0
    # A drive network for a real metro-area bbox should be overwhelmingly
    # one connected component; a much lower fraction signals a bad bbox or
    # download rather than a real property of LA's road network.
    assert fraction > 0.9, (
        f"largest SCC is only {fraction:.1%} of nodes ({scc_size}/{n_nodes}) "
        "-- investigate before trusting downstream routing experiments"
    )


def test_scc_helper_matches_networkx_directly():
    graph = ox.load_graphml(GRAPH_PATH)
    scc_size, n_nodes, fraction = largest_scc_fraction(graph)
    expected = len(max(nx.strongly_connected_components(graph), key=len))
    assert scc_size == expected
    assert n_nodes == graph.number_of_nodes()
