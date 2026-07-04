.PHONY: setup data train experiments figures test lint api demo

PYTHON ?= python3.11
VENV := .venv
VENV_BIN := $(VENV)/bin

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_BIN)/pip install --upgrade pip
	$(VENV_BIN)/pip install -r requirements.txt
	$(VENV_BIN)/pip install -e .

data:
	$(VENV_BIN)/python -m roar.graph.load_graph
	$(VENV_BIN)/python -m roar.graph.map_sensors
	$(VENV_BIN)/python -m roar.graph.features

train:
	$(VENV_BIN)/python -m roar.predictor.train

experiments:
	$(VENV_BIN)/python -m roar.eval.harness

figures:
	$(VENV_BIN)/python -m roar.eval.figures

test:
	$(VENV_BIN)/pytest -q

lint:
	$(VENV_BIN)/ruff check roar tests

api:
	$(VENV_BIN)/uvicorn roar.api.app:app --reload

demo:
	$(VENV_BIN)/streamlit run demo/app.py
