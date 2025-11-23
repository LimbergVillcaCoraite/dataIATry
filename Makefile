PYTHON=python3
VENV=.venv

.PHONY: help venv install build up down test train-xgb plot shell

help:
	@echo "Targets: venv, install, build, up, down, test, train-xgb, plot, shell"

venv:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -U pip

install: venv
	$(VENV)/bin/pip install -r requirements.txt

build:
	docker build -t dataiatry-api:latest .

up:
	docker-compose up --build -d

down:
	docker-compose down

test:
	PYTHONPATH=. $(VENV)/bin/pytest -q

train-xgb:
	PYTHONPATH=. $(VENV)/bin/python scripts/train_xgb_full.py

plot:
	$(VENV)/bin/python scripts/plot_comparison.py

shell:
	docker run --rm -it -v ${PWD}:/app dataiatry-runner:latest /bin/bash
PY=.venv/bin/python

.PHONY: build docker-build run hpo test validate clean

build:
	$(PY) -m pip install -r requirements.txt

docker-build:
	docker build -t dataiatry:latest .

run:
	$(PY) -m uvicorn api.app:app --host 0.0.0.0 --port 8001 --reload

hpo:
	PYTHONPATH=. $(PY) scripts/hpo_timeseries.py

hpo-quick:
	PYTHONPATH=. $(PY) scripts/hpo_timeseries_quick.py

test:
	PYTHONPATH=. $(PY) -m pytest -q

validate:
	$(PY) scripts/validate_submission.py

clean:
	rm -rf __pycache__ .pytest_cache logs
