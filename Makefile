PY?= ./.venv/bin/python
PIP?= ./.venv/bin/pip

.PHONY: help build up down test train-xgb plot docs

help:
	@echo "Targets: build up down test train-xgb plot"

build:
	docker build -t dataiatry:latest .

up:
	docker-compose up --build

down:
	docker-compose down

test:
	PYTHONPATH=. $(PY?) -m pytest -q

train-xgb:
	PYTHONPATH=. $(PY?) scripts/train_xgb_full.py

plot:
	$(PY?) scripts/plot_comparison.py

docs:
	@echo "See README.md for usage"
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
