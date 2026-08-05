# One-command workflows. Run `make help` to list them.
.PHONY: help install train eval test lint demo serve mlflow monitor clean

help:
	@echo "install  - install package with all extras"
	@echo "train    - train both models"
	@echo "eval     - run the full evaluation report"
	@echo "test     - run the test suite"
	@echo "lint     - run ruff"
	@echo "demo     - run the CLI demo"
	@echo "serve    - start the FastAPI service"
	@echo "mlflow   - train with MLflow tracking"
	@echo "monitor  - start the Docker monitoring stack"
	@echo "clean    - remove caches and build artifacts"

install:
	pip install -e ".[dev,app,mcp,explain,serve,mlops,notebook]"

train:
	python -m scripts.train_model
	python -m scripts.train_temporal

eval:
	python -m scripts.evaluate_system

test:
	pytest -q

lint:
	ruff check src tests scripts

demo:
	python -m scripts.cli_demo

serve:
	uvicorn jetpilotguard.io.service:app --reload

mlflow:
	python -m scripts.train_with_mlflow

monitor:
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
