.PHONY: help install install-dev test clean lint format type-check run-experiment

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \\033[36m%-20s\\033[0m %s\\n", $$1, $$2}'

install: ## Install the package and dependencies
	pip install -e .

install-dev: ## Install the package with development dependencies
	pip install -e ".[dev]"

test: ## Run tests
	python -m pytest tests/ -v --cov=src --cov-report=html

clean: ## Clean build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

lint: ## Run linting
	flake8 src/ scripts/ tests/
	black --check src/ scripts/ tests/

format: ## Format code
	black src/ scripts/ tests/

type-check: ## Run type checking
	mypy src/

setup-dirs: ## Create project directory structure
	mkdir -p data/raw data/processed data/models data/results/figures logs

download-data: ## Download RAVDESS dataset
	python -c "import kagglehub; kagglehub.dataset_download('uwrfkaggler/ravdess-emotional-speech-audio')"

run-experiment: ## Run complete experiment
	python scripts/run_experiment.py --config config/config.yaml --download --optimize --visualize --save-model

run-quick: ## Run quick experiment without optimization
	python scripts/run_experiment.py --config config/config.yaml --no-optimize

evaluate-only: ## Run evaluation only (requires trained model)
	python scripts/evaluate_model.py --config config/config.yaml --model-dir data/models/