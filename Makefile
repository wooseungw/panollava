# PanoLLaVA Makefile
# ==================

.PHONY: help install install-dev install-all clean test lint format check docs build publish docker-build docker-run

# Default target
help: ## Show this help message
	@echo "PanoLLaVA Development Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation
install: ## Install PanoLLaVA in development mode
	pip install -e .

install-dev: ## Install with development dependencies
	pip install -e ".[dev]"

install-all: ## Install with all optional dependencies
	pip install -e ".[dev,web,notebook]"

install-pre-commit: ## Install pre-commit hooks
	pre-commit install

# Cleaning
clean: ## Clean build artifacts and cache files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf __pycache__/
	rm -rf src/panovlm/__pycache__/
	rm -rf src/panovlm/**/*.pyc
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete

clean-models: ## Clean downloaded models and checkpoints
	rm -rf results/checkpoints/
	rm -rf results/runs/
	rm -rf ~/.cache/huggingface/

clean-all: clean clean-models ## Clean everything

# Testing
test: ## Run all tests
	pytest tests/ -v

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	pytest tests/integration/ -v --tb=short

test-cov: ## Run tests with coverage
	pytest tests/ --cov=src/panovlm --cov-report=html --cov-report=term-missing

test-gpu: ## Run GPU-specific tests
	pytest tests/ -v -m gpu

# Code Quality
lint: ## Run all linters
	ruff check src/ tests/ scripts/
	flake8 src/ tests/ scripts/
	mypy src/panovlm/

lint-fix: ## Auto-fix linting issues
	ruff check --fix src/ tests/ scripts/
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

format: ## Format code with black and isort
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

check: ## Run all code quality checks
	$(MAKE) lint
	$(MAKE) test
	pre-commit run --all-files

# Documentation
docs: ## Build documentation
	@echo "Documentation build not yet implemented"

docs-serve: ## Serve documentation locally
	@echo "Documentation serve not yet implemented"

# Building & Publishing
build: ## Build distribution packages
	python -m build

publish-test: ## Publish to TestPyPI
	python -m twine upload --repository testpypi dist/*

publish: ## Publish to PyPI
	python -m twine upload dist/*

# Docker
docker-build: ## Build Docker image
	docker build -t panollava:latest .

docker-run: ## Run Docker container
	docker run --gpus all -it --rm panollava:latest

docker-dev: ## Run Docker container for development
	docker run --gpus all -it --rm -v $(PWD):/workspace panollava:latest /bin/bash

# Training & Evaluation
train-vision: ## Train vision encoder (stage 1)
	python scripts/train.py --config configs/experiments/vision_pretraining.yaml

train-resampler: ## Train resampler (stage 2)
	python scripts/train.py --config configs/experiments/resampler_training.yaml

train-finetune: ## Fine-tune the full model (stage 3)
	python scripts/train.py --config configs/experiments/instruction_tuning.yaml

train-lora: ## Fine-tune with LoRA
	python scripts/train.py --config configs/experiments/lora_finetune.yaml

eval: ## Run evaluation
	python scripts/eval.py --config configs/default.yaml

infer: ## Run inference example
	python scripts/simple_inference.py

# Data Processing
download-data: ## Download required datasets
	python scripts/download_raw.py

process-data: ## Process raw data
	@echo "Data processing scripts to be implemented"

# Development Utilities
setup-dev: ## Setup development environment
	$(MAKE) install-dev
	$(MAKE) install-pre-commit
	pre-commit install

update-deps: ## Update dependencies
	pip install --upgrade pip
	pip install --upgrade -e ".[dev]"

check-env: ## Check development environment
	python -c "import torch; print(f'PyTorch: {torch.__version__}')"
	python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
	python -c "import lightning; print(f'Lightning: {lightning.__version__}')"
	python -c "import panovlm; print('PanoLLaVA import successful')"

# CI/CD
ci-test: ## Run CI test suite
	$(MAKE) clean
	$(MAKE) check
	$(MAKE) test-cov

ci-build: ## Run CI build
	$(MAKE) clean
	$(MAKE) build

# Quick commands for common tasks
quick-test: ## Quick test to verify installation
	python -c "import panovlm; print('✓ PanoLLaVA installed successfully')"

quick-train: ## Quick training test
	python scripts/train.py --config configs/quick_test.yaml

# Help for specific stages
help-train: ## Show training help
	@echo "Training Stages:"
	@echo "  1. make train-vision    - Train vision encoder with VICReg"
	@echo "  2. make train-resampler - Train resampler module"
	@echo "  3. make train-finetune  - Fine-tune full model"
	@echo "  4. make train-lora      - Fine-tune with LoRA (parameter-efficient)"
	@echo ""
	@echo "Configuration files are in configs/experiments/"

help-docker: ## Show Docker help
	@echo "Docker Commands:"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run container with GPU support"
	@echo "  make docker-dev    - Run container with volume mount for development"