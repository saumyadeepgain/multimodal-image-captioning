# Shortcuts for the common workflows. `make help` lists them.
# Override variables on the command line, e.g.:
#   make train ARGS="--epochs 30 --lr 1e-4"
#   make predict INPUT=photos/ ARGS="--format csv --output captions.csv"

PYTHON          ?= python
CONFIG          ?= configs/default.yaml
CKPT            ?= checkpoints/best_model.pth
LAST            ?= checkpoints/last.pth
INPUT           ?= data/archive/Images
SPLIT           ?= test
ARGS            ?=
IMAGE           ?= image-captioning
TORCH_CPU_INDEX ?= https://download.pytorch.org/whl/cpu
DOCKER_RUN      ?= docker run --rm -v "$(CURDIR)/data:/app/data" -v "$(CURDIR)/checkpoints:/app/checkpoints"

.DEFAULT_GOAL := help
.PHONY: help install install-cpu install-dev install-all test cov lint format check \
        train resume eval predict demo tensorboard report \
        docker-build docker-build-demo docker-train docker-eval docker-predict docker-demo clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

## ---------------------------------------------------------------- setup
install: ## Install pinned runtime deps + this package
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install --no-deps -e .

install-cpu: ## Same, with CPU-only PyTorch (smaller download)
	$(PYTHON) -m pip install -r requirements.txt --extra-index-url $(TORCH_CPU_INDEX)
	$(PYTHON) -m pip install --no-deps -e .

install-dev: ## Pinned runtime + test/lint tooling
	$(PYTHON) -m pip install -r requirements-dev.txt --extra-index-url $(TORCH_CPU_INDEX)
	$(PYTHON) -m pip install --no-deps -e .

install-all: ## Everything: demo, tracking, notebook, dev tools
	$(PYTHON) -m pip install -r requirements-optional.txt -r requirements-dev.txt
	$(PYTHON) -m pip install --no-deps -e .

## ---------------------------------------------------------------- quality
test: ## Run the test suite (CPU, synthetic data)
	$(PYTHON) -m pytest $(ARGS)

cov: ## Tests with a coverage report
	$(PYTHON) -m pytest --cov=src --cov-report=term-missing $(ARGS)

lint: ## ruff lint + format check (what CI runs)
	ruff check .
	ruff format --check .

format: ## Auto-format and auto-fix lint issues
	ruff format .
	ruff check --fix .

check: lint test ## Lint + tests

## ---------------------------------------------------------------- ML workflow
train: ## Train (ARGS="--epochs 30 --lr 1e-4 ...")
	$(PYTHON) -m src.train --config $(CONFIG) $(ARGS)

resume: ## Resume the last run from checkpoints/last.pth
	$(PYTHON) -m src.train --config checkpoints/config.yaml --resume $(LAST) $(ARGS)

eval: ## Evaluate CKPT on SPLIT (test by default)
	$(PYTHON) -m src.evaluate --checkpoint $(CKPT) --split $(SPLIT) $(ARGS)

predict: ## Caption INPUT (file or folder)
	$(PYTHON) -m src.predict $(INPUT) --checkpoint $(CKPT) $(ARGS)

demo: ## Launch the Gradio web demo
	$(PYTHON) app.py --checkpoint $(CKPT) $(ARGS)

tensorboard: ## Open TensorBoard on the tracked runs
	tensorboard --logdir checkpoints/tensorboard

report: ## Rebuild docs/IMPROVEMENTS.pdf
	$(PYTHON) docs/build_report.py

## ---------------------------------------------------------------- docker
docker-build: ## Build the CPU image
	docker build -t $(IMAGE) .

docker-build-demo: ## Build the image with the Gradio demo
	docker build --target demo -t $(IMAGE):demo .

docker-train: ## Train inside Docker (data/ and checkpoints/ are mounted)
	$(DOCKER_RUN) $(IMAGE) python -m src.train $(ARGS)

docker-eval: ## Evaluate inside Docker
	$(DOCKER_RUN) $(IMAGE) python -m src.evaluate --checkpoint $(CKPT) --split $(SPLIT) $(ARGS)

docker-predict: ## Caption INPUT (must be under data/) inside Docker
	$(DOCKER_RUN) $(IMAGE) python -m src.predict $(INPUT) --checkpoint $(CKPT) $(ARGS)

docker-demo: ## Serve the demo on http://localhost:7860
	$(DOCKER_RUN) -p 7860:7860 $(IMAGE):demo

## ---------------------------------------------------------------- housekeeping
clean: ## Remove caches and build artefacts (keeps data/ and checkpoints/)
	rm -rf .pytest_cache .ruff_cache .coverage coverage.xml pytest-report.xml htmlcov \
		build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
