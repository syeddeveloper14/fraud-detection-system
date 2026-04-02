CONFIG ?= configs/default.yaml

.PHONY: install lint format test train replay serve docker clean

install:
	pip install -e ".[dev,lightgbm]"

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff format src tests
	ruff check --fix src tests

test:
