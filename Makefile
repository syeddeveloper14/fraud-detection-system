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
	pytest

train:
	fraud-detect train --config $(CONFIG)

replay:
	fraud-detect replay --config $(CONFIG) --limit 20000 --check-parity

serve:
	fraud-detect serve --config $(CONFIG)

docker:
	docker compose up --build

clean:
	rm -rf artifacts reports .pytest_cache .ruff_cache build dist *.egg-info
	find . -name __pycache__ -type d -exec rm -rf {} +
