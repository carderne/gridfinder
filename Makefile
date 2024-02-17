.PHONY: lint
lint:
	ruff format .
	ruff check .

.PHONY: test
test:
	pytest
