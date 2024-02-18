.PHONY: lint
lint:
	ruff format .
	ruff check .

.PHONY: check
check:
	pyright

.PHONY: test
test:
	pytest
	./examples/test.sh
