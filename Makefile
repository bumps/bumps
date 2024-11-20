ROOTDIR = $(shell pwd)

# This nifty perl one-liner collects all comments headed by the double "#" symbols next to each target and recycles them as comments
.PHONY: help
help: ## Print this help message
	@perl -nle'print $& if m{^[/a-zA-Z_-]+:.*?## .*$$}' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'

.PHONY: clean
clean: ## Delete some cruft from builds/testing/etc.
	rm -f `find . -type f -name '*.py[co]'`
	rm -rf `find . -name __pycache__ -o -name "*.egg-info"` \
		`find . -name 'output-*'` \
		.coverage build dist \
		doc/_build doc/api doc/tutorial \
		.pytest_cache \
		.ruff_cache

.PHONY: test
test: ## Run pytest and doc tests
	pytest -v
	python check_examples.py --chisq
	python check_fitters.py

##############################
### Linting and formatting ###
##############################

.PHONY: lint
lint: lint-backend lint-frontend ## Run all linters

.PHONY: lint-backend-check
lint-backend-check: ## Run ruff linting on python code
	@ruff check bumps/ run.py test.py check_*.py

.PHONY: lint-backend
lint-backend: ## Run ruff linting fix on python code
	@ruff check --fix bumps/ run.py test.py check_*.py

.PHONY: lint-frontend-check
lint-frontend-check: ## Run bun linting check on javascript code
	cd bumps/webview/client && \
		$(FE_CMD) run test:lint

.PHONY: lint-frontend
lint-frontend: ## Run bun linting fix on javascript code
	cd bumps/webview/client && \
		$(FE_CMD) run lint

.PHONY: format
format: format-backend format-frontend ## Run all formatters

.PHONY: format-backend
format-backend: ## Run ruff formatting on python code
	@ruff format bumps/ run.py test.py check_*.py

.PHONY: format-frontend
format-frontend: ## Run bun formatting on javascript code
	cd bumps/webview/client && \
		$(FE_CMD) run format
