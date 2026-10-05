.PHONY: clean clean-test clean-pyc clean-build dist docs help install
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint: ## check formatting and style with Ruff, as in CI
	ruff format --check .
	ruff check .

test: ## run tests with the default Python (coverage options come from pyproject.toml)
	pytest

coverage: ## run tests and write an HTML coverage report
	pytest --cov-report=html
	$(BROWSER) htmlcov/index.html

docs: ## build the Sphinx HTML documentation with warnings as errors, as in CI
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m sphinx \
		-W --keep-going -b html docs docs/_build/html
	$(BROWSER) docs/_build/html/index.html

dist: clean-build ## build and validate the source and wheel distributions
	python -m build
	python -m twine check --strict dist/*
	ls -l dist

install: ## install the package to the active Python's site-packages
	python -m pip install .
