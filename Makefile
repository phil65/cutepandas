.PHONY: help clean lint test docs serve release bump
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys
from urllib.request import pathname2url
webbrowser.open("file:" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define BUMP_SCRIPT
import os, cutepandas
version = cutepandas.__version__
os.system(f'cz changelog --unreleased-version "v{version}"')
endef
export BUMP_SCRIPT

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

clean: ## remove all build, test, coverage and Python artifacts
	git clean -dfX

test_pyside6: ## run tests with pyside
	export QT_API=pyside6; poetry run pytest # --mypy

test_pyqt6: ## run tests with pyside
	export QT_API=pyqt6; poetry run pytest # --mypy

test: test_pyside6 ## run tests with pyside6

mypy: ## run mypy type checking
	poetry run mypy cutepandas

docs: ## builds the documentation
	poetry run mkdocs build

serve: ## run html server watching file changes in realtime
	poetry run mkdocs serve

changelog: ## create changelog
	python -c "$$BUMP_SCRIPT"
	mv CHANGELOG.md docs/changelog.md

bump: ## version bump
	poetry run cz bump --no-verify
# 	cp CHANGELOG.md docs/changelog.md
