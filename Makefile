#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PYTHON_INTERPRETER = python3


#################################################################################
# COMMANDS                                                                      #
#################################################################################

.PHONY:
## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip poetry==1.1.* setuptools wheel -q
	poetry install

.PHONY:
## Export python requirement.txt
export-requirements:
	poetry export --without-hashes -f requirements.txt --output requirements.txt

.PHONY:
## Lock new Python Dependencies
lock-requirements:
	poetry lock --no-update
	@make export-requirements


.PHONY:
## load env variables form .env
env:
	export $$(grep -v '#.*' .env | xargs)


.PHONY:
## Create the poetry PyPi repository credentials
pypi-config:
	. ./.env && poetry config repositories.snackable $${PRIVATE_PYPI_URL}
	. ./.env && poetry config http-basic.snackable $${PRIVATE_PYPI_USERNAME} $${PRIVATE_PYPI_PASSWORD}


.PHONY:
## Reset the poetry PyPi repository credentials
pypi-config-unset:
	poetry config --unset http-basic.snackable
	poetry config --unset repositories.snackable


.PHONY:
## Build the python package filea
build:
	poetry build


.PHONY:
## Publish the python package file to our python package repository
## Builds the package first to check validity, then bumps the 'patch' version and rebuilds
## Make sure that you have credentials configured, or use `make pypi-config`
publish:
	@make build
	poetry version patch
	git commit -m $$(grep pyproject.toml -e '(?<=^version = ")(.*)(?=")' -Po) pyproject.toml
	git tag $$(grep pyproject.toml -e '(?<=^version = ")(.*)(?=")' -Po)
	poetry publish --build --repository snackable


.PHONY:
## Creates the release-*.*.* tag for production deployment. Then pushes it.
release:
	git tag release-$$(grep pyproject.toml -e '(?<=^version = ")(.*)(?=")' -Po)
	git push --tags


.PHONY:
## Start Jupyter with Colab support
## alternative is to just run "jupyter-lab"
jupyter:
	jupyter-lab \
  --port=8888 \
  --ip=0.0.0.0


.PHONY:
## Start Jupyter with Colab support
## alternative is to just run "jupyter-lab"
colab-jupyter:
	jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --port=8888 \
  --NotebookApp.port_retries=0


.PHONY:
## Fix quality problems greedily
## TODO: The result does depend on the order of execution and the config could be improved
quality:
	@make clean-cache
	@make docformatter trim
	@make flynt
	@make pyupgrade
	@make autoflake autopep
	@make yapf isort-fix
	@make prospector

.PHONY:
## Run tests
test:
	@pytest -p no:cacheprovider

.PHONY:
## Clear poetry cache
clear-poetry-cache:
	@poetry cache clear . --all

#################################################################################
#  Quality commands                                                             #
#################################################################################

.PHONY:
## The uncompromising Python code formatter
## https://github.com/psf/black
black-format-all:
	@echo "Running Black:"
	@black --line-length 120 .

.PHONY:
## A Python utility / library to sort imports.
## https://github.com/PyCQA/isort
isort-fix:
	@isort -y --recursive -p $$(find . -name '*.py')

.PHONY:
## Inspects Python source files and provides information about type and location of classes, methods etc
## https://github.com/PyCQA/prospector
prospector:
	@echo "Running prospector:"
	@prospector $$(find . -name '*.py')

.PHONY:
## A formatter for Python files
## https://github.com/google/yapf
yapf:
	@echo "Running yapf:"
	@yapf --style setup.cfg --in-place --recursive $$(find . -name '*.py')

.PHONY:
## A tool to automatically convert old string literal formatting to f-strings
## https://github.com/ikamensh/flynt
flynt:
	@flynt .

.PHONY:
## Removes unused imports and unused variables as reported by pyflakes
## https://github.com/myint/autoflake
autoflake:
	@autoflake -r --in-place --remove-unused-variables --remove-all-unused-imports --remove-duplicate-keys --ignore-init-module-imports .

.PHONY:
## A tool that automatically formats Python code to conform to the PEP 8 style guide.
## https://github.com/hhatto/autopep8
autopep:
	@autopep8 --in-place --aggressive --aggressive --max-line-length 120 -j 4 -r .

.PHONY:
## Formats docstrings to follow PEP 257
## https://github.com/myint/docformatter
docformatter:
	@docformatter --in-place --wrap-summaries 120 --wrap-descriptions 120  --blank -r .

.PHONY:
## Find dead Python code
## https://github.com/jendrikseipp/vulture
vulture:
	@vulture .

.PHONY:
## cleanpy is a CLI tool to remove caches and temporary files that related to Python.
## https://github.com/thombashi/cleanpy
clean-cache:
	@cleanpy -avf --exclude-envs .

.PHONY:
## Clean .py and .ipynb source files using black, isort and autoflake.
## https://github.com/samhardyhey/clean-py
clean-notebook:
	@clean_py notebooks

.PHONY:
## A tool (and pre-commit hook) to automatically add trailing commas to calls and literals.
## https://github.com/asottile/add-trailing-comma
trailing-comma:
	@add-trailing-comma --exit-zero-even-if-changed --py36-plus $$(find . -name '*.py')

.PHONY:
## It's not just a linter that annoys you!
## https://github.com/PyCQA/pylint
## https://github.com/PyCQA/pylint/issues/4081
pylint:
	@pylint -j 8 $$(find . -name '*.py')

.PHONY:
## Trims trailing whitespace from files
## https://github.com/myint/trim
trim:
	@trim $$(find . -name '*.*' -not -path './.*'  -not -path './data/*')

.PHONY:
## Modifies strings to all use the same quote where possible
## https://github.com/myint/unify
unify:
	@unify --in-place -r .

.PHONY:
## A tool (and pre-commit hook) to automatically upgrade syntax for newer versions of the language.
## https://github.com/asottile/pyupgrade
pyupgrade:
	@pyupgrade --exit-zero-even-if-changed --py38-plus $$(find . -name '*.py')


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available Makefile commands:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=24 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
