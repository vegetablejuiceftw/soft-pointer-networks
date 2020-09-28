# Get root dir and project dir
PROJECT_ROOT ?= $(CURDIR)

.PHONY:
lint: black-check-all prospector isort

.PHONY:
lint-fix: black-format-all isort-fix yapf flynt autoflake prospector

.PHONY:
black-check-all:
	@black --check .

.PHONY:
black-format-all:
	@black .

.PHONY:
isort:
	@isort --recursive --check-only -p . --diff

.PHONY:
isort-fix:
	@isort -y --recursive -p .

.PHONY:
prospector:
	@prospector

.PHONY:
yapf:
	@yapf --in-place --recursive --style="{indent_width: 4}" *.py

.PHONY:
flynt:
	@flynt .

.PHONY:
autoflake:
	@autoflake --in-place --remove-unused-variables .
