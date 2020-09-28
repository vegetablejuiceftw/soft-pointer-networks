# Get root dir and project dir
PROJECT_ROOT ?= $(CURDIR)

.PHONY:
lint: black-check-all prospector isort

.PHONY:
lint-fix: isort-fix black-format-all yapf flynt autoflake prospector

.PHONY:
black-check-all:
	@black --check .

.PHONY:
black-format-all:
	@black --line-length 120 .

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
	@autoflake -r --in-place --remove-unused-variables --remove-all-unused-imports --remove-duplicate-keys .
