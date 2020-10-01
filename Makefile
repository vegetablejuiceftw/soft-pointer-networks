# Get root dir and project dir
PROJECT_ROOT ?= $(CURDIR)

.PHONY:
quality: black-check-all prospector isort

.PHONY:
lint-fix:
	@make clean-cache
	@make clean-notebook
	@make docformatter trim unify
	@make flynt trailing-comma
	@make pyupgrade
	@make isort-fix autoflake autopep yapf
	@make black-format-all
	@make prospector

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
#	@prospector --without-tool pylint

.PHONY:
yapf:
	@yapf --in-place --recursive --style="{indent_width: 4}" *.py

.PHONY:
flynt:
	@flynt .

.PHONY:
autoflake:
	@autoflake -r --in-place --remove-unused-variables --remove-all-unused-imports --remove-duplicate-keys .

.PHONY:
autopep:
	@autopep8 --in-place --aggressive --aggressive --max-line-length 120 -j 4 -r .

.PHONY:
docformatter:
	@docformatter --in-place --wrap-summaries 120 --wrap-descriptions 120  --blank -r .

.PHONY:
vulture:
	@vulture .

.PHONY:
clean-cache:
	@cleanpy -av --exclude-envs .

.PHONY:
clean-notebook:
	@cleanpy -av --exclude-envs .

.PHONY:
trailing-comma:
	@add-trailing-comma --py36-plus $$(find . -name '*.py')

.PHONY:
pylint:
	@pylint -j 8 .

.PHONY:
trim:
	@trim .

.PHONY:
unify:
	@unify --in-place -r .

.PHONY:
pyupgrade:
	@pyupgrade --exit-zero-even-if-changed --py38-plus $$(find . -name '*.py')
