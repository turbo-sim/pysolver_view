# Compatibility shortcuts; build_docs.py owns the documentation workflow.
PYTHON ?= poetry run python

.PHONY: help html livehtml clean

help:
	@$(PYTHON) build_docs.py --help

html:
	@$(PYTHON) build_docs.py --no-autobuild

livehtml:
	@$(PYTHON) build_docs.py

clean:
	@$(PYTHON) build_docs.py --clean-only
