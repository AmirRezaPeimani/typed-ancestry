PYTHON ?= python3
MPLCONFIGDIR ?= .cache/matplotlib

.PHONY: test figures reproduce-light verify manifest

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

figures:
	MPLCONFIGDIR=$(MPLCONFIGDIR) $(PYTHON) analysis/figures/learning_figures.py
	MPLCONFIGDIR=$(MPLCONFIGDIR) $(PYTHON) analysis/figures/cross_learner_figure.py
	MPLCONFIGDIR=$(MPLCONFIGDIR) $(PYTHON) analysis/figures/candidate_selection_figures.py

verify:
	$(PYTHON) analysis/verify_controlled_release.py

manifest:
	$(PYTHON) analysis/build_artifact_manifest.py \
		--root . \
		--output manifests/artifact_manifest.json

reproduce-light:
	$(PYTHON) analysis/write_controlled_macros.py
	$(MAKE) PYTHON=$(PYTHON) figures
	$(PYTHON) analysis/build_evidence_registry.py
	$(MAKE) PYTHON=$(PYTHON) verify
	$(MAKE) PYTHON=$(PYTHON) manifest
