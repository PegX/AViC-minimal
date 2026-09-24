# AViC

Attention analysis for graph-enhanced multimedia research. This minimal release provides an executable, offline synthetic example of one part of the research system. It is **not a complete paper artifact or benchmark reproduction**.

## What runs

Synthetic node features → attention cache → head ablation → output-distance report → restore baseline.

Included: Circuit identifiers + instrumented attention model. The command prints a structured JSON report; `--output` writes the same report to a file. Inputs are built into the demonstration, so no datasets, credentials, GPU, API requests or model downloads are needed at runtime.

## Quick start

Python 3.10+; run from this repository directory:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
avic-demo
avic-demo --output reports/demo.json
python -m pytest -q
```

Dependency installation requires access to a Python package index. A representative output is checked in at `examples/demo-output.json`. Floating-point results may vary slightly by PyTorch/platform version. The example is reproducible within the tested environment; dependencies are declared but not locked.

## Layout

- `avic_gatn/`: extracted components and the new `demo.py` entry point.
- `tests/`: component checks and repeatability checks.
- `examples/demo-output.json`: synthetic output from the installed command.
- `PROVENANCE.md` and `SOURCE-MANIFEST.json`: source mapping and modifications.
- `pyproject.toml`: dependencies, package discovery and CLI installation.

## Scope and extension

The entry point exercises only the instrumented attention component. Output L2 change is sensitivity, not accuracy, causal proof or an attack success rate. Real multimedia encoders, data and the full GATN scoring path are not tested here. The included gatn_min.py credits upstream GATN; attribution is retained, but upstream licensing must be checked before public redistribution.

No private corpora, weights, checkpoints, experiment outputs, original Git history, third-party repository copies, credentials, attack-generation runners or APK modification/deployment pipelines are included. These examples analyze synthetic model or program components only. They do not establish the security of real systems.

## License and attribution

No open-source license has been selected. Confirm ownership, collaborators' publication rights and third-party obligations before making a public release; then add the appropriate LICENSE and required notices. This preparation does not claim license clearance. See `PROVENANCE.md` for details.
