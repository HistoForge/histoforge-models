# Contributing

Use Python 3.10+ and install the development dependencies:

```bash
python -m pip install -e ".[dinov3,training,dev]"
pytest -q
```

The default suite uses synthetic inputs and does not download model weights.
Cached-model integration requires `--runlegacy`; slow tests require `--runslow`.
Some parity tests also need an external reference implementation. A skipped
test is not a passed compatibility or model-quality check.

Add focused regressions for changed contracts. Preserve checkpoint keys, class
ordering, coordinate conventions and documented metric reductions unless the
change explicitly revises them. Upstream-owned function tests detect semantic
changes separately from formatting and docstrings.

Keep datasets, checkpoints, credentials and generated research outputs out of
Git. Use small synthetic fixtures for ordinary tests. Retain applicable upstream
copyright and licence notices.
