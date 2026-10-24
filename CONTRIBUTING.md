# Contributing

Thank you for helping. Openlab is alpha, so an issue that describes what you tried, what you expected and what openlab
printed is as useful as a patch.

## Set up

With uv:

```
git clone https://github.com/daniel-debrun/openlab
cd openlab
uv sync --extra study --extra test
```

Or with pip, in a virtual environment:

```
pip install -e ".[study,test]"
```

The sandbox needs Linux with bubblewrap (`apt install bubblewrap`). Without it, runs are recorded as
`unsandboxed`, and some tests that run under bwrap are skipped. `openlab doctor` says what is missing.

## Run the tests

```
make test            # the full suite, pytest -q (about 9 minutes)
make test-changed    # only the tests the changed code reaches (pytest-testmon)
pytest -q tests/test_study_paper.py      # one file
```

With uv, `uv run --extra study --extra test pytest -q`. CI runs the suite on Python 3.10 to 3.13, a lint for
undefined names and syntax (`uvx ruff check --select E9,F63,F7,F82 openlab bench tests`), and a study end to end.

`make docs` builds the docs site (`docs/index.md` is generated from this repository's README by
`docs/build_docs.py`; edit the README, not `docs/index.md`). `make bench` runs the benchmark harness (needs
bwrap).

## Changes

- Keep a change small and give it a test.
- Commit messages are short and lowercase, and say what changed: `show: escape every agent-controlled value`.
- Docs use plain, short, declarative sentences.
- A refusal says what was refused, what it means and what to do instead (error, meaning, fix).

## Security

Report a vulnerability privately, not in an issue; see [SECURITY.md](SECURITY.md).

## License

By contributing, you agree that your contribution is licensed under the Apache License 2.0 ([LICENSE](LICENSE)), as section 5 of that license provides.
