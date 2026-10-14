# Tests

    python3 -m pytest tests -q -p no:cacheprovider > /tmp/t.out 2>&1; rc=$?; tail -3 /tmp/t.out; echo rc=$rc

Read the exit code, not the tail of the output. The full suite takes about twelve minutes; the app, holdout
and model suites (`test_app_*`, `test_holdout.py`, `test_models.py`, `test_doctor.py`) run in under a minute.
For one change, run `make test-changed` (pytest-testmon: only the tests whose code the change touches); the
full suite is for before a release. `conftest.py` turns the module cache off; app tests run sessions in-process (`OPENLAB_MCP_INPROCESS=1`).
