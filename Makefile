# Local build targets. Nothing here publishes anything.
.PHONY: test test-changed docs bench clean-install

test:            ## the full suite (about 9 minutes)
	pytest -q

test-changed:    ## only the tests the changed code reaches (pytest-testmon; the first run seeds .testmondata)
	pytest -q --testmon

docs:            ## build the docs site into site/ from the README
	python3 docs/build_docs.py && mkdocs build -q

bench:           ## what the structure catches, from study records (needs bwrap): traps, traceability, reproducibility
	python3 -m bench.harness

clean-install:   ## prove the package installs on a clean machine and runs a study
	scripts/fresh_install_check.sh
