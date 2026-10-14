"""The research -> paper builder: the agent's prose in paper/sections/*.tex, with binding macros, plus the sections
openlab generates from the record, built to paper/build/main.pdf, recorded as a Paper node and signed.

`build.py` runs it (build, lint, verify); `bindings.py` resolves the macros and runs the equation gate; `lint.py`
holds the lints, gates and waivers; `sections.py` writes the LaTeX. Kept empty so a sandboxed child can import
`bindings` alone.
"""
