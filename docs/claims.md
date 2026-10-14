# Claims, tests and the paper

A finding is a claim on the record, and a claim is a sentence the paper may print. Openlab computes its verdict and
its level from what it rests on, so neither can be typed.

```
$ openlab study claim demo "The x-only model has lower MAE than climatology." --state observed --rests-on result:a6041e0e2db0
claim:2ee3feda334b  observed, supported, recorded, selected from 2 trials: The x-only model has lower MAE than climatology.
```

## States

| State | Example | Rests on |
|---|---|---|
| `observed` | "the paper reports a CRPS of 0.88" | evidence (a quote checked on its page) or a result |
| `derived` | "EMOS beats the raw ensemble by 0.45" | claims, results or evidence, through `--expression '{a} - {b}'` with `--inputs '{"a": "result:…#metrics.crps.value", ...}'`; openlab computes the value |
| `inferred` | "the gain comes from the spread correction" | other claims; the paper checks its wording |
| `hypothesized` | "deeper networks will be more robust" | one hypothesis, with its prediction and what would falsify it |
| `unknown` | "untested outside Europe" | the coverage of a recorded search |

A number is never typed. An observed value is read from its one input; a derived value is computed and checked
within a tolerance. A result whose run's truncation audit is not clean cannot ground a claim, and neither can
evidence from a retracted source unless the statement mentions the retraction or a person waives it. A claim that
rests on one of several trials says "selected from N trials", and `study runs` lists all N.

## Verdicts and levels

The verdict is one of `supported`, `refuted`, `inconclusive` or `untested`. Tests decide when there are any: the
claim's own tests, the tests of the hypothesis it tests, and for an inferred claim the tests of the hypotheses it
answers. Without tests, an observed claim is supported by its checked ground, a derived claim takes its weakest
premise's verdict, and the other states are untested.

The level comes from the results a claim rests on: `measured` if every one is from the holdout, else `recorded`, plus
`+reproduced` once each of their runs has a matched reproduction. `openlab study claims demo` shows each claim's
verdict and level now, along with any broken assumption it needs.

## Comparing variants

```
$ openlab study compare demo result:eafbcfd859d6 result:266f11605121 --metric mae
comparison:452c4d7968c9  mae 0.4222 vs 0.4231: better (protocol same sealed data, target and split, rows 1340); a - b -0.000878 [-0.001857, -2.505e-05], p 0.04595
  1340 shared rows (block_bootstrap); better: result:eafbcfd859d6
```

Two results of one study are paired on their shared rows. `study test ... --baseline result:B` tests a result against
another result instead of a declared baseline, which is how an ablation is tested.

## Tests and the alpha budget

```
$ openlab study test demo claim:2ee3feda334b --result result:a6041e0e2db0 --baseline climatology --alpha 0.02
test:40324b1c99e6  primary: p 0.001998 at alpha 0.02 vs climatology, rejected; alpha earned 0.05, alpha budget left: 0.0975
```

A test reads the p-value of a result's paired difference against a baseline. It is supported when it rejects with
the model better, refuted when it rejects with the baseline better, and inconclusive otherwise. Each test spends
some of the alpha budget, the study's error budget (alpha-investing, Foster and Stine): `alpha` is at most
`budget / (1 + budget)`, and a rejection earns some back. Once the budget is spent, the next test is refused, so an
unattended loop cannot test its way to p < 0.05. `[inference]` in `study.toml` sets it, and only a person amends it.

## Organising the research

- **Questions and hypotheses.** `study question` and `study hypothesis` build the question tree. `study tree` derives
  each node's status (answered, refuted, inconclusive, untested, out of scope) from its children; a status is never
  stored as text.
- **Assumptions.** `study assume` records what claims may `--needs`. `study break` records that one failed, and every
  claim that needs it is at risk.
- **Dead ends.** `study dead-end` keeps a falsified hypothesis or abandoned run with its reason. A new hypothesis with
  a shared tag must link it and say what is different now, and `openlab watch` flags an agent working on one.
- **Contradictions.** `study contra` records a search for disagreeing evidence, which ends `found`, `none_found` or
  `insufficient` (below the review's coverage floor).
- **What to do next.** `study frontier` lists untested hypotheses, claims at risk, inconclusive tests, gaps in the
  evidence table and a stale paper, ranked by how much each step should tell you for the time it costs. `study take`
  leases one item to you, so two agents do not do the same thing.
- **Traceability.** `study trace` shows a claim's chain down to the source page or the dataset; `study traceability`
  is the share of claims whose chain is complete. The paper build refuses a bound claim whose chain is incomplete.

## The paper

You write the prose in `paper/sections/*.tex` around bindings to the record:

| Binding | Inserts |
|---|---|
| `\lab{<id>#<path>}` | a number or value from a node; `\lab[ablation]{...}` binds an earlier variant in an ablation table |
| `\labclaim{<claim>}` | the claim's statement, with its state and verdict |
| `\labtab{<result>}`, `\labfig{<result>}` | a table, a figure |
| `\labcite{<source>}{<quote>}` | a citation whose quote must be in the source |
| `\labeq{<name>}{<module.fn>}{<equation>}` | an equation, checked against your Python function on probe inputs |

Openlab generates the data, methods, literature, results, limitations and reproducibility sections from the record.
`openlab study paper-lint demo` lists the findings without building; `openlab study paper demo` refuses to build while
a binding is unresolved or stale, a number in the prose is not bound, the wording outruns its claim ("significant"
on an inconclusive test, "novel" anywhere), a cited quote is not in its source, an equation disagrees with its
function, or a bound claim's chain is incomplete. Only a person can waive a finding (`openlab study approve demo ID
--decision waive --reason ...`), and the paper prints every waiver. A retracted citation is printed as retracted.

## What you hand on, signed

- `openlab study paper demo` writes `paper/build/main.pdf`, a `paper` node, and a signed manifest (the record's head,
  the bindings, the figures). `openlab study verify-paper demo` checks the signature, the PDF's sha256, the hash
  chain, and that no source the paper rests on was retracted after the build.
- `openlab study bundle demo out.tar` signs a tar of the record, the data and the code; `openlab study reproduce
  --bundle out.tar` reruns every run from it in a temporary directory.
- `openlab study reproduce demo` reruns the latest run (`--all`: every run) from the record alone and records whether
  it matched.

These files are signed with the workspace's signing key (under `~/.local/share/openlab/keys`, or
`$OPENLAB_SIGNING_KEY`), which is separate from your person key. A key id is only as trustworthy as the place you got
it, so publish yours (`openlab doctor` shows it) where your readers can compare it.
