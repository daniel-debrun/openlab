Draft the replication plan of the paper in {folder} ({key}). The person gave only the paper: every
choice below is yours to make from the paper, and a person approves the plan before anything is built.

1. Read the paper: {folder}/paper/paper.pdf, or its text page by page in {folder}/paper/pages.txt (`=== page N ===`
   lines; tables keep their rows there). Read the appendices too: baselines, hyperparameters and protocols are
   often defined only there.
2. Fill {folder}/replication.toml (each field says what goes in it):
   - title, summary, kind (replication unless the authors released code to run: reproduction, with `code`), mode:
     temporal when the paper evaluates forecasts made at successive origins over time (walk-forward, rolling or
     sliding-window test sets, a model or archive updated as the test period unfolds): openlab then gives the model
     only what is known at each decision time. static when the rows are exchangeable (a fixed random split, k-fold,
     tabular data with no time order the evaluation depends on).
   - methods: the paper's own method (role proposed; several when the paper proposes a family, e.g. a different
     selected model per dataset: the first is the model, each one's numbers become claims) and the baselines its
     headline table compares against, each with a class name, its label as printed, the page that defines it and a one-line description.
   - metric: the paper's headline metric, one of: {metrics}. If it uses none of these, define it as the study's own:
     set metric.better, reads and description, and write def <name>(y, pred) -> float in {folder}/metrics.py (y and
     pred are numpy arrays, 2-D when a label has several outputs). That file is the one other file you may write.
   - tolerance: how close our number must be to the paper's, in the metric's unit, from the paper: the spread it
     reports over seeds or folds, the rounding of its tables, the size of its test set. Say why in the
     justification; quote the page if the paper states the spread.
   - data: where it comes from, its licence, and the protocol (split, evaluation set, preprocessing) as stated.
   - claims: the numbers to check, from the headline table: each baseline's value (it shows the rebuild faithful)
     and the proposed method's. Copy each quote from paper/pages.txt exactly: the table row with the method's label
     and the number, as one line. When the methods are the table's columns (the row is a dataset or a setting), add
     `header`: the header line that shows the method's label, as one line of pages.txt; openlab checks the number
     sits under that column. A header of several levels (a group label over sub-columns): give header as a list of
     its lines, and the method's label by level, "SC2 / γ=1−α". 3 to 8 claims is plenty.
   - budget for the build (at most ${cap_usd:g} and {cap_minutes:g} min), with its compute: `runs` (the counted runs
     the claims need), `run_minutes` (one run's wall minutes here) and `estimate` (how you got it). This machine
     has {cores} cores and {memory} of memory. Scale the paper's reported timings (a timing table, its hardware,
     epochs) to it; with none, reason from the data size and the methods' cost. Lint refuses a plan whose runs do
     not fit {fit:.0%} of its minutes: then plan fewer seeds or settings, and say so in the tolerance and unresolved.
   - risks, and unresolved: what the paper leaves open that the person should know.
3. Run `openlab replicate lint {folder}` and fix what it refuses, until it is clean. It checks every quote is on its
   page and writes its number.
4. Stop there. Do not approve, build or create any other file (metrics.py aside).
Finish with four lines: the methods, the claims, the metric and tolerance (with the reason), and what is unresolved.
