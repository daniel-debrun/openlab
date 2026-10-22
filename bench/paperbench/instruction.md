# Task: replicate a research paper

The paper is in `./paper`: `paper.pdf` and `paper.md`, `addendum.md` (the authors' clarifications and what is out of
scope), `blacklist.txt` and `config.yaml` (its `id` is the paper's id). Read the addendum before you plan.

## What to replicate

Replicate as many of the paper's core contributions as you can in the time and budget you have; you will not finish
everything, so put the most important results first. An experiment only in the appendix is out of scope; one
described in the main text is in scope even when some of its details are in the appendix. Credit is partial and
weighted by importance: implementing a method counts, running it counts, and results that follow the paper's trends
within a reasonable margin count. Exact numbers are not expected.

You may use the open web, but never the paper's own code or anything `blacklist.txt` lists.

## What you hand in

A git repository at `./submission` (it is already initialised). Only what is committed is handed in:

- `reproduce.sh` at its root runs everything. It will be run as `bash reproduce.sh` from the repository's root, in a
  fresh Ubuntu 24.04 container with one NVIDIA A10 GPU, for at most 7 days, possibly at another path: no absolute
  paths, and it installs what it needs.
- Every output to be judged is made by `reproduce.sh`, not committed by hand.
- `README.md` at its root says what you achieved, how the code maps onto the paper, and what `reproduce.sh` should
  produce.
- Committed files total under 1 GB: commit code, and let `reproduce.sh` fetch or make anything heavy.

Test `reproduce.sh` as you go, on a short setting.

## How to hand it in

Through the `paperbench_submit` instrument, never by copying files yourself. Check first (nothing is copied):

    openlab_tool(name="instrument_run", args={"ref": "paperbench_submit",
                 "args": ["check", "--paper", "<id>", "<absolute path of ./submission>"]})

then submit with `"submit"` in place of `"check"`. It hands in your committed HEAD and says what it refused or
warned about; fix those and submit again. You may submit as often as you like: the last submission is the one
judged, so submit a working version early and again as it improves. The judging happens after you finish, and its
result is not shown to you.
