"""`openlab bench`: benchmarks of openlab itself, run by the person (openlab.benchmark.mle: MLE-bench)."""

import sys


def main(argv) -> int:
    if argv[:1] == ["mle"]:
        from .mle import main as mle
        return mle(argv[1:])
    print("usage: openlab bench mle {doctor,prepare,job,status,grade,ab,report} ...  (openlab bench mle --help)",
          file=sys.stderr)
    return 2
