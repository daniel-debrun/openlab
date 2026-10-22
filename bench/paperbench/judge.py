"""Grade a PaperBench submission with Claude as the judge: PaperBench's own SimpleJudge, prompts and rubric, with its
three model calls (choosing files, grading a criterion, parsing the verdict) made through the Anthropic SDK.

Code-Dev by default (PaperBench's code_only): only the Code Development criteria are judged. --full judges the whole
rubric, for a submission folder in which reproduce.sh has been run as PaperBench runs it (`bash reproduce.sh 2>&1 | tee
reproduce.log`, with reproduce.log.creation_time beside it). Run it in PaperBench's environment, as the person; the
grade goes to --out and never to an agent:

    uv run --project $PB python bench/paperbench/judge.py --paper mechanistic-understanding \
        --submission ~/pb/submissions/mechanistic-understanding/submission --out ~/pb-grades/mechanistic-understanding

ANTHROPIC_API_KEY (or an `ant auth login` profile) pays for the judge. Scores from a Claude judge are not comparable
with PaperBench's published ones (an o3-mini judge); compare arms judged alike.
"""

import argparse
import asyncio
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from openai.types.chat import ChatCompletionMessage
from paperbench.judge.simple import ParsedJudgeResponseFloat, ParsedJudgeResponseInt, SimpleJudge
from paperbench.paper_registry import paper_registry
from paperbench.rubric.tasks import TaskNode
from preparedness_turn_completer.turn_completer import TurnCompleter
from pydantic import BaseModel, ConfigDict

PRICES = {"claude-opus-5-5": (4.00, 20.00, 0.20), "claude-sonnet-5-5": (2.00, 10.00, 0.20),
          "claude-haiku-5-5": (0.10, 0.50, 0.01)}           # $/MTok: input, output, cache read (writes: 1.25x input)
FALLBACK = ("claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5")
USAGE: dict = {}
WARM: dict = {}                                            # prefix -> set once its first call has cached it


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content or [] if isinstance(p, dict))


def _count(model: str, usage) -> None:
    u = USAGE.setdefault(model, {"calls": 0, "input": 0, "output": 0, "cache_read": 0, "cache_write": 0})
    u["calls"] += 1
    u["input"] += usage.input_tokens
    u["output"] += usage.output_tokens
    u["cache_read"] += usage.cache_read_input_tokens or 0
    u["cache_write"] += usage.cache_creation_input_tokens or 0


def cost() -> float:
    total = 0.0
    for model, u in USAGE.items():
        i, o, r = PRICES.get(model, (0, 0, 0))
        total += (u["input"] * i + u["output"] * o + u["cache_read"] * r + u["cache_write"] * i * 1.25) / 1e6
    return round(total, 4)


class ClaudeTurnCompleter(TurnCompleter):
    """A PaperBench TurnCompleter over the Messages API. The system prompt, the paper and the addendum open every
    judge call, so they are cached; a response_format makes the call a structured output (the verdict parse)."""

    def __init__(self, model: str, max_tokens: int, effort: str, n_ctx: int, response_format=None):
        self.model, self.max_tokens, self.effort = model, max_tokens, effort
        self.response_format = response_format
        self.encoding_name = "o200k_base"                  # SimpleJudge truncates with tiktoken; n_ctx leaves room
        self.n_ctx = n_ctx
        self._client = None

    class Config(TurnCompleter.Config):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        model: str
        max_tokens: int = 16000
        effort: str = "medium"
        n_ctx: int = 150_000
        response_format: type[BaseModel] | None = None

        def build(self) -> "ClaudeTurnCompleter":
            return ClaudeTurnCompleter(self.model, self.max_tokens, self.effort, self.n_ctx, self.response_format)

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(max_retries=8)
        return self._client

    def completion(self, conversation, **params):
        return asyncio.run(self.async_completion(conversation, **params))

    async def async_completion(self, conversation, **params):
        system = "\n\n".join(_text(m["content"]) for m in conversation if m["role"] in ("system", "developer"))
        messages = [{"role": m["role"], "content": [{"type": "text", "text": _text(m["content"])}]}
                    for m in conversation if m["role"] in ("user", "assistant")]
        warm = None
        if len(messages) > 2:                              # the paper and the addendum: the same for every criterion
            messages[1]["content"][0]["cache_control"] = {"type": "ephemeral"}
            key = hashlib.sha256(json.dumps([self.model, system, messages[:2]]).encode()).hexdigest()
            if key in WARM:
                await WARM[key].wait()                     # the first call on this prefix has written the cache
            else:
                warm = WARM[key] = asyncio.Event()
        try:
            return await self._complete(conversation, system, messages)
        finally:
            if warm is not None:
                warm.set()

    async def _complete(self, conversation, system: str, messages: list):
        kw = {"model": self.model, "max_tokens": self.max_tokens, "messages": messages}
        if system:
            kw["system"] = [{"type": "text", "text": system}]
        if self.response_format is not None:
            r = await self.client.messages.parse(**kw, output_format=self.response_format)
            text = r.parsed_output.model_dump_json() if r.parsed_output is not None else None
        else:
            fallback = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"} \
                if self.model in FALLBACK else {}                # a declined criterion is retried on another model
            r = await self.client.beta.messages.create(**kw, output_config={"effort": self.effort}, **fallback)
            text = "".join(b.text for b in r.content if b.type == "text") or None
        _count(self.model, r.usage)
        if r.stop_reason == "refusal":
            text = None                                    # SimpleJudge marks the criterion invalid, not passed
        return TurnCompleter.Completion(input_conversation=conversation,
                                        output_messages=[ChatCompletionMessage(role="assistant", content=text)])


async def judge(paper_id: str, submission: Path, out: Path, model: str, parse_model: str, effort: str,
                n_ctx: int, full: bool = False) -> dict:
    paper = paper_registry.get_paper(paper_id)
    tree = TaskNode.from_dict(json.loads(Path(paper.rubric).read_text()))
    if not full:
        tree = tree.code_only() or tree.set_task_category("Code Development").set_sub_tasks([])
    out.mkdir(parents=True, exist_ok=True)
    parse = dict(model=parse_model, max_tokens=4000, n_ctx=n_ctx)
    j = SimpleJudge(
        paper_path=paper.paper_pdf, rubric=tree, submission_dir=submission, paper_md=paper.paper_md,
        addendum=paper.addendum.read_text() if paper.addendum else None,
        judge_addendum=paper.judge_addendum.read_text() if paper.judge_addendum.exists() else None,
        completer_config=ClaudeTurnCompleter.Config(model=model, effort=effort, n_ctx=n_ctx),
        int_completer_config=ClaudeTurnCompleter.Config(**parse, response_format=ParsedJudgeResponseInt),
        float_completer_config=ClaudeTurnCompleter.Config(**parse, response_format=ParsedJudgeResponseFloat),
        log_path=out, code_only=not full)
    started = time.time()
    graded = await j.judge()
    leaves = graded.get_leaf_nodes()
    commit = subprocess.run(["git", "-C", str(submission), "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    return {"paper": paper_id, "submission": str(submission), "commit": commit, "variant": "full" if full else "code-dev",
            "score": graded.score, "criteria": len(leaves), "invalid": sum(not n.valid_score for n in leaves),
            "judge": {"model": model, "parse_model": parse_model, "effort": effort, "n_ctx": n_ctx},
            "usage": USAGE, "usd": cost(), "seconds": round(time.time() - started, 1),
            "graded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "graded_task_tree": graded.to_dict()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Judge a PaperBench submission with Claude.")
    ap.add_argument("--paper", required=True)
    ap.add_argument("--submission", required=True, help="the submission folder (<submissions>/<paper>/submission)")
    ap.add_argument("--out", required=True, help="a folder for grade.json and the judge's logs (keep it from agents)")
    ap.add_argument("--model", default="claude-opus-5-5", help="the judge (default: claude-opus-5-5)")
    ap.add_argument("--parse-model", default="", help="parses each verdict into a score (default: --model)")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--full", action="store_true", help="the whole rubric, for a folder where reproduce.sh has run")
    ap.add_argument("--context", type=int, default=150_000, help="tiktoken tokens the judge may fill per call")
    a = ap.parse_args(argv)
    submission, out = Path(a.submission).expanduser().resolve(), Path(a.out).expanduser().resolve()
    res = asyncio.run(judge(a.paper, submission, out, a.model, a.parse_model or a.model, a.effort, a.context, a.full))
    (out / "grade.json").write_text(json.dumps(res, indent=2) + "\n")
    print(f"{a.paper} @ {res['commit'][:12]}: {'Full' if a.full else 'Code-Dev'} score {res['score']:.3f} over {res['criteria']} criteria "
          f"({res['invalid']} invalid), judge ${res['usd']:.2f}, {res['seconds']:.0f}s -> {out / 'grade.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
