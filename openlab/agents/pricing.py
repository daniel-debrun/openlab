"""What a model's tokens cost, for the live estimate before the CLI reports its own bill.

The truth is the CLI's: Claude Code's stream-json `result` event carries `total_cost_usd` (and `modelUsage[m].costUSD`),
which replaces the estimate the moment it arrives; a Claude Code session's transcript has no bill, so its spend stays
an estimate and is shown with `~`. The estimate prices each usage block by the model that produced it:

    input * in + output * out + cache_read * read + cache_write_5m * 1.25 * in + cache_write_1h * 2 * in

($ per million tokens; cache reads are 0.1x input unless the model says otherwise). Prices as published for the
Claude API in 2026-10; an unknown model is priced as the most expensive of its family, so an estimate errs high,
never low. Codex models use their own table.
"""

from __future__ import annotations

#: (model id fragment, $/M input, $/M output, $/M cache read or None for 0.1x input); the first fragment the model
#: name contains wins, so the specific ones come first
PRICES = [
    ("fable-5-1", 10.0, 50.0, 0.25), ("mythos-5-1", 10.0, 50.0, 0.25), ("fable", 10.0, 50.0, None),
    ("mythos", 10.0, 50.0, None),
    ("opus-5-5", 4.0, 20.0, None), ("opus-5", 5.0, 25.0, None), ("opus-4-8", 5.0, 25.0, None),
    ("opus-4-7", 5.0, 25.0, None), ("opus-4-6", 5.0, 25.0, None), ("opus-4-5", 5.0, 25.0, None),
    ("opus", 15.0, 75.0, None),
    ("sonnet-5-5", 2.0, 10.0, None), ("sonnet-5", 2.0, 10.0, None), ("sonnet", 3.0, 15.0, None),
    ("haiku-5-5", 0.1, 0.5, None), ("haiku-5", 0.1, 0.5, None), ("haiku-4-5", 1.0, 5.0, None),
    ("haiku", 1.0, 5.0, None),
    ("gpt-5-mini", 0.25, 2.0, None), ("gpt-5", 1.25, 10.0, None), ("o3", 2.0, 8.0, None), ("o4-mini", 1.1, 4.4, None),
]
DEFAULT = ("sonnet", 3.0, 15.0, None)


def price(model: str) -> tuple:
    """(input, output, cache read) $ per million tokens for a model id."""
    m = (model or "").lower()
    _, pin, pout, read = next((p for p in PRICES if p[0] in m), DEFAULT)
    return pin, pout, (read if read is not None else 0.1 * pin)


def tokens_in(u: dict) -> int:
    return sum(int(u.get(k) or 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))


def cost(model: str, u: dict) -> float:
    """The $ of one usage block (Anthropic `usage` shape) on `model`."""
    if not u:
        return 0.0
    pin, pout, read = price(model)
    write = int(u.get("cache_creation_input_tokens") or 0)
    split = u.get("cache_creation") if isinstance(u.get("cache_creation"), dict) else {}
    w1h = int(split.get("ephemeral_1h_input_tokens") or 0)
    w5m = int(split.get("ephemeral_5m_input_tokens") or 0) if split else write
    if split and w1h + w5m < write:                                 # an unsplit remainder: the 5-minute price
        w5m = write - w1h
    return (int(u.get("input_tokens") or 0) * pin + int(u.get("output_tokens") or 0) * pout
            + int(u.get("cache_read_input_tokens") or 0) * read + w5m * 1.25 * pin + w1h * 2.0 * pin) / 1e6


def codex_cost(model: str, u: dict) -> float:
    """OpenAI usage: input_tokens includes cached_input_tokens (billed at 0.1x)."""
    pin, pout, read = price(model or "gpt-5")
    cached = int(u.get("cached_input_tokens") or 0)
    return ((int(u.get("input_tokens") or 0) - cached) * pin + cached * read
            + int(u.get("output_tokens") or 0) * pout) / 1e6
