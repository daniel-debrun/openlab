r"""The design system is a conformance surface, not a preference.

``openlab/app/ui/src/styles/tokens.css`` is the single source of truth for every colour,
space, size and z-index in the web UI. This module pins the rules a stylesheet cannot
enforce on itself:

* **hex literals live in exactly one file** — otherwise "change the palette" becomes a
  grep-and-pray, and the chart layer silently drifts from the CSS;
* **light and dark declare the same token names**, so a theme is never half-authored;
* **the three light bodies are identical**, because they are a deliberate duplication
  (OS preference / explicit choice / print): duplication a test owns is fine, duplication
  that drifts is a bug;
* **``--vd-*`` and ``--sys-*`` never share a value**, because "it is false" and "it
  crashed" are two different facts and they must not look alike;
* **contrast floors**, in *both* palettes; and

* **the verdict hues are luminance-matched.** That last one is the product thesis in
  executable form. openlab's claim is that a verdict is not an opinion: a disproof is as
  much of a result as a proof. If ``--vd-disproved`` drifts darker or louder than
  ``--vd-proved``, the interface starts teaching that a disproof is a failure, which
  inverts the whole pitch. "Check that the positive state does not visually outrank it"
  is a measurement, so it is measured here rather than left to whoever last opened the
  file.

Dependency-free on purpose: this is a stylesheet parser and some colour arithmetic, and
it must keep working in an environment that has no Node and no browser.
"""
import re
from pathlib import Path

_UI_SRC = Path(__file__).resolve().parents[1] / "openlab" / "app" / "ui" / "src"
_TOKENS = _UI_SRC / "styles" / "tokens.css"
_SRC = _TOKENS.read_text()

#: `#root` must not read as a colour, so the class requires hex digits only.
_HEX = re.compile(r"#[0-9a-fA-F]{3,8}\b")
_DECL = re.compile(r"(--[a-z0-9-]+)\s*:\s*([^;}]+)")
_LIGHT_BODY = re.compile(r"/\* @@LIGHT@@ \*/(.*?)/\* @@/LIGHT@@ \*/", re.DOTALL)

#: every state `<Verdict>` can draw, and the two that are not a verdict colour.
VERDICT_TOKENS = ("proved", "supported", "disproved", "unproved", "inconclusive", "value")
SYSTEM_TOKENS = ("ok", "danger", "warn", "info")


# --------------------------------------------------------------------------- #
# colour math (WCAG 2.1 relative luminance + contrast ratio)
# --------------------------------------------------------------------------- #
def _channel(c: int) -> float:
    x = c / 255.0
    return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4


def luminance(hex_colour: str) -> float:
    h = hex_colour.lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)


def _srgb_to_oklab(hex_colour: str):
    """sRGB hex -> OKLab (Ottosson 2020). Needed for the chroma floor: WCAG luminance
    cannot tell a saturated hue from a grey of the same lightness, and "reads as grey" is
    exactly the failure mode a categorical slot must not have."""
    h = hex_colour.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = lin(r), lin(g), lin(b)
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s_ = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s2 = l ** (1 / 3), m ** (1 / 3), s_ ** (1 / 3)
    return (0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s2,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s2,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s2)


def chroma(hex_colour: str) -> float:
    """OKLCH chroma — how much hue a colour actually carries."""
    _, a, b = _srgb_to_oklab(hex_colour)
    return (a * a + b * b) ** 0.5


def contrast(a: str, b: str) -> float:
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def _decls(text: str) -> dict:
    """Every ``--token: value`` in a chunk of CSS; later wins, as CSS does."""
    return {m.group(1): m.group(2).strip() for m in _DECL.finditer(text)}


def _block(selector: str) -> str:
    """The body of the first rule whose selector text contains ``selector``."""
    i = _SRC.index(selector)
    start = _SRC.index("{", i) + 1
    depth, j = 1, start
    while depth:
        if _SRC[j] == "{":
            depth += 1
        elif _SRC[j] == "}":
            depth -= 1
        j += 1
    return _SRC[start:j - 1]


def _light_bodies() -> list:
    return _LIGHT_BODY.findall(_SRC)


DARK = _decls(_block(':root[data-theme="dark"]'))
LIGHT = _decls(_light_bodies()[0])
PALETTES = {"dark": DARK, "light": LIGHT}


def _hexes_with_prefix(palette: dict, prefix: str) -> set:
    out = set()
    for name, value in palette.items():
        if name.startswith(prefix):
            out.update(m.group(0).lower() for m in _HEX.finditer(value))
    return out


# --------------------------------------------------------------------------- #
# 1. hex literals live in exactly one file
# --------------------------------------------------------------------------- #
def test_only_tokens_css_declares_hex_colours():
    """Every source file under src/, not just the stylesheets: a hex in a .tsx is the
    same second source of truth as a hex in a .css, and easier to miss."""
    offenders = {}
    for path in sorted(_UI_SRC.rglob("*")):
        if path == _TOKENS or not path.is_file():
            continue
        if path.suffix not in (".css", ".ts", ".tsx"):
            continue
        found = sorted({m.group(0) for m in _HEX.finditer(path.read_text())})
        if found:
            offenders[path.relative_to(_UI_SRC).as_posix()] = found
    assert not offenders, (
        "hex literals outside src/styles/tokens.css:\n"
        + "\n".join(f"  {k}: {v}" for k, v in offenders.items())
        + "\nUse a token. tokens.css is the only place a colour is defined."
    )


def test_tokens_css_is_the_palette_and_nothing_else():
    """Every hex in tokens.css belongs to a custom-property declaration."""
    stripped = re.sub(r"/\*.*?\*/", "", _SRC, flags=re.DOTALL)
    for line in stripped.splitlines():
        if _HEX.search(line):
            assert "--" in line and ":" in line, (
                f"tokens.css has a hex outside a token declaration: {line.strip()!r}"
            )


# --------------------------------------------------------------------------- #
# 2. a theme is never half-authored
# --------------------------------------------------------------------------- #
def test_light_and_dark_declare_the_same_tokens():
    only_dark = set(DARK) - set(LIGHT)
    only_light = set(LIGHT) - set(DARK)
    assert not only_dark and not only_light, (
        f"palette token sets disagree.\n  dark only: {sorted(only_dark)}\n"
        f"  light only: {sorted(only_light)}\n"
        "A token defined in one theme and not the other renders as an inherited value "
        "from the other theme, which is how a light mode ends up with a dark surface in "
        "one corner."
    )


def test_every_colour_token_resolves_to_a_hex_or_a_token():
    for theme, palette in PALETTES.items():
        for name, value in palette.items():
            if name.startswith(("--sp-", "--fs-", "--r-", "--z-", "--mo-", "--lh-",
                                "--fw-", "--font-", "--el-")):
                continue
            ok = _HEX.search(value) or "var(" in value or value in ("transparent", "light", "dark")
            assert ok, f"{theme}: {name} is not a colour or a reference: {value!r}"


# --------------------------------------------------------------------------- #
# 3. the deliberate duplication must not drift
# --------------------------------------------------------------------------- #
def test_the_three_light_bodies_are_identical():
    bodies = _light_bodies()
    assert len(bodies) == 3, (
        f"expected 3 light palette bodies (OS preference, explicit choice, print), "
        f"found {len(bodies)}"
    )
    normalised = [re.sub(r"\s+", " ", b).strip() for b in bodies]
    assert normalised[0] == normalised[1] == normalised[2], (
        "the light palette bodies have drifted apart. They are duplicated on purpose "
        "(3 selectors, one palette); keep them byte-identical."
    )


def test_explicit_theme_choice_beats_the_os_preference():
    """The classic bug: ``@media (prefers-color-scheme: light) { :root {...} }`` has the
    same specificity as ``:root``, so an explicit dark choice loses on a light-OS
    machine. The media query must be guarded, and the explicit light selector must come
    last."""
    assert ':root:not([data-theme="dark"])' in _SRC, (
        'the light media query is unguarded, so [data-theme="dark"] cannot win on a '
        'light-OS machine. Guard it with :root:not([data-theme="dark"]).'
    )
    assert _SRC.index(':root[data-theme="light"]') > _SRC.index("prefers-color-scheme"), (
        'the explicit :root[data-theme="light"] block must come AFTER the '
        "prefers-color-scheme block so the toggle wins in both directions"
    )


def test_print_is_always_light():
    """A notebook gets printed. A dark printout wastes toner, and it must override an
    explicit dark choice, not just the default."""
    i = _SRC.index("@media print")
    head = _SRC[i:_SRC.index("{", _SRC.index("{", i) + 1)]
    assert ":root[data-theme]" in head, (
        "the print block must also select :root[data-theme] so it beats an explicit dark "
        "choice (equal specificity, and it comes last)"
    )


# --------------------------------------------------------------------------- #
# 4. the three colour channels never mix
# --------------------------------------------------------------------------- #
def test_the_three_colour_families_are_pairwise_disjoint():
    """``--ch-*`` (identity), ``--vd-*`` (evidence) and ``--sys-*`` (the machine).

    Status and verdict colours are RESERVED. A chart series may never wear one: a reader
    who has learned what the violet means would otherwise be told that "series 3" means
    "this was disproved".
    """
    for theme, palette in PALETTES.items():
        fams = {p: _hexes_with_prefix(palette, p) for p in ("--ch-", "--vd-", "--sys-")}
        for a, b in (("--ch-", "--vd-"), ("--ch-", "--sys-"), ("--vd-", "--sys-")):
            shared = fams[a] & fams[b]
            assert not shared, (
                f"{theme}: {a}* and {b}* share {sorted(shared)}. Identity, evidence and "
                "machine state are three channels; a colour may only serve one."
            )


def test_the_categorical_palette_has_no_grey_slots():
    """OKLCH chroma floor ~0.10: below it a hue stops doing identity work, and a grey
    series is indistinguishable from a disabled or missing one."""
    for theme, palette in PALETTES.items():
        for i in range(1, 9):
            token = f"--ch-{i}"
            if token not in palette:
                continue
            assert chroma(palette[token]) >= 0.095, (
                f"{theme}: {token} = {palette[token]} has chroma "
                f"{chroma(palette[token]):.3f}, below the ~0.10 floor — it reads as grey"
            )


# --------------------------------------------------------------------------- #
# 5. every token the UI names actually exists, in both themes
# --------------------------------------------------------------------------- #
def test_every_verdict_and_system_token_exists_in_both_themes():
    missing = []
    for theme, palette in PALETTES.items():
        for state in VERDICT_TOKENS:
            if f"--vd-{state}" not in palette:
                missing.append(f"{theme}: --vd-{state}")
        for state in SYSTEM_TOKENS:
            if f"--sys-{state}" not in palette:
                missing.append(f"{theme}: --sys-{state}")
    assert not missing, (
        "tokens the interface draws with are not declared: " + ", ".join(missing)
    )


def test_the_verdict_states_are_distinct_colours():
    for theme, palette in PALETTES.items():
        values = {s: palette[f"--vd-{s}"] for s in VERDICT_TOKENS}
        assert len(set(values.values())) == len(values), (
            f"{theme}: two verdict states share a colour: {values}. Hue is one of the "
            "three channels a verdict is read on; collapsing two states onto one hue "
            "leaves only the glyph and the word."
        )


def test_unproved_is_encoded_as_absence_not_as_a_colour():
    """"not proved" is the absence of a verdict, so it is drawn with a texture as well as
    a chromaless hue. A hatch survives greyscale printing and every form of colour
    blindness; a further hue would not, and would invite reading it as a disproof."""
    assert "--vd-unproved-hatch" in _SRC, (
        "--vd-unproved-hatch is missing: an unproved statement must be visibly unfilled, "
        "not merely a further colour"
    )
    assert "repeating-linear-gradient" in _SRC


# --------------------------------------------------------------------------- #
# 6. contrast floors — in BOTH palettes
# --------------------------------------------------------------------------- #
def test_text_and_verdict_contrast_floors():
    floors = [("--text-1", 7.0), ("--text-2", 4.5), ("--text-3", 4.5)]     # text-3 labels are 11px: no large-text allowance
    floors += [(f"--vd-{s}", 4.5) for s in VERDICT_TOKENS]
    floors += [(f"--sys-{s}", 4.5) for s in SYSTEM_TOKENS]
    failures = []
    for theme, palette in PALETTES.items():
        bg = palette["--surface-1"]
        for token, floor in floors:
            got = contrast(palette[token], bg)
            if got < floor:
                failures.append(f"{theme}: {token} on --surface-1 is {got:.2f}, need {floor}")
    assert not failures, "\n".join(failures)


# --------------------------------------------------------------------------- #
# 7. THE THESIS
# --------------------------------------------------------------------------- #
def test_a_disproof_does_not_visually_outrank_a_proof():
    """The design thesis as an assertion.

    "A screenshot of a disproved statement must look as authoritative as a proved one."
    Equal authority starts with equal optical weight: if one verdict hue is brighter than
    the other it reads as the louder claim, no matter what the type does. So the two are
    held to the same relative luminance and the same contrast against the surface they
    sit on.
    """
    for theme, palette in PALETTES.items():
        lp = luminance(palette["--vd-proved"])
        ld = luminance(palette["--vd-disproved"])
        assert abs(lp - ld) <= 0.08, (
            f"{theme}: verdict hues are luminance-mismatched (proved {lp:.4f} vs "
            f"disproved {ld:.4f}, gap {abs(lp - ld):.4f} > 0.08).\n"
            "One verdict now reads louder than the other, which inverts the thesis: a "
            "disproof is a result, not a failure."
        )
        bg = palette["--surface-1"]
        cp = contrast(palette["--vd-proved"], bg)
        cd = contrast(palette["--vd-disproved"], bg)
        assert abs(cp - cd) <= 1.0, (
            f"{theme}: verdict contrast is lopsided (proved {cp:.2f} vs disproved "
            f"{cd:.2f}); neither verdict may be easier to read than the other"
        )


def test_proved_is_not_the_system_ok_green():
    """Green is spent on "a tool finished". If `proved` were also green, the user would
    learn one signal for two unrelated meanings."""
    for theme, palette in PALETTES.items():
        assert palette["--vd-proved"] != palette["--sys-ok"], (
            f"{theme}: --vd-proved must not be the same green as --sys-ok"
        )


def test_disproved_is_not_the_system_danger_red():
    """The single most important colour rule in the product."""
    for theme, palette in PALETTES.items():
        assert palette["--vd-disproved"] != palette["--sys-danger"], (
            f"{theme}: --vd-disproved must never be --sys-danger. A disproof is a result; "
            "rendering it in error-red teaches the user that a disproof is a crash."
        )
