"""Text as LaTeX: `esc` escapes LaTeX's special characters and writes the Unicode a text font may lack as math."""

from __future__ import annotations

_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
            "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


#: Unicode the engine's text font may lack, as LaTeX
_UNICODE = {"∈": r"$\in$", "∉": r"$\notin$", "∀": r"$\forall$", "∃": r"$\exists$", "ℕ": r"$\mathbb{N}$",
            "ℤ": r"$\mathbb{Z}$", "ℝ": r"$\mathbb{R}$", "ℚ": r"$\mathbb{Q}$", "≤": r"$\le$", "≥": r"$\ge$",
            "≠": r"$\neq$", "≈": r"$\approx$", "→": r"$\to$", "⇒": r"$\Rightarrow$", "✓": r"\checkmark{}",
            "✗": r"$\times$", "−": "-", "·": r"$\cdot$", "×": r"$\times$", "±": r"$\pm$", "∞": r"$\infty$",
            "√": r"$\surd$", "²": r"$^2$", "³": r"$^3$", "…": r"\dots{}", "∑": r"$\sum$", "∫": r"$\int$",
            "α": r"$\alpha$", "β": r"$\beta$", "σ": r"$\sigma$", "μ": r"$\mu$", "λ": r"$\lambda$", "θ": r"$\theta$"}


def esc(s) -> str:
    return "".join(_SPECIAL.get(ch) or _UNICODE.get(ch, ch) for ch in str("" if s is None else s))
