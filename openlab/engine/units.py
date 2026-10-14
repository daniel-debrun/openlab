r"""Physical units: ``3\,\text{m/s}``, ``9.81 \mathrm{m}/\mathrm{s}^2``, ``5 \text{ km} + 300 \text{ m}``.

A unit is recognised only where a unit is written -- a ``\text{...}``/``\mathrm{...}`` group
directly after a number -- so ``\mathrm{m} = 3`` (a variable called m) is unaffected. The
preprocessor rewrites a recognised group to ``\openlabunit{m/s^2}``; the transformer turns that
into a ``sympy.physics.units`` expression, which never collides with namespace names.
"""

from __future__ import annotations

import re
from typing import Optional

import sympy
import sympy.physics.units as u
from sympy.physics.units import convert_to
from sympy.physics.units.quantities import Quantity


def _scaled(factor, q, name, abbrev):
    """A prefixed unit sympy does not ship (kW, MHz, ...) as its own Quantity."""
    new = Quantity(name, abbrev=abbrev)
    u.systems.SI.set_quantity_dimension(new, u.systems.SI.get_quantity_dimension(q))
    u.systems.SI.set_quantity_scale_factor(new, factor * q)
    return new


UNITS: dict = {
    # length
    "m": u.m, "km": u.km, "cm": u.cm, "mm": u.mm, "um": u.um, "µm": u.um, "nm": u.nm,
    "ft": u.foot, "inch": u.inch, "mi": u.mile, "mile": u.mile, "yd": u.yard,
    # time
    "s": u.s, "ms": u.ms, "us": u.us, "µs": u.us, "ns": u.ns, "min": u.minute,
    "h": u.hour, "hr": u.hour, "day": u.day, "yr": u.year, "year": u.year,
    # mass
    "kg": u.kg, "g": u.g, "mg": u.mg,
    "lb": u.pound if hasattr(u, "pound") else None,
    # force, energy, power, pressure
    "N": u.newton, "kN": _scaled(1000, u.newton, "kilonewton", "kN"),
    "J": u.joule, "kJ": _scaled(1000, u.joule, "kilojoule", "kJ"), "eV": u.electronvolt,
    "W": u.watt, "kW": _scaled(1000, u.watt, "kilowatt", "kW"),
    "MW": _scaled(10**6, u.watt, "megawatt", "MW"),
    "Pa": u.pascal, "kPa": _scaled(1000, u.pascal, "kilopascal", "kPa"),
    "MPa": _scaled(10**6, u.pascal, "megapascal", "MPa"), "bar": u.bar, "atm": u.atmosphere,
    "psi": u.psi,
    # electromagnetism
    "V": u.volt, "mV": _scaled(sympy.Rational(1, 1000), u.volt, "millivolt", "mV"),
    "kV": _scaled(1000, u.volt, "kilovolt", "kV"),
    "A": u.ampere, "mA": _scaled(sympy.Rational(1, 1000), u.ampere, "milliampere", "mA"),
    "C": u.coulomb, "F": u.farad, "H": u.henry, "T": u.tesla, "Wb": u.weber,
    "ohm": u.ohm, "Ohm": u.ohm, "Ω": u.ohm, r"\Omega": u.ohm,
    r"k\Omega": _scaled(1000, u.ohm, "kiloohm", "kOhm"),
    # frequency, amount, temperature, volume, angle
    "Hz": u.hertz, "kHz": _scaled(1000, u.hertz, "kilohertz", "kHz"),
    "MHz": _scaled(10**6, u.hertz, "megahertz", "MHz"), "GHz": _scaled(10**9, u.hertz, "gigahertz", "GHz"),
    "mol": u.mole, "K": u.kelvin, "L": u.liter, "l": u.liter, "mL": u.milliliter, "ml": u.milliliter,
    "rad": u.radian, "cd": u.candela,
}
_uH = _scaled(sympy.Rational(1, 10**6), u.henry, "microhenry", "uH")
_uF = _scaled(sympy.Rational(1, 10**6), u.farad, "microfarad", "uF")
UNITS.update({
    "mH": _scaled(sympy.Rational(1, 10**3), u.henry, "millihenry", "mH"), "uH": _uH, "µH": _uH,
    "nH": _scaled(sympy.Rational(1, 10**9), u.henry, "nanohenry", "nH"),
    "mF": _scaled(sympy.Rational(1, 10**3), u.farad, "millifarad", "mF"), "uF": _uF, "µF": _uF,
    "nF": _scaled(sympy.Rational(1, 10**9), u.farad, "nanofarad", "nF"),
    "pF": _scaled(sympy.Rational(1, 10**12), u.farad, "picofarad", "pF"),
    "kΩ": UNITS[r"k\Omega"], "MΩ": _scaled(10**6, u.ohm, "megaohm", "MOhm"),
    "°C": _scaled(1, u.kelvin, "degree_Celsius", "°C"),     # as a temperature difference
    "in": u.inch,                                             # after a number: 4\,\text{in}
    "Wh": _scaled(3600, u.joule, "watt_hour", "Wh"), "kWh": _scaled(3_600_000, u.joule, "kilowatt_hour", "kWh"),
    "MJ": _scaled(10**6, u.joule, "megajoule", "MJ"), "GJ": _scaled(10**9, u.joule, "gigajoule", "GJ"),
    "GPa": _scaled(10**9, u.pascal, "gigapascal", "GPa"), "GW": _scaled(10**9, u.watt, "gigawatt", "GW"),
    "mW": _scaled(sympy.Rational(1, 1000), u.watt, "milliwatt", "mW"),
    "°F": _scaled(sympy.Rational(5, 9), u.kelvin, "degree_Fahrenheit", "°F"),   # as a difference
    "kcal": _scaled(4184, u.joule, "kilocalorie", "kcal"), "cal": _scaled(sympy.Rational(4184, 1000), u.joule, "calorie", "cal"),
})
UNITS["Ah"] = _scaled(3600, u.coulomb, "ampere_hour", "Ah")
# every SI prefix on the common units: µC, nA, kWh, mAh, GHz, … (hand-listed ones win)
_PREFIXES = {"p": sympy.Rational(1, 10**12), "n": sympy.Rational(1, 10**9), "µ": sympy.Rational(1, 10**6),
             "u": sympy.Rational(1, 10**6), "m": sympy.Rational(1, 1000), "k": 1000, "M": 10**6, "G": 10**9}
for _base in ("C", "A", "V", "W", "J", "N", "Pa", "Hz", "F", "H", "Ω", "s", "g", "m", "L", "mol", "Wh", "Ah", "T", "eV"):
    for _p, _f in _PREFIXES.items():
        _key = _p + _base
        if _key not in UNITS and _base in UNITS and not (_p == "m" and _base == "in"):
            UNITS[_key] = _scaled(_f, UNITS[_base], f"{_p}{_base}", _key)
UNITS["degC"] = UNITS["°C"]
UNITS["degF"] = UNITS["°F"]
for _name, _scale, _dim in (("mph", u.mile / u.hour, u.meter / u.second), ("knot", 1852 * u.meter / u.hour, u.meter / u.second),
                            ("kn", 1852 * u.meter / u.hour, u.meter / u.second)):
    _q = Quantity(_name, abbrev=_name)
    u.systems.SI.set_quantity_dimension(_q, u.velocity)
    u.systems.SI.set_quantity_scale_factor(_q, _scale)
    UNITS[_name] = _q
_rpm = Quantity("rpm", abbrev="rpm")
u.systems.SI.set_quantity_dimension(_rpm, u.systems.SI.get_quantity_dimension(u.radian) / u.systems.SI.get_quantity_dimension(u.second))
u.systems.SI.set_quantity_scale_factor(_rpm, 2 * sympy.pi * u.radian / (60 * u.second))
UNITS["rpm"] = _rpm
UNITS = {k: v for k, v in UNITS.items() if v is not None}

_TOKEN = re.compile(r"k\\Omega|\\Omega|[A-Za-zµΩ°]+|\^\s*\{?\s*-?\d+\s*\}?|/|\*|\\cdot|\s+|\.")


def parse_unit(text: str) -> Optional[sympy.Expr]:
    """``"m/s^2"`` -> ``meter/second**2``; ``None`` if any token is not a known unit."""
    text = text.strip().replace("\\,", " ").replace("\\ ", " ").replace("{", "").replace("}", "")
    text = text.replace("(", " ").replace(")", " ")        # J/(mol K): everything after / is below
    if not text:
        return None
    pos, sign = 0, 1
    factors: list = []
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m:
            return None
        tok = m.group()
        pos = m.end()
        if tok.strip() in ("", "*", r"\cdot", "."):
            continue
        if tok == "/":
            sign = -1
            continue
        if tok.startswith("^"):
            if not factors:
                return None
            q, s, _ = factors[-1]
            factors[-1] = (q, s, int(re.sub(r"[^\d-]", "", tok)))
            continue
        q = UNITS.get(tok)
        if q is None:
            return None
        factors.append((q, sign, 1))
    if not factors:
        return None
    out = sympy.S.One
    for q, s, p in factors:
        out = out * q ** (s * p)
    return out


def has_units(expr) -> bool:
    return isinstance(expr, sympy.Basic) and bool(expr.atoms(Quantity))


def _dims(term):
    from sympy.physics.units.systems.si import SI, dimsys_SI
    return frozenset(dimsys_SI.get_dimensional_dependencies(SI.get_dimensional_expr(term)).items())


_TRANSCENDENTAL = (sympy.exp, sympy.log, sympy.sin, sympy.cos, sympy.tan, sympy.sinh, sympy.cosh,
                   sympy.tanh, sympy.asin, sympy.acos, sympy.atan)


def check_dims(expr):
    """``5 m + 2 s`` is an error, not a sum; so is ``\\ln(5 m)``."""
    from .errors import SemanticsError
    for pw in expr.atoms(sympy.Pow):
        if pw.exp.atoms(Quantity) and _dims(pw.exp) != _dims(sympy.S.One):
            raise SemanticsError(f"{sympy.sstr(pw)}: an exponent cannot have units",
                                 hint="an exponent must be a pure number; divide by a reference value first.")
    for f in expr.atoms(*_TRANSCENDENTAL):
        arg = f.args[0]
        if arg.atoms(Quantity) and _dims(arg) != _dims(sympy.S.One) and not (
                isinstance(f, (sympy.sin, sympy.cos, sympy.tan)) and arg.has(u.radian)):
            raise SemanticsError(f"{type(f).__name__} of {sympy.sstr(arg)}: the argument has units",
                                 hint="a logarithm, exponential or trig function needs a pure number; "
                                      "divide by a reference value first, e.g. \\ln(5\\,\\text{m} / 1\\,\\text{m}).")
    for add in expr.atoms(sympy.Add):
        if not add.atoms(Quantity):
            continue
        seen: dict = {}
        for term in add.args:
            seen.setdefault(_dims(term), term)
        if len(seen) > 1:
            a, b = list(seen.values())[:2]
            raise SemanticsError(f"cannot add {sympy.sstr(a)} and {sympy.sstr(b)}: the units measure different things",
                                 hint="check the formula's units, or convert one side.")


def convert(expr, text: str, notes: list, src: str = ""):
    """``5\\,\\text{km}\\text{ in }\\text{m}``: express ``expr`` in the unit ``text``."""
    from .errors import SemanticsError
    target = parse_unit(text)
    if target is None:
        raise SemanticsError(f"unknown unit `{text}`",
                             hint="known units include m, km, s, h, kg, N, J, W, Pa, V, A, Ω, Hz, K, °C, mol, L.")
    degc, kelvin = UNITS["°C"], u.kelvin
    temps = {degc: lambda c: c + sympy.Rational(27315, 100), UNITS["°F"]: lambda f: (f - 32) * sympy.Rational(5, 9)
             + sympy.Rational(27315, 100), kelvin: lambda k: k}
    back = {degc: lambda k: k - sympy.Rational(27315, 100), UNITS["°F"]: lambda k: (k - sympy.Rational(27315, 100))
            * sympy.Rational(9, 5) + 32, kelvin: lambda k: k}
    diff = len(re.findall(r"openlabunit\{deg[CF]\}", src)) >= 2 and "-" in src
    if target in temps and isinstance(expr, sympy.Expr) and diff:
        notes.append("a difference of temperatures converts by scale only")
    if target in temps and isinstance(expr, sympy.Expr) and not diff:   # temperatures convert as temperatures
        for src_u, to_k in temps.items():
            c_ = expr.coeff(src_u)
            if src_u != target and c_ != 0 and not c_.atoms(Quantity) and sympy.simplify(expr - c_ * src_u) == 0:
                notes.append("converted as a temperature (not a temperature difference)")
                return back[target](to_k(c_)) * target
    if expr == 0 and target == kelvin and not diff:   # 0 °C: the product with 0 lost its unit
        m = re.search(r"\\openlabunit\{deg([CF])\}", src)
        if m:
            expr = sympy.Mul(0, UNITS["°" + m.group(1)], evaluate=False)
            c0 = sympy.Rational(27315, 100) if m.group(1) == "C" else (-32 * sympy.Rational(5, 9) + sympy.Rational(27315, 100))
            notes.append(f"°{m.group(1)} → K as a temperature")
            return c0 * kelvin
    if not has_units(expr):
        raise SemanticsError(f"nothing to convert: {sympy.sstr(expr)} has no units openlab knows",
                             hint="write the value with a unit, e.g. 5\\,\\text{km} \\text{ in } \\text{m}.")
    out = convert_to(expr, target)
    if out.atoms(Quantity) - target.atoms(Quantity):
        raise SemanticsError(f"cannot express {sympy.sstr(expr)} in {text}: the units measure different things")
    notes.append(f"expressed in {text}")
    return out


def tidy(expr, notes: list):
    """Present a unit expression in the unit it was first written in, else in SI."""
    if not expr.atoms(Quantity):
        return expr
    trig = (sympy.sin, sympy.cos, sympy.tan, sympy.cot, sympy.sec, sympy.csc)
    if expr.has(u.radian):                   # cos(π rad) = cos π: a radian is a pure number
        expr = expr.replace(lambda f: isinstance(f, trig) and f.args[0].has(u.radian),
                            lambda f: f.func(f.args[0].subs(u.radian, 1)))
        if not expr.atoms(Quantity):
            return expr
    expr = _affine(expr, notes)
    check_dims(expr)
    try:
        written = _first_units(expr)
        first = written[0]
        conv = convert_to(expr, first)
        if len(conv.atoms(Quantity)) == 1:
            return conv
        si = convert_to(expr, [u.m, u.kg, u.s, u.A, u.K, u.mol, u.cd])
        for derived in (u.newton, u.joule, u.watt, u.pascal, u.volt, u.coulomb, u.ohm, u.hertz, u.second):
            d = convert_to(si, derived)
            if d.atoms(Quantity) == {derived} and d.as_powers_dict().get(derived, 0) > 0:
                return d
        conv = convert_to(expr, written)                # J/(mol K), km/h stay in the units written
        if conv.atoms(Quantity) <= set(written) and len(conv.atoms(Quantity)) <= len(si.atoms(Quantity)):
            return conv
        return si
    except Exception:   # conversion is presentation only
        return expr


def _affine(expr, notes):
    """A sum that mixes °C/°F with kelvin adds temperatures: each °C/°F term is taken as an
    absolute temperature (converted to K), since SymPy's °C is only a scale."""
    degc, degf = UNITS["°C"], UNITS["°F"]
    for add in list(expr.atoms(sympy.Add)):
        terms = add.args
        kinds = set()
        for t in terms:
            q = t.atoms(Quantity)
            if q == {degc} or q == {degf}:
                kinds.add("rel")
            elif q == {u.kelvin}:
                kinds.add("K")
            else:
                kinds.add("other")
        if kinds != {"rel", "K"}:
            continue
        new = []
        for t in terms:                 # a negative term is a subtracted temperature: 300 K − 25 °C
            for q, to_k in ((degc, lambda c: c + sympy.Rational(27315, 100)),
                            (degf, lambda f: (f - 32) * sympy.Rational(5, 9) + sympy.Rational(27315, 100))):
                c = t.coeff(q)
                if c != 0 and not c.atoms(Quantity):
                    sign = -1 if c.could_extract_minus_sign() else 1
                    new.append(sign * to_k(sign * c) * u.kelvin)
                    break
            else:
                new.append(t)
        notes.append("°C/°F with kelvin: the Celsius/Fahrenheit values are taken as temperatures (+273.15 K)")
        expr = expr.xreplace({add: sympy.Add(*new)})
    return expr


def _first_units(expr):
    """The units in the order they occur in the printed expression."""
    seen = []
    for node in sympy.preorder_traversal(expr):
        if isinstance(node, Quantity) and node not in seen:
            seen.append(node)
    return seen
