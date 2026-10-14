"""Fixes and features requested by the Goldbach, twin-prime and Collatz solvers."""
import pytest
import sympy

from openlab.engine import OpenlabError, run_all


def last(src, ns=None, **o):
    return run_all(src, {} if ns is None else ns, **o)[-1]


COLLATZ_T = r"T(n) = \begin{cases} \frac{n}{2} & n \bmod 2 = 0 \\ \frac{3n+1}{2} & \text{otherwise} \end{cases}"


def _T():
    ns = {}
    run_all(COLLATZ_T, ns)
    return ns


# --- soundness -------------------------------------------------------------------------------------

def test_a_false_indicator_claim_is_disproved_not_proved():
    assert last(r"\forall n \ge 1: \operatorname{isprime}(2n+1) = 1 \text{ by induction}").value is sympy.false
    assert last(r"\forall n \ge 0: \operatorname{isprime}(2^{2^n}+1) = 1 \text{ by induction}").value is sympy.false


def test_a_lemma_is_not_instantiated_at_a_fraction():
    ns = {}
    run_all(r"\forall n \ge 1: \sum_{k=1}^{n} (2k-1) = n^2 \text{ by induction}", ns)
    assert last(r"\forall n \ge 1: \sum_{k=1}^{2n} (2k-1) = 4n^2 \text{ by algebra}", ns).value is sympy.true


def test_algebra_inequalities_respect_the_domain():
    assert last(r"\forall n \ge 1: 3n+1 < 4n+1 \text{ by algebra}").value is sympy.true
    assert last(r"\forall n \ge 1: n^2 \ge n \text{ by algebra}").value is sympy.true
    assert last(r"\forall n \in \mathbb{Z}: n^2 \ge n \text{ by algebra}").value is sympy.true
    assert last(r"\forall n \in \mathbb{Z}, n \ge 1: 3n+1 < 4n+1 \text{ by algebra}").value is sympy.true
    r = last(r"\forall n \in \mathbb{N}: 3n+1 < 4n+1 \text{ by algebra}")
    assert r.value is sympy.false and "n = 0" in r.notes[-1]
    assert last(r"\forall x \in \mathbb{R}: x^2 \ge x \text{ by algebra}").value is sympy.false


def test_the_induction_variable_is_an_integer_whatever_its_letter():
    assert last(r"\forall a \ge 0: 3^a \equiv 1 \pmod 2 \text{ by induction}").value is sympy.true


def test_powers_with_a_bounded_exponent():
    assert last(r"\forall n \ge 1: 2 \cdot 2^{n-1} \equiv 0 \pmod{2} \text{ by algebra}").value is sympy.true
    assert last(r"\forall n \ge 0: 2^{n} \equiv 0 \pmod{2} \text{ by algebra}").value is None


# --- evaluation --------------------------------------------------------------------------------------

def test_evaluation_bar_with_named_lower_limit():
    assert last(r"\left. n^2 \right|_{n=1}^{5}").value == 24


def test_sums_of_gcds_and_nested_sums():
    assert last(r"\sum_{k=1}^{10} \gcd(k, 10)").value == 27
    ns = {}
    run_all(r"r(n) = \sum_{k=2}^{n-2} \operatorname{isprime}(k)\operatorname{isprime}(n-k)", ns)
    assert last(r"\sum_{m=2}^{50} r(2m)", ns).value == 361
    assert last(r"\sum_{m=2}^{50} r(2m)^2", ns).value == 3363


def test_matrix_entries_and_factorint_of_an_expression():
    ns = {}
    run_all("f(n) = n^2", ns)
    assert list(last(r"\begin{bmatrix} f(1) & f(2) & f(3) \end{bmatrix}", ns).value) == [1, 4, 9]
    v = last(r"\operatorname{factorint}(2^{32}+1)").value
    assert [tuple(t) for t in v] == [(641, 1), (6700417, 1)]


def test_numeric_integral_without_closed_form_and_ln_products():
    r = last(r"\int_2^{998} \frac{1}{\ln t \ln(1000-t)} \, dt", numeric=True)
    assert abs(float(r.value) - 30.3717142036035) < 1e-9
    assert "log(t)*log(n - t)" in last(r"\ln t \ln(n-t)").interpretation


def test_isprime_renders_as_an_operator_name():
    assert r"\operatorname{isprime}" in last(r"\operatorname{isprime}(k)").latex_out


def test_sums_over_primes_divisors_and_sets():
    assert last(r"\sum_{p \in \operatorname{primes}(30)} p").value == 129
    assert last(r"\sum_{\substack{p \le 30 \\ p \text{ prime}}} p").value == 129
    assert last(r"\prod_{p \mid 60,\, p > 2,\, p \text{ prime}} \frac{p-1}{p-2}").value == sympy.Rational(8, 3)
    assert last(r"\sum_{d \mid 60} d").value == 168
    assert last(r"7 \in \mathbb{P}").value is sympy.true
    assert last(r"9 \notin \mathbb{P}").value is sympy.true
    with pytest.raises(OpenlabError):
        run_all(r"\sum_{p \le 100} \ln p", {})              # no lower bound, no `prime`: refused


def test_ranges_set_builder_and_exhaustion():
    assert last(r"7 \in \{2, \ldots, 10\}").value is sympy.true
    ns = {}
    run_all(r"r(n) = \sum_{k=2}^{n-2} \operatorname{isprime}(k)\operatorname{isprime}(n-k)", ns)
    assert last(r"\{n \in \mathbb{Z} : 4 \le n \le 40, r(n) = 0\}", ns).value == sympy.FiniteSet(11, 17, 23, 27, 29, 35, 37)
    assert last(r"\min_{m \in \{2, \ldots, 50\}} r(2m)", ns).value == 1
    assert last(r"\forall m \in \{2, 3, \ldots, 200\}: r(2m) \ge 1 \text{ by exhaustion}", ns).value is sympy.true
    r = last(r"\forall n \in \{0, \ldots, 40\}: \operatorname{isprime}(n^2 + n + 41) = 1 \text{ by exhaustion}")
    assert r.value is sympy.false and "n = 40" in r.notes[-1]
    with pytest.raises(OpenlabError):                          # an infinite set is never proved by cases
        run_all(r"\forall n \ge 1: n^2 \ge n \text{ by exhaustion}", {})


# --- Collatz ------------------------------------------------------------------------------------------

def test_exhaustive_residue_conditions_compose():
    ns = {}
    run_all(r"T(n) = \begin{cases} \frac{n}{2} & n \bmod 2 = 0 \\ \frac{3n+1}{2} & n \bmod 2 = 1 \end{cases}", ns)
    assert last(r"\forall n \in \mathbb{Z}: T(T(4n+1)) = 3n+1 \text{ by algebra}", ns).value is sympy.true
    run_all(r"U(n) = \begin{cases} 1 & n \bmod 3 = 0 \\ 2 & n \bmod 3 = 1 \end{cases}", ns)
    with pytest.raises(OpenlabError):
        run_all("U(5)", ns)                                  # no row applies: an error, never nan


def test_discrete_max_argmax_over_sequences_and_piecewise():
    ns = _T()
    assert last(r"\max_{1 \le n \le 10} T(n)", ns).value == 14
    assert last(r"\arg\max_{1 \le n \le 10} T(n)", ns).value == 9
    run_all("a_0 = 27\na_{k+1} = T(a_k)", ns)
    assert last(r"\max_{0 \le k \le 70} a_k", ns).value == 4616
    assert last(r"\sum_{k=0}^{69} (a_k \bmod 2)", ns).value == 41
    assert last(r"\max_{0 \le x \le 3} x(3-x)").value == sympy.Rational(9, 4)      # real variable: continuous


def test_iterates_and_stopping_times():
    ns = _T()
    assert last(r"T^{\circ 3}(7)", ns).value == 26
    assert last(r"T^{3}(7)", ns).value == 1331                  # a power, as before
    assert last(r"\forall m \in \mathbb{Z}: T^{\circ 4}(16m+3) = 9m+2 \text{ by algebra}", ns).value is sympy.true
    run_all(r"m \in \mathbb{Z}", ns)
    assert str(last(r"T^{\circ 12}(4096m+27)", ns).value) == "19683*m + 137"
    run_all(r"\sigma(n) = \min\{k \ge 1 : T^{\circ k}(n) < n\}" "\n" r"\tau(n) = \min\{k \ge 0 : T^{\circ k}(n) = 1\}", ns)
    assert last(r"\sigma(27)", ns).value == 59 and last(r"\tau(27)", ns).value == 70
    assert last(r"\max_{2 \le n \le 1000} \sigma(n)", ns).value == 81
    assert last(r"\sum_{n=2}^{64} [\sigma(n) > 3]", ns).value == 16
    assert last(r"\#\{n : 2 \le n \le 64, \sigma(n) > 3\}", ns).value == 16


def test_symbolic_power_of_two_parity():
    ns = _T()
    assert last(r"\forall k \ge 1, m \in \mathbb{Z}: T(2^k m - 1) = 3 \cdot 2^{k-1} m - 1 \text{ by algebra}",
                ns).value is sympy.true


def test_binomial_theorem_by_algebra():
    assert last(r"\forall k \ge 0: \sum_{j=0}^{k} \binom{k}{j} 2^j = 3^k \text{ by algebra}").value is sympy.true
    assert last(r"\forall k \ge 0: \sum_{j=0}^{k} \binom{k}{j} x^j y^{k-j} = (x+y)^k \text{ by algebra}").value is sympy.true
    assert last(r"\forall k \ge 0: \sum_{j=0}^{k} \binom{k}{j} 2^j = 4^k \text{ by algebra}").value is None


def test_differential_numerator_in_an_integral():
    assert last(r"\int_0^1 \frac{dx}{1+x^2}").value == sympy.pi / 4


# --- round 2: twin-prime R5–R13, Goldbach R9–R16 ------------------------------------------------------

R_DEF = r"r(n) = \sum_{k=2}^{n-2} \operatorname{isprime}(k)\operatorname{isprime}(n-k)"


def test_a_claims_bound_variable_shadows_a_notebook_function():
    ns = {}
    run_all("M(p) = 0", ns)
    assert last(r"\forall j \in \mathbb{Z}, M \in \mathbb{Z}: 2 - j(2j+3) M = 2 \text{ by algebra}", ns).value is None


def test_induction_with_an_implicit_product():
    assert last(r"\forall n \ge 0: n(n+2)(n+4) \equiv 0 \pmod 3 \text{ by induction}").value is sympy.true
    assert last(r"\forall n \ge 0: n(n+1) \equiv 0 \pmod 3 \text{ by induction}").value is sympy.false


def test_subscripted_greek_function_and_differential_numerator():
    ns = {}
    run_all(r"\pi_2(x) = \sum_{k=2}^{x-2} \operatorname{isprime}(k)\operatorname{isprime}(k+2)", ns)
    assert last(r"\pi_2(100)", ns).value == 8
    assert last(r"\pi r^2").value == sympy.pi * sympy.Symbol("r") ** 2
    assert "primepi" in last(r"\pi(100)").interpretation


def test_iverson_divisibility_and_image_sets():
    assert last(r"[3 \mid 12]").value == 1 and last(r"[5 \nmid 12]").value == 1
    ns = {}
    run_all(r"N(p) = 4((p-1)!+1)+p", ns)
    assert last(r"\sum_{p=3}^{200} [p(p+2) \mid N(p)]", ns).value == 15          # Clement: 15 twin pairs
    v = last(r"\{(p, p+2) : p \in \operatorname{primes}(30), \operatorname{isprime}(p+2) = 1\}").value
    assert v == sympy.FiniteSet(*[sympy.Tuple(a, a + 2) for a in (3, 5, 11, 17, 29)])
    ns = {}
    run_all(R_DEF, ns)
    assert last(r"\#\{m \in \{2, \ldots, 50\} : r(2m) \le 6\}", ns).value == 23


def test_integrals_and_series_that_needed_care():
    r = last(r"\int_{e}^{\infty} \frac{(\ln \ln t)^2}{t \ln^2 t} \, dt")
    assert abs(float(r.value) - 2) < 1e-9                                          # never the 2-digit 1.8
    assert abs(float(last(r"\sum_{n=2}^{\infty} \frac{1}{n \ln^2 n}").numeric) - 2.10974280123689) < 1e-10
    assert abs(float(last(r"\sum_{n=3}^{\infty} \frac{1}{n (\ln n)^3}").numeric) - 0.564496185305683) < 1e-10
    assert abs(float(last(r"4 e^{-2\gamma_E}").numeric) - 1.26094700674877) < 1e-10


def test_prime_sums_in_integrals_and_symbolic_sets():
    assert last(r"\int_0^1 \left(\sum_{2 \le p \le 20,\, p \text{ prime}} e^{2\pi i p x}\right)^2 e^{-40 \pi i x} \, dx",
                complex_i=True).value == 4
    ns = {}
    run_all(r"\mathrm{Ch}(n) = \sum_{p \in \operatorname{primes}(n - 2)} [\operatorname{Omega}(n - p) \le 2]", ns)
    assert last(r"\mathrm{Ch}(100)", ns).value == 21
    assert last(r"\operatorname{isprime}(-7)").value == 0
    assert "not only primes" not in last(r"\sum_{p \in \operatorname{primes}(30)} p").interpretation


def test_isprime_facts_reflection_sos_and_case_split():
    assert last(r"\forall m \ge 2: \operatorname{isprime}(2m) = 0 \text{ by algebra}").value is sympy.true
    assert last(r"\forall m \ge 1: \operatorname{isprime}(2m) = 0 \text{ by algebra}").value is None
    ns = {}
    run_all(R_DEF, ns)
    assert last(r"\forall n \ge 4: r(n) \le n - 3 \text{ by algebra}", ns).value is sympy.true
    assert last(r"\forall n \ge 4: r(n) \le n - 4 \text{ by algebra}", ns).value is None     # a bound never disproves
    assert last(r"\forall m \ge 3: r(2m) = 2\sum_{k=2}^{m-1} \operatorname{isprime}(k)\operatorname{isprime}(2m-k)"
                r" + \operatorname{isprime}(m) \text{ by algebra}", ns).value is sympy.true
    assert last(r"\forall a, b, x, y \in \mathbb{R}: (ax + by)^2 \le (a^2+b^2)(x^2+y^2) \text{ by algebra}").value is sympy.true
    assert last(r"\forall a, b \in \mathbb{R}: 3ab \le a^2 + b^2 \text{ by algebra}").value is None
    assert last(r"\forall k \in \mathbb{Z}, k \ne 0: \int_0^1 e^{2\pi i k x} \, dx = 0 \text{ by algebra}",
                complex_i=True).value is sympy.true


def test_numeric_sums_at_decimal_points_are_fast_and_right():
    ns = {}
    run_all(r"\Phi(x) = \sum_{2 \le p \le 2000,\, p \text{ prime}} e^{2\pi i p x}", ns, complex_i=True)
    assert abs(float(last(r"|\Phi(0.6180339887)|", ns, complex_i=True).value) - 7.216600941917446) < 1e-8


def test_by_cases_on_residues_and_positive_coefficients():
    assert last(r"\forall n \ge 4: \operatorname{isprime}(n)\operatorname{isprime}(n+2)\,[n \bmod 6 \ne 5] = 0 "
                r"\text{ by cases on } n \bmod 6").value is sympy.true
    r = last(r"\forall n \ge 4: \operatorname{isprime}(n)\operatorname{isprime}(n+2) = 0 \text{ by cases on } n \bmod 6")
    assert r.value is sympy.false and "n = 5" in r.notes[-1]
    assert last(r"\forall n \ge 0: n^2 \bmod 4 \le 1 \text{ by cases on } n \bmod 4").value is sympy.true
    assert last(r"\forall k \ge 1: \operatorname{isprime}(6k) = 0 \text{ by algebra}").value is sympy.true
    assert last(r"\forall k \ge 1: \operatorname{isprime}(3k) = 0 \text{ by algebra}").value is None
    assert last(r"\forall k \ge 0, l \ge 0: k(2l+1) < 2(l+1)(k+2l+1) \text{ by algebra}").value is sympy.true
    assert last(r"\forall k \ge 0, l \ge 0: k(2l+3) < 2(l+1)(k+2l+1) \text{ by algebra}").value is None


# --- round 3: Goldbach R17–R19, Collatz R10–R15 -------------------------------------------------------

def test_vanishing_sums_parity_split_and_bare_indicator():
    ns = {}
    run_all(R_DEF, ns)
    assert last(r"\forall m \ge 2: r(2m+1) = 2\operatorname{isprime}(2m-1) \text{ by algebra}", ns).value is sympy.true
    assert last(r"\forall m \ge 1: \sum_{j=1}^{m} \operatorname{isprime}(2j) = 1 \text{ by algebra}").value is sympy.true
    assert last(r"\forall m \ge 1: \sum_{j=1}^{m} \operatorname{isprime}(2j) = 0 \text{ by algebra}").value is None
    assert last(r"\forall m \ge 2: \operatorname{isprime}(m) \le 1 \text{ by algebra}").value is sympy.true


def test_a_failed_definition_is_not_replaced_by_a_builtin():
    ns = {}
    with pytest.raises(OpenlabError):
        run_all(r"\Phi(x) = \undefinedcmd", ns)
    with pytest.raises(OpenlabError):
        run_all(r"\Phi(0.5)", ns)
    assert abs(float(last(r"\Phi(0.5)").value) - 0.691462461274013) < 1e-12


def test_nested_and_deferred_aggregates():
    ns = {}
    run_all(r"B(k, a) = \frac{2^{k-a}(3^a - 2^a)}{2^k - 3^a}", ns)
    top = sympy.Rational(27762853642600448, 252223018333)
    assert last(r"\max_{1 \le k \le 40} \max_{1 \le a \le 40} B(k, a)", ns).value == top
    run_all(r"M(k) = \max_{a \in \{1, \ldots, k\}} B(k, a)", ns)
    assert last("M(40)", ns).value == top
    run_all(COLLATZ_T + "\n" + r"\sigma(n) = \min\{k \ge 1 : T^{\circ k}(n) < n\}", ns)
    assert last(r"\{n \in \mathbb{Z} : 2 \le n \le 2000, \sigma(n) > \max_{2 \le j \le n-1} \sigma(j)\}", ns).value \
        == sympy.FiniteSet(2, 3, 7, 27, 703)
    run_all(r"S(k) = \#\{r : 0 \le r \le 2^k - 1, \sigma(r + 2^k) > k\}", ns)       # k is not captured by σ's k
    assert last("S(10)", ns).value == 64


def test_pairs_logs_continued_fractions_and_exact_binding():
    assert last(r"\#\{(k, a) : 1 \le k \le 40, 1 \le a \le 40, 3^a < 2^k\}").value == 497
    assert abs(float(last(r"\log_2 \log_3 2").numeric) + 0.664448707453889) < 1e-12
    assert list(last(r"\operatorname{cf}(\log_2 3, 15)").value) == [1, 1, 1, 2, 2, 3, 1, 5, 2, 23, 2, 2, 1, 1, 55]
    assert last(r"\operatorname{convergents}(\log_2 3, 10)").value[-1] == sympy.Rational(24727, 15601)
    ns = {}
    run_all(r"x_0 = \log_2 3", ns, numeric=True)
    assert not ns["x_0"].has(sympy.Float)                                         # -n shows, never rounds
    v = last(r"\{a \in \mathbb{Z} : 1 \le a \le 2000, \lceil a \log_2 3 \rceil - a\log_2 3 < \frac{a}{3\cdot 10^5 \ln 2}\}").value
    assert v == sympy.FiniteSet(971, 1277, 1583, 1636, 1889, 1942)


def test_claim_iterates_use_the_binder_and_undecided_iterates_wait():
    ns = {}
    run_all(COLLATZ_T, ns)
    assert last(r"\forall m \in \mathbb{Z}: T^{\circ 10}(1024m - 1) = 59049m - 1 \text{ by algebra}", ns).value is sympy.true
    run_all(r"P(r) = \sum_{i=0}^{5} 2^i (T^{\circ i}(r) \bmod 2)", ns)
    assert last(r"\sum_{r=0}^{63}\sum_{s=0}^{63}[P(r) = P(s)]", ns).value == 64


def test_aggregates_over_tuple_entries_and_exact_tuples_under_n():
    ns = {}
    run_all(r"c = \operatorname{convergents}(\log_2 3, 30)", ns)
    assert last(r"\sum_{n=1}^{3} \operatorname{denom}(c_n)", ns).value == 4       # never denom of the symbol
    assert last(r"\min_{n \in \{2, 4\}} c_n", ns).value == sympy.Rational(8, 5)
    assert last(r"\sum_{n=1}^{30} [\operatorname{denom}(c_n) \le 322] [c_n > \log_2 3]", ns).value == 4
    with pytest.raises(OpenlabError):
        run_all(r"\sum_{n=1}^{40} c_n", ns)                                         # past the last entry: refused
    assert r"c = \left(" in last(r"c = \operatorname{convergents}(\log_2 3, 5)", {}).latex_out
    ns = {}
    run_all(r"c = \operatorname{convergents}(\log_2 3, 30)", ns, numeric=True)
    assert ns["c"][17] == sympy.Rational(272500658, 171928773)
    assert float(last(r"(c_{18} - \log_2 3) \cdot 3 \cdot 2^{68} \ln 2", ns, numeric=True).value) > 0


def test_vanishing_terms_respect_the_boundary_and_lemmas_do_not_block():
    ns = {}
    run_all(R_DEF, ns)
    # false at m = 2 (r(4) = 1 from 2 + 2): never proved
    assert last(r"\forall m \ge 2: r(2m) = \sum_{j=1}^{m-2} \operatorname{isprime}(2j+1)\operatorname{isprime}(2m-2j-1)"
                r" \text{ by algebra}", ns).value is not sympy.true
    for first, second in ((r"\sum_{j=1}^{m-2} \operatorname{isprime}(2j+1)\operatorname{isprime}(2m-2j-1)",
                           r"2\sum_{k=2}^{m-1} \operatorname{isprime}(k)\operatorname{isprime}(2m-k) + \operatorname{isprime}(m)"),
                          (r"2\sum_{k=2}^{m-1} \operatorname{isprime}(k)\operatorname{isprime}(2m-k) + \operatorname{isprime}(m)",
                           r"\sum_{j=1}^{m-2} \operatorname{isprime}(2j+1)\operatorname{isprime}(2m-2j-1)")):
        ns = {}
        run_all(R_DEF, ns)
        assert last(rf"\forall m \ge 3: r(2m) = {first} \text{{ by algebra}}", ns).value is sympy.true
        assert last(rf"\forall m \ge 3: r(2m) = {second} \text{{ by algebra}}", ns).value is sympy.true   # either order


def test_a_huge_exact_value_is_bound_as_a_decimal_under_n():
    ns = {}
    run_all(r"C_2 = \prod_{3 \le p \le 20000,\, p \text{ prime}} (1 - \frac{1}{(p-1)^2})", ns, numeric=True)
    assert isinstance(ns["C_2"], sympy.Float)


# --- round 4 ---------------------------------------------------------------------------------------------

def test_image_sets_wait_for_outer_variables():
    assert last(r"\forall p \in \operatorname{primes}(3): |\{h \bmod p : h \in \{0, 2\}\}| = 2 \text{ by exhaustion}").value \
        is sympy.false                                                    # {0} at p = 2
    assert last(r"\sum_{p \in \{2,3,5\}} |\{h \bmod p : h \in \{0, 2, 6\}\}|").value == 6
    ns = {}
    run_all(r"\nu(p) = |\{h \bmod p : h \in \{0,2,6\}\}|", ns)
    assert last(r"\nu(2)", ns).value == 1


def test_functions_of_list_entries_and_certified_comparisons():
    ns = {}
    run_all(r"c = \operatorname{convergents}(\log_2 3, 200)", ns)
    run_all(r"G(n) = \log_2(c_n - \log_2 3) + \operatorname{denom}(c_n)", ns)
    assert abs(float(last("G(4)", ns).numeric) + 1.05529152156097) < 1e-10
    assert last(r"\sum_{j=1}^{100} [c_{2j} > \log_2 3]", ns).value == 100      # differences ~10⁻¹⁹⁰
    assert last(r"\sum_{n=3}^{60} [c_n > \log_2 3] \left[\log_2(c_n - \log_2 3) < 0\right]", ns).value == 29
    assert last(r"\arg\min_{j \in \{60, \ldots, 100\}} (\log_2(c_{2j} - \log_2 3) + 2\log_2(\operatorname{denom}(c_{2j})))",
                ns).value == 93


def test_aggregate_index_outside_its_body_is_refused():
    ns = {}
    run_all("A(m) = m^2\nB(m) = 3m", ns)
    with pytest.raises(OpenlabError):
        run_all(r"\min_{m \in \{1, \ldots, 5\}} A(m) - B(m)", ns)
    assert last(r"\min_{m \in \{1, \ldots, 5\}} (A(m) - B(m))", ns).value == -2


def test_min_over_a_deferred_set_and_max_over_a_bound_set():
    ns = {}
    run_all(r"p_0(n) = \min \{p \in \operatorname{primes}(3, n) : \operatorname{isprime}(n - p) = 1\}", ns)
    assert last("p_0(7426)", ns).value == 173
    assert last(r"\max_{m \in \{3, \ldots, 500\}} p_0(2m)", ns).value == 73
    ns = {}
    run_all(COLLATZ_T + "\n" + r"\sigma(n) = \min\{k \ge 1 : T^{\circ k}(n) < n\}", ns)
    run_all(r"Q = \{r \in \mathbb{Z} : 0 \le r \le 255, \sigma(r + 256) > 5\}", ns)
    assert last(r"\max_{r \in Q} \sigma(r + 256)", ns).value == 40


def test_exhaustion_on_one_variable_and_nested_forall():
    ns = {}
    run_all(COLLATZ_T + "\n" + r"A(n, j) = \sum_{i=0}^{j-1} (T^{\circ i}(n) \bmod 2)", ns)
    assert last(r"\forall r \in \{0, \ldots, 15\}: \forall m \in \mathbb{Z}: T^{\circ 4}(16m + r) = 3^{A(r,4)} m + T^{\circ 4}(r)"
                r" \text{ by exhaustion on } r", ns).value is sympy.true
    assert last(r"\forall n \ge 1: n + 1 = n + 2 \text{ by algebra}").value is sympy.false


def test_fermat_little_theorem_and_prime_conditions():
    assert last(r"\forall p \text{ prime}, k \ge 0: k^p \equiv k \pmod p \text{ by induction on } k").value is sympy.true
    assert last(r"\forall p \ge 2, k \ge 0: k^p \equiv k \pmod p \text{ by induction on } k").value is not sympy.true
    assert last(r"\forall p \ge 2, j \ge 1, \gcd(j, p) = 1: \binom{p}{j} \equiv 0 \pmod p \text{ by algebra}").value is sympy.true
    assert last(r"\forall n \ge 0: (x+1)^n = \sum_{j=0}^{n} \binom{n}{j} x^j \text{ by induction}").value is sympy.true


def test_gcd_harmonic_periodic_and_sieve():
    assert last(r"\forall k \ge 1: \gcd(6k + 1, 6) = 1 \text{ by algebra}").value is sympy.true
    assert last(r"\forall n \ge 1: \sum_{k=1}^{n} \frac{1}{k} \ge \ln(n+1) \text{ by induction}").value is sympy.true
    assert last(r"\forall q \ge 0: \sum_{n=1}^{30q} [\gcd(n(n+2), 30) = 1] = 3q \text{ by induction on } q").value is sympy.true
    ns = {}
    run_all(r"\pi_2(x) = \sum_{k=2}^{x-2} \operatorname{isprime}(k)\operatorname{isprime}(k+2)", ns)
    assert last(r"\pi_2(1000000)", ns).value == 8169


def test_iverson_split_range_decisions_and_term_lower_bounds():
    assert last(r"\forall x \ge 0, J > 0: [x = 0] \le \frac{(x - J)^2}{J^2} \text{ by algebra}").value is sympy.true
    assert last(r"\forall q \ge 3: \sum_{a=1}^{q-1} [(q - a) \bmod q \ne 0] = q - 1 \text{ by algebra}").value is sympy.true
    assert last(r"\forall q \ge 3: \sum_{a=1}^{q-1} [(q - a) \bmod q \ne 0] = q - 2 \text{ by algebra}").value is not sympy.true
    ns = {}
    run_all(R_DEF + "\n" + r"r_3(n) = \sum_{k=2}^{n-4} \operatorname{isprime}(k)\, r(n-k)", ns)
    assert last(r"\forall n \ge 7: r_3(n) \ge r(n-3) \text{ by algebra}", ns).value is sympy.true


def test_short_circuit_two_sided_binders_and_interior_splits():
    ns = {}
    run_all(r"h(n) = \sum_{p \in \operatorname{primes}(3, 31)} \operatorname{isprime}(n - p)", ns)
    assert last(r"\forall m \in \{20, \ldots, 20000\}: h(2m) \ge 1 \text{ by exhaustion}", ns).value is not None
    assert last(r"\forall q \ge 3, 1 \le n \le q - 1: \sum_{a=1}^{q-1} [a \ne n] = q - 2 \text{ by algebra}").value is sympy.true
    assert last(r"\forall q \ge 3, 1 \le n \le q - 1: \sum_{a=1}^{q-1} [a \ne n] = q - 1 \text{ by algebra}").value is not sympy.true
    assert last(r"\forall j \ge 1, c \ge 1: \sum_{a=1}^{j+c} [(j - a) \bmod (j + c + 1) \ne 0] = j + c - 1 \text{ by algebra}"
                ).value is sympy.true
    assert last(r"\forall j \ge 1, c \ge 1: \sum_{a=1}^{j+c} [(j - a) \bmod (j + c + 1) \ne 0] = j + c \text{ by algebra}"
                ).value is not sympy.true


def test_iterate_count_from_a_function_value():
    ns = {}
    run_all(COLLATZ_T + "\n" + r"A(n, j) = \sum_{i=0}^{j-1} (T^{\circ i}(n) \bmod 2)" + "\n"
            + r"\kappa(n) = \min\{j \ge 1 : 3^{A(n,j)} < 2^j\}", ns)
    assert last(r"\forall m \ge 0: T^{\circ \kappa(2)}(256m+2) < 256m+2 \text{ by algebra}", ns).value is sympy.true


def test_units_from_ranges_fermat_corollary_and_prime_conditions_in_lemmas():
    ns = {}
    assert last(r"\forall p \text{ prime}, k \in \{1, \ldots, p-1\}: \binom{p}{k} \equiv 0 \pmod p \text{ by algebra}", ns).value \
        is sympy.true
    run_all(r"\forall p \text{ prime}, k \ge 0: k^p \equiv k \pmod p \text{ by induction on } k", ns)
    assert last(r"\forall p \text{ prime}, k \in \{1, \ldots, p-1\}: k^{p-1} \equiv 1 \pmod p \text{ by algebra}", ns).value \
        is sympy.true
    # a lemma about primes must not be used for every p ≥ 2 (false at p = 4, k = 2)
    assert last(r"\forall p \ge 2, k \in \{1, \ldots, p-1\}: \binom{p}{k} \equiv 0 \pmod p \text{ by algebra}", ns).value \
        is not sympy.true
    assert last(r"\forall p \ge 2, k \in \{1, \ldots, p-1\}: k^{p-1} \equiv 1 \pmod p \text{ by algebra}", ns).value \
        is not sympy.true


def test_summing_a_proved_pointwise_inequality():
    ns = {}
    run_all(r"\forall n \ge 6: \operatorname{isprime}(n)\operatorname{isprime}(n+2) \le [n \bmod 30 = 11] + [n \bmod 30 = 17]"
            r" + [n \bmod 30 = 29] \text{ by cases on } n \bmod 30", ns, timeout=300)
    base = r"\sum_{n=%d}^{30q} \operatorname{isprime}(n)\operatorname{isprime}(n+2) \le \sum_{n=%d}^{30q} \left([n \bmod 30 = 11]" \
           r" + [n \bmod 30 = 17] + [n \bmod 30 = 29]\right)"
    assert last(r"\forall q \ge 1: " + base % (6, 6) + r" \text{ by algebra}", ns).value is sympy.true
    assert last(r"\forall q \ge 1: " + base % (5, 5) + r" \text{ by algebra}", ns).value is not sympy.true


def test_fast_orbit_searches():
    ns = {}
    run_all(COLLATZ_T + "\n" + r"\sigma(n) = \min\{k \ge 1 : T^{\circ k}(n) < n\}", ns)
    import time
    t = time.time()
    assert last(r"\max_{2 \le n \le 100000} \sigma(n)", ns, timeout=120).value == 135
    assert time.time() - t < 30
    assert last(r"\max_{2 < n < 10} \sigma(n)", ns).value == 7          # open ends: n = 3 … 9


def test_local_count_single_variable_and_binder_forms():
    assert last(r"\forall j \ge 1: \sum_{a=1}^{j} [(j - a) \bmod (j + 1) \ne 0] = j - 1 \text{ by algebra}").value is sympy.true
    assert last(r"\forall j \ge 1: \sum_{a=1}^{j} [(j - a) \bmod (j + 1) \ne 0] = j \text{ by algebra}").value is not sympy.true
    assert last(r"\forall q \ge 3, 1 \le n \le q - 1: \sum_{a=1}^{q-1} [(n - a) \bmod q \ne 0] = q - 2 \text{ by algebra}"
                ).value is sympy.true


def test_prime_lemma_variables_kept_as_is_are_checked():
    ns = {}
    run_all(r"\forall p \text{ prime}, k \ge 0: k^p \equiv k \pmod p \text{ by induction on } k", ns)
    assert last(r"\forall p \ge 2, k \ge 0: k^p \equiv k \pmod p \text{ by induction on } k", ns).value is not sympy.true
    assert last(r"\forall p \text{ prime}, k \ge 0: (k+1)^p \equiv k + 1 \pmod p \text{ by algebra}", ns).value is sympy.true


def test_chaining_an_upper_bound_and_an_equation():
    ns = {}
    B3 = r"\left([n \bmod 30 = 11] + [n \bmod 30 = 17] + [n \bmod 30 = 29]\right)"
    run_all(r"\forall n \ge 6: \operatorname{isprime}(n)\operatorname{isprime}(n+2) \le [n \bmod 30 = 11] + [n \bmod 30 = 17]"
            r" + [n \bmod 30 = 29] \text{ by cases on } n \bmod 30", ns, timeout=300)
    run_all(r"\forall q \ge 1: \sum_{n=6}^{30q} \operatorname{isprime}(n)\operatorname{isprime}(n+2) \le \sum_{n=6}^{30q} " + B3
            + r" \text{ by algebra}", ns)
    run_all(r"\forall q \ge 1: \sum_{n=6}^{30q} " + B3 + r" = 3q \text{ by induction on } q", ns, timeout=120)
    lhs = r"\forall q \ge 1: \sum_{n=6}^{30q} \operatorname{isprime}(n)\operatorname{isprime}(n+2) \le "
    assert last(lhs + r"3q \text{ by algebra}", ns).value is sympy.true
    assert last(lhs + r"3q - 1 \text{ by algebra}", ns).value is not sympy.true
