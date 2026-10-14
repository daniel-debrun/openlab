r"""The openlab LaTeX grammar (Lark, Earley, dynamic lexer).

Design rules, each of which removes an ambiguity *structurally* rather than by priority:

* Implicit multiplication is ``nt power -> imul``: the right factor is a ``power`` (never
  signed), so ``a x - b y`` is ``a*x - b*y`` and never ``a*x*(-b)*y``.
* ``f(x)`` is a call and ``2(x)`` / ``x^2(x+1)`` are products because a name followed by
  ``(`` (optionally via ``'``, ``^{-1}`` or ``^{(n)}``) lexes as a distinct terminal
  (``FSYMBOL``) that only ``func_call`` accepts.
* A *tail* -- a bare-argument function (``\sin x``) or a big operator (``\int``, ``\sum``,
  ``\lim``, ``\frac{d}{dx}``, ``\max_x``, ``\nabla``) -- swallows the product to its right,
  so it may only be the LAST factor of an implicit product. ``\sin x y`` is ``sin(x*y)``,
  ``\sin x \cos x`` is ``sin(x)*cos(x)``, and after a big operator only ``+``/``-`` can
  follow (``\sum k \cdot 2`` sums ``k*2``).
* ``d`` followed by a letter is never a symbol: ``dx`` is the ``DIFF`` keyword, which only
  an integral or a derivative may consume.
* ``\frac{d}{dx}`` never parses as a fraction because ``{dx}`` is not an expression.
* Bars: ``|x|`` is normalised to ``\lvert x \rvert`` before parsing; ``\|x\|`` is a norm.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

from lark import Lark

GREEK_NAMES = (
    "varepsilon|vartheta|varphi|varrho|varsigma|varpi|alpha|beta|gamma|delta|epsilon|"
    "zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|rho|sigma|tau|upsilon|phi|chi|psi|omega|"
    "Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega|hbar|ell|aleph|imath|jmath"
)

# Functions that take a bare argument (``\sin x``) or a parenthesised one.
FUNC_NAMES = (
    "arcsinh|arccosh|arctanh|arcsin|arccos|arctan|arccot|arcsec|arccsc|sinh|cosh|tanh|coth|"
    "sech|csch|sin|cos|tan|cot|sec|csc|ln|lg|log|exp|det|Re|Im|arg|sgn|deg|dim|ker|Pr"
)
# n-ary functions: only the parenthesised form.
NARY_NAMES = "max|min|gcd|lcm"
# Optimisation-style big operators: ``\max_{x} f``, ``\sup S``, ``\arg\max_x f``.
OPTIM = r"\\arg\s*\\(?:max|min)|\\(?:argmax|argmin|max|min|sup|inf|limsup|liminf)(?![a-zA-Z])(?!\s*(?:\(|\\left\())"
DECO_NAMES = (
    "widehat|widetilde|overline|underline|boldsymbol|mathring|mathcal|mathfrak|mathbb|"
    "mathsf|mathtt|mathbf|dddot|ddot|breve|check|acute|grave|tilde|bar|hat|vec|dot|bm"
)

# "applied" = followed by ( or [ (optionally after ', ^{-1} or ^{(n)}), so `f(x)` is a call.
_CALLMOD = (r"(?:\s*(?:'+|\^\s*\{\s*-\s*1\s*\}|\^\s*\{\s*\(\s*(?:\d+|[A-Za-z])\s*\)\s*\}"
            r"|\^\s*\d|\^\s*\{\s*\d+\s*\}|\^\s*\{\s*\\circ(?![a-zA-Z])[^{}]*\}))?")
_CALL = r"(?=" + _CALLMOD + r"\s*(?:\(|\[|\\left\s*[\(\[]))"
_NOCALL = r"(?!" + _CALLMOD + r"\s*(?:\(|\[|\\left\s*[\(\[]))"

_DECO_BODY = r"\s*(?:\{(?:[^{}]|\{[^{}]*\})*\}|\\[a-zA-Z]+(?![a-zA-Z])|[a-zA-Z])"
_NOT_TRANSFORM = r"(?!\\mathcal\s*\{\s*[LF]\s*\}(?:\s*\^\s*\{\s*-\s*1\s*\})?\s*(?:\\\{|\\left\s*\\\{|\[|\\left\s*\[))"

GRAMMAR = r"""
    ?start: quant
          | stmt

    ?stmt: logic ("," logic)+                         -> tuple_
         | logic "\\mid" logic                         -> divides
         | logic "\\nmid" logic                        -> ndivides
         | logic

    quant: QUANT relation ("," | ":" | ".") logic     -> quant
         | QUANT relation                              -> quant_bare

    // connectives bind in the usual order: \land tighter than \lor, then \implies, then \iff
    ?logic: logic IFFOP limp                          -> logic
          | limp
    ?limp: lor IMPOP limp                             -> logic
         | lor
    ?lor: lor OROP land                               -> logic
        | land
    ?land: land ANDOP lunit                           -> logic
         | lunit
    ?lunit: NEG lunit                                 -> lnot
          | relation
    // a parenthesised proposition: needs a connective or a comparison, so it never
    // competes with a parenthesised expression
    ?boolx: logic IFFOP limp                          -> logic
          | lor IMPOP limp                            -> logic
          | lor OROP land                             -> logic
          | land ANDOP lunit                          -> logic
          | NEG lunit                                 -> lnot
          | setexpr (REL setexpr)+                    -> chain
          | setexpr SETOP expr                        -> setop    // U \setminus (A \cup B)

    ?relation: setexpr (REL setexpr)+                 -> chain
             | setexpr EQUIV setexpr pmod             -> congruence
             | setexpr EQUIV setexpr                  -> chain
             | setexpr
    pmod: "\\pmod" fracarg                            -> pmod
        | "(" "\\bmod" expr ")"                       -> pmod

    ?setexpr: setexpr SETOP expr                      -> setop
            | expr

    ?expr: expr "+" term                              -> add
         | expr "-" term                              -> sub
         | expr PM term                               -> pm
         | expr EVALBAR evalspec                      -> evalat
         | term

    // ---- products -----------------------------------------------------------
    // `nt` ends in a plain power, `ftl` in a bare-argument function chain, `btl` in a
    // big operator. Juxtaposition only continues an `nt`; an explicit operator may
    // continue `nt` or `ftl` but never `btl` (the big operator's body owns its right).
    ?term: SIGN term                                  -> signed
         | PM term                                    -> pm_unary
         | chain

    ?chain: nt
          | ftl
          | btl

    ?cont: nt
         | ftl

    ?nt: cont MULOP sfactor                           -> mul
       | cont TIMES sfactor                           -> times
       | cont DIVOP sfactor                           -> div
       | cont MODOP sfactor                           -> modop
       | cont KRON sfactor                            -> kron
       | cont COMPOSE sfactor                        -> compose
       | nt power                                     -> imul
       | ftl fpp                                      -> imul
       | power

    // after a bare-argument function, a parenthesised call starts a new factor:
    // `\ln t \ln(n-t)` = ln(t)·ln(n−t) (the call cannot be part of the bare argument)
    ?fpp: func_paren "^" exponent                     -> pow
        | func_paren

    ?ftl: cont MULOP sfunc                            -> mul
        | cont TIMES sfunc                            -> times
        | cont DIVOP sfunc                            -> div
        | nt funcchain                                -> imul
        | funcchain

    ?btl: cont MULOP sbig                             -> mul
        | cont TIMES sbig                             -> times
        | cont DIVOP sbig                             -> div
        | nt bigop                                    -> imul
        | ftl bigop                                   -> imul
        | bigop

    ?sfactor: SIGN sfactor                            -> signed
            | power

    ?sfunc: SIGN sfunc                                -> signed
          | funcchain

    ?sbig: SIGN sbig                                  -> signed
         | bigop

    funcchain: funcbare+                              -> funcchain

    ?power: postfix "^" exponent                      -> pow
          | postfix "^" SPECIALEXP                    -> specialexp
          | symbol "^" exponent "_" subpart           -> supsub
          | postfix

    ?exponent: SIGN exponent                          -> signed
             | DIGIT                                  -> number
             | group
             | subscripted
             | symbol
             | fname
             | CONST                                  -> const

    ?postfix: postfix SFACT                           -> factorial
            | postfix DFACT                           -> factorial2
            | postfix PRIMES                          -> prime
            | postfix "\\%"                           -> percent
            | atom

    ?atom: NUMBER                                     -> number
         | CONST                                      -> const
         | frac
         | root
         | binom
         | ncr
         | func_paren
         | absval
         | norm
         | floorceil
         | inner
         | matrix
         | cases
         | set_
         | interval
         | func_call
         | subscripted
         | symbol
         | "(" expr ")"
         | "(" boolx ")"
         | "\\left(" boolx "\\right)"
         | "(" expr ("," expr)+ ")"                   -> tuple_
         | "\\left(" expr "\\right)"
         | "\\left(" setexpr SETOP expr "\\right)"   -> setop
         | "\\left(" expr ("," expr)+ "\\right)"      -> tuple_
         | "[" expr "]"
         | "[" boolx "]"                              -> iverson
         | "[" expr "\\mid" expr "]"                  -> iverson_mid
         | "[" expr "\\nmid" expr "]"                 -> iverson_nmid
         | "\\left[" boolx "\\right]"               -> iverson
         | "\\left[" expr "\\right]"
         | "[" expr "]" evalspec                      -> evalat
         | "\\left[" expr "\\right]" evalspec         -> evalat
         | "\\left." expr "\\right|" evalspec         -> evalat
         | TRANSFORM set_                             -> transform_
         | TRANSFORM "[" expr "]"                     -> transform_
         | TRANSFORM "\\left[" expr "\\right]"         -> transform_
         | UNIT                                       -> unit
         | derivof
         | partialof
         | group@ATOMX@

    group: "{" expr "}"                               -> group

    // ---- named functions -----------------------------------------------------
    funcbare: FUNC fpow? bareargs                     -> func_bare
            | FUNC fpow? funcbare                     -> func_nest
            | FUNC "_" subatom fpow? bareargs         -> func_base_bare
            | FUNC "_" subatom fpow? funcbare         -> func_base_nest
            | FUNC "_" subatom fpow? func_paren       -> func_base_nest
            | OPNAME bareargs                         -> op_bare
            | OPNAME "_" subatom bareargs             -> op_sub_bare
    func_paren: FUNC fpow? "(" arglist ")"            -> func_paren
              | FUNC fpow? "\\left(" arglist "\\right)" -> func_paren
              | FUNC "_" subatom fpow? "(" arglist ")" -> func_base_paren
              | FUNC "_" subatom fpow? "\\left(" arglist "\\right)" -> func_base_paren
              | NARY "(" arglist ")"                  -> func_paren
              | NARY "\\left(" arglist "\\right)"     -> func_paren
              | OPNAME "(" arglist ")"                -> op_paren
              | OPNAME "\\left(" arglist "\\right)"   -> op_paren
              | OPNAME "[" arglist "]"                -> op_paren
              | OPNAME "_" subatom "(" arglist ")"    -> op_sub_paren
    fpow: "^" DIGIT                                   -> fpow
        | "^" group                                   -> fpow
    bareargs: barefactor+                             -> bareargs
    ?barefactor: bareatom "^" exponent                -> pow
               | bareatom "^" SPECIALEXP              -> specialexp
               | bareatom
    ?bareatom: NUMBER                                 -> number
             | CONST                                  -> const
             | frac
             | root
             | subscripted
             | symbol
             | group
             | func_call
             | absval
             | norm
             | matrix

    func_call: fname callmods "(" arglist ")"         -> call
             | fname callmods "\\left(" arglist "\\right)" -> call
             | fname callmods "[" arglist "]"         -> call
    callmods: PRIMES                                  -> cm_prime
            | "^" "{" "-" "1" "}"                     -> cm_inverse
            | "^" "{" COMPOSE expr "}"                -> cm_iter
            | "^" "{" "(" NDER ")" "}"                -> cm_nderiv
            | "^" DIGIT                               -> cm_power
            | "^" "{" NUMBER "}"                      -> cm_power
            |                                         -> cm_none
    arglist: carg ("," carg)*                         -> arglist
    ?carg: logic "\\mid" logic                         -> given
         | logic

    // ---- big operators ---------------------------------------------------------
    ?bigop: integral | sum_prod | limit | deriv | partial | optim | nabla@BIGOPX@

    integral: INT limits? expr DIFF+                  -> integral
            | INT limits? DIFF+                       -> integral_one
            | INT "_" subatom expr DIFF+              -> integral_region
    limits: "_" subatom "^" subatom                   -> limits
          | "^" subatom "_" subatom                   -> limits_rev
    ?subatom: SUBCHAR                                 -> subchar
            | group
            | CONST                                   -> const
            | GREEK                                   -> greek
            | SIGN subatom                            -> signed
    evalspec: "_" "{" anyname "=" expr "}"            -> eval_subs
            | "_" "{" anyname "=" expr "}" "^" "{" anyname "=" expr "}" -> eval_subs2
            | evalhi                                  -> eval_pass
            | limits                                  -> eval_between
            | "_" "{" evalpair ("," evalpair)+ "}"    -> eval_multi
            | "_" "{" "(" expr ("," expr)+ ")" "}"    -> eval_point
            | "_" "{" "(" anyname ("," anyname)+ ")" "=" "(" expr ("," expr)+ ")" "}" -> eval_tuple
    evalpair: anyname "=" expr                        -> evalpair
    evalhi.3: "_" "{" anyname "=" expr "}" "^" subatom -> eval_subs_hi

    sum_prod: SUMPROD "_" "{" anyname "=" expr "}" "^" subatom term  -> sum_prod
            | SUMPROD "^" subatom "_" "{" anyname "=" expr "}" term  -> sum_prod_rev
            | SUMPROD "_" SUBCHAR term                              -> sum_unbounded
            | SUMPROD SUMSUB term                                   -> sum_cond
            | SUMPROD term                                          -> sum_unbounded

    limit: LIM "_" "{" anyname TO expr LIMDIR? "}" term  -> limit

    deriv: FRAC "{" DNUM dpow? "}" "{" DIFF dpow? "}" term   -> deriv_op
    // \frac{df}{dx} names its function: a complete factor, so `\frac{\partial f}{\partial x}\sigma_x` works
    derivof: FRAC "{" DNUM dpow? expr "}" "{" DIFF dpow? "}" -> deriv_of
    partial: FRAC "{" PARTIAL dpow? "}" "{" pvars "}" term   -> partial_op
    partialof: FRAC "{" PARTIAL dpow? expr "}" "{" pvars "}" -> partial_of
           | PARTIAL "_" anyname term                     -> partial_sub
           | PARTIAL "_" "{" anyname "}" term             -> partial_sub
    pvars: pvar+                                      -> pvars
    pvar: PARTIAL anyname dpow?                       -> pvar
    dpow: "^" DIGIT                                   -> dpow
        | "^" "{" NUMBER "}"                          -> dpow

    optim: OPTIMOP "_" "{" optspec "}" term           -> optim
         | OPTIMOP "_" SUBCHAR term                   -> optim_char
         | OPTIMOP "_" GREEK term                     -> optim_greek
         | OPTIMOP term                               -> optim_plain
    ?optspec: relation
            | anyname TO expr                     -> optspec_to
    nabla: NABLA term                                 -> grad
         | NABLA MULOP term                           -> divergence
         | NABLA TIMES term                           -> curl
         | NABLA "^" DIGIT term                       -> laplacian

    // ---- atoms ---------------------------------------------------------------
    frac: FRAC fracarg fracarg                        -> frac
    binom: BINOM fracarg fracarg                      -> binom
         | "{" expr "\\choose" expr "}"             -> binom
    BINOM: /\\[dt]?binom(?![a-zA-Z])/
    ncr: "{" "}" "^" subatom "C" "_" subatom          -> ncr
       | "{" "}" "^" subatom "P" "_" subatom          -> npr
    root: SQRT "[" expr "]" fracarg                   -> nroot
        | SQRT fracarg                                -> sqrt
    ?fracarg: group
            | DIGIT                                   -> number
            | symbol
            | CONST                                   -> const
    absval: "\\left|" setexpr "\\right|"              -> absval
          | "\\lvert" setexpr "\\rvert"               -> absval
    norm: "\\|" expr "\\|" normsub?                   -> norm
        | "\\left\\|" expr "\\right\\|" normsub?      -> norm
        | "\\lVert" expr "\\rVert" normsub?           -> norm
    normsub: "_" subatom                              -> normsub
           | "_" "{" SUBTEXT "}"                      -> normsub
    floorceil: "\\lfloor" expr "\\rfloor"             -> floor
             | "\\lceil" expr "\\rceil"               -> ceil
             | "\\left\\lfloor" expr "\\right\\rfloor" -> floor
             | "\\left\\lceil" expr "\\right\\rceil"   -> ceil
    inner: "\\langle" arglist "\\rangle"              -> inner
         | "\\left\\langle" arglist "\\right\\rangle" -> inner

    set_: LBRACE RBRACE                               -> emptyset
        | LBRACE logic ("," logic)* RBRACE            -> finiteset
        | LBRACE logic ("\\mid" | ":") logic RBRACE   -> condset
        | LBRACE logic ("\\mid" | ":") logic ("," logic)+ RBRACE -> condset_multi
        | LBRACE expr "," DOTS "," expr RBRACE        -> intrange
        | LBRACE expr "," expr "," DOTS "," expr RBRACE -> intrange_step
    LBRACE: /\\left\s*\\\{|\\\{/
    RBRACE: /\\right\s*\\\}|\\\}/
    interval: "[" expr "," expr "]"                   -> interval_cc
            | "[" expr "," expr ")"                   -> interval_co
            | "(" expr "," expr "]"                   -> interval_oc
            | "\\left[" expr "," expr "\\right]"      -> interval_cc
            | "\\left[" expr "," expr "\\right)"      -> interval_co
            | "\\left(" expr "," expr "\\right]"      -> interval_oc

    matrix: MATBEGIN mrow (ROWSEP mrow)* ROWSEP? MATEND -> matrix
    mrow: expr ("&" expr)*                            -> mrow
    cases: "\\begin{cases}" crow (ROWSEP crow)* ROWSEP? "\\end{cases}"  -> cases
    crow: logic "&" logic                             -> crow_cond
        | logic                                       -> crow_plain

    subscripted: symbol "_" subpart                   -> subscripted
    ?subpart: SUBCHAR | GREEK | "{" SUBTEXT "}"

    ?symbol: SYMBOL                                   -> sym
           | GREEK                                    -> greek
           | TEXTNAME                                 -> textname
           | DECO                                     -> deco
    ?fname: FSYMBOL                                   -> sym
          | FGREEK                                    -> greek
          | FTEXTNAME                                 -> textname
          | FDECO                                     -> deco
          | fsubscripted
    fsubscripted: fname "_" subpart                   -> subscripted
    ?anyname: symbol | fname

    // ---- terminals -----------------------------------------------------------
    // `d` followed (after optional space) by a letter is never a symbol: it is DIFF.
    SYMBOL: /[a-ce-zA-Z](?!_(?:[a-zA-Z0-9]|\\[a-zA-Z]+|\{[^{}]*\})@CALL@)@NOCALL@|d(?!\s*(?:[a-zA-Z]|\\(?:@GREEK@)(?![a-zA-Z])))(?!_(?:[a-zA-Z0-9]|\\[a-zA-Z]+|\{[^{}]*\})@CALL@)@NOCALL@/
    FSYMBOL: /[a-zA-Z](?!_)@CALL@|[a-zA-Z](?=_(?:[a-zA-Z0-9]|\\[a-zA-Z]+|\{[^{}]*\})@CALL@)/
    GREEK: /\\@GREEKX@(?:@GREEK@)(?![a-zA-Z])(?!_(?:[a-zA-Z0-9]|\\[a-zA-Z]+|\{[^{}]*\})@CALL@)@NOCALL@/
    FGREEK: /\\@GREEKX@(?:@GREEK@)(?![a-zA-Z])(?!_)@CALL@|\\@GREEKX@(?:@GREEK@|pi)(?![a-zA-Z])(?=_(?:[a-zA-Z0-9]|\\[a-zA-Z]+|\{[^{}]*\})@CALL@)/
    TEXTNAME: /\\(?:mathrm|text|mathit|textit)\{(?!d\}\s*[a-zA-Z])[^{}]*\}@NOCALL@/
    FTEXTNAME: /\\(?:mathrm|text|mathit|textit)\{[^{}]*\}@CALL@/
    DECO: /@NOTTRANSFORM@\\(?:@DECO@)(?![a-zA-Z])@DECOBODY@@NOCALL@/
    FDECO: /@NOTTRANSFORM@\\(?:@DECO@)(?![a-zA-Z])@DECOBODY@@CALL@/
    OPNAME: /\\operatorname\*?\{[^{}]*\}/
    TRANSFORM: /\\mathcal\s*\{\s*[LF]\s*\}(?:\s*\^\s*\{\s*-\s*1\s*\})?(?=\s*(?:\\\{|\\left\s*\\\{|\[|\\left\s*\[))/
    UNIT: /\\openlabunit\{[^{}]*\}/
    FUNC: /\\(?:@FUNC@)(?![a-zA-Z])(?!\s*\\(?:max|min))/
    NARY: /\\(?:@NARY@)(?![a-zA-Z])(?=\s*(?:\(|\\left\())/
    OPTIMOP: /@OPTIM@/
    CONST: /\\(?:pi(?!_(?:[a-zA-Z0-9]|\\[a-zA-Z]+|\{[^{}]*\})@CALL@)|infty|emptyset|varnothing)(?![a-zA-Z])/
    DIFF: /(?:\\mathrm\{d\}|\\text\{d\}|d)\s*(?:\{[a-zA-Z]\}|[a-zA-Z](?![a-zA-Z])|\\(?:@GREEK@)(?![a-zA-Z]))(?:_(?:\{[a-zA-Z0-9]+\}|[a-zA-Z0-9]))?/
    DNUM: /d|\\mathrm\{d\}|\\text\{d\}/
    PARTIAL: /\\partial(?![a-zA-Z])/
    NABLA: /\\nabla(?![a-zA-Z])/
    INT: /\\(?:iiint|iint|int|oint)(?![a-zA-Z])/
    SUMPROD: /\\(?:sum|prod)(?![a-zA-Z])/
    LIM: /\\lim(?![a-zA-Z])/
    TO: /\\(?:to|rightarrow)(?![a-zA-Z])/
    LIMDIR: /\^\s*(?:\{\s*[+-]\s*\}|[+-])/
    FRAC: /\\[dt]?frac(?![a-zA-Z])/
    SQRT: /\\sqrt(?![a-zA-Z])/
    MATBEGIN: /\\begin\{(?:(?:p|b|v|V|small)?matrix|array)\}(?:\{[lcr|: ]*\})?/
    MATEND: /\\end\{(?:(?:p|b|v|V|small)?matrix|array)\}/
    ROWSEP: "\\\\"
    EVALBAR: /\\[bB]igg?\|/
    MULOP: /\*|\\(?:cdot|ast)(?![a-zA-Z])/
    TIMES: /\\times(?![a-zA-Z])/
    DIVOP: /\/|\\div(?![a-zA-Z])/
    MODOP: /\\(?:bmod|mod)(?![a-zA-Z])/
    KRON: /\\otimes(?![a-zA-Z])/
    COMPOSE: /\\circ(?![a-zA-Z])/
    PM: /\\(?:pm|mp)(?![a-zA-Z])/
    REL: /<|>|=|\\(?:leq|le|geq|ge|neq|ne|notin|in|subseteq|subset|supseteq|supset|propto|simeq|sim|cong|gg|ll|approx)(?![a-zA-Z])/
    EQUIV: /\\equiv(?![a-zA-Z])/
    DOTS: /\\(?:ldots|dots|cdots)(?![a-zA-Z])/
    SUMSUB: /_\s*\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}/
    SETOP: /\\(?:cup|cap|setminus|sqcup|triangle)(?![a-zA-Z])/
    ANDOP: /\\(?:land|wedge)(?![a-zA-Z])/
    OROP: /\\(?:lor|vee|oplus|veebar)(?![a-zA-Z])/
    IMPOP: /\\(?:implies|Rightarrow|Longrightarrow)(?![a-zA-Z])/
    IFFOP: /\\(?:iff|Leftrightarrow|Longleftrightarrow)(?![a-zA-Z])/
    NEG: /\\(?:neg|lnot)(?![a-zA-Z])/
    QUANT: /\\(?:forall|exists|nexists)(?![a-zA-Z])/
    SPECIALEXP: /\\(?:circ|dagger|top|ast|star|intercal)(?![a-zA-Z])|\*|\{\s*\\(?:ast|star|dagger|top|circ|intercal)\s*\}|\{\s*(?:\\prime\s*)+\}|(?:\\prime)+/
    SIGN: /[+-]/
    SFACT: /!(?!!)/
    DFACT: "!!"
    PRIMES: /'+/
    SUBCHAR: /[a-zA-Z0-9]/
    NDER: /\d+|[A-Za-z]/
    SUBTEXT: /[^{}]+/
    DIGIT: /\d/
    NUMBER: /\d{1,3}(?:\\,\d{3})+(?:\.\d+)?|(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?/
    THIN: /\\(?:[,;:!]|quad|qquad|\s)/
@RULESX@
    %import common.WS
    %ignore WS
    %ignore THIN
"""

_TEMPLATE = (GRAMMAR.replace("@NOCALL@", _NOCALL).replace("@CALL@", _CALL)
             .replace("@GREEK@", GREEK_NAMES).replace("@DECO@", DECO_NAMES)
             .replace("@DECOBODY@", _DECO_BODY).replace("@FUNC@", FUNC_NAMES)
             .replace("@NARY@", NARY_NAMES).replace("@NOTTRANSFORM@", _NOT_TRANSFORM).replace("@OPTIM@", OPTIM))


@dataclass(frozen=True)
class Dialect:
    """A reading that extends the one grammar; switched on explicitly (``%%openlab --panel``), never inferred.

    A dialect only *adds*: rules and terminals (``rules``), alternatives for big operators
    (``bigops``, e.g. ``" | xsred"``) and atoms (``atoms``), a negative lookahead that keeps a
    Greek letter from lexing as a name where the dialect reads it as an operator (``greek_guard``,
    placed right after the backslash), and the transformer subclass that builds its SymPy.
    """
    name: str
    rules: str = ""
    bigops: str = ""
    atoms: str = ""
    greek_guard: str = ""
    transformer: Any = None
    reading: str = ""                  # the phrase the interpretation line uses for this dialect


#: Registered dialects, by name. The core registers none.
DIALECTS: dict = {}


def register_dialect(d: Dialect) -> None:
    DIALECTS[d.name] = d


def dialect(name: Optional[str]) -> Optional[Dialect]:
    if name is None:
        return None
    if name not in DIALECTS:
        raise KeyError(f"no openlab dialect named {name!r} (registered: {', '.join(DIALECTS) or 'none'})")
    return DIALECTS[name]


def grammar_for(name: Optional[str] = None) -> str:
    d = dialect(name) or Dialect("")
    return (_TEMPLATE.replace("@BIGOPX@", d.bigops).replace("@ATOMX@", d.atoms)
            .replace("@GREEKX@", d.greek_guard).replace("@RULESX@", d.rules))


#: The core grammar (no dialect).
GRAMMAR = grammar_for(None)


@lru_cache(maxsize=None)
def build_parser(ambiguity: str = "resolve", dialect_name: Optional[str] = None) -> Lark:
    """The shared parser, one per dialect. ``ambiguity="explicit"`` is used by the test-suite audit only."""
    return Lark(grammar_for(dialect_name), parser="earley", lexer="dynamic", ambiguity=ambiguity,
                start="start", maybe_placeholders=False, propagate_positions=False)
