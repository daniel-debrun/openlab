"""Static guard against cross-site scripting in the live page (openlab/app/show): agents write plan titles, notes,
evidence, inbox reasons and choices, roles and questions, and the page renders them with innerHTML."""
import re
from pathlib import Path

SHOW = Path(__file__).resolve().parents[1] / "openlab" / "app" / "show"
PAGE = (SHOW / "index.html").read_text()
CORE = (SHOW / "core.js").read_text()

# agent-, person- or workspace-controlled fields the page reads from the snapshot
FIELDS = ("title|q|role|note|evidence|text|what|detail|from|study|reason|owner|until|op|model|who|hours|cap|head|"
          "name|path|choices|message|by|state|kind|ev|id|pid|v|n|seq|prev|hash|heard|since|turns|tools|running|"
          "waiting|live|pending|decision")
# lines that build plain text, not HTML: textContent, toasts, SVG text through LAB.svg (textContent), the demo's
# own record entries and events
TEXT_ONLY = ("toast(", "textContent", ".title =", "addEntry(", "LAB.emit(", 'E("title"', "const lab =", "why =",
             "w.last =", "hint: `", "text: `${e.kind}", "ps.push(", "LAB.answer = ", "LAB.spawn = ", "LAB.declare = ", "out.push(", "item: `", "const lt =", 'mark[data-c="')


def _esc_chars(src: str, name: str) -> str:
    m = re.search(r"const esc = \([a-z]\) => String\([^;]*?\.replace\(/\[([^\]]+)\]/g", src)
    assert m, f"no esc() in {name}"
    return m.group(1)


def test_esc_escapes_quotes_and_angles():
    for src, name in ((PAGE, "index.html"), (CORE, "core.js")):
        chars = _esc_chars(src, name)
        for c in "&<>\"'":
            assert c in chars, f"esc() in {name} does not escape {c}"
        assert "&#39;" in src


def test_no_raw_field_in_markup():
    bad = []
    for i, line in enumerate(PAGE.splitlines(), 1):
        if any(t in line for t in TEXT_ONLY):
            continue
        line = re.sub(r'esc\(`.*?`\)\}"', "", line)               # a template escaped whole
        line = re.sub(r"esc\(LAB\.fileUrl\(`.*?`\)\)", "", line)   # a link, its parts URL-encoded, then escaped
        for m in re.finditer(r"\$\{((?:[A-Za-z_][\w]*\.)+(?:" + FIELDS + r"))(?:\.(?:slice|toUpperCase|toLowerCase|trim)\([^)]*\))*\}", line):
            bad.append(f"{i}: ${{{m.group(1)}}}")
    assert not bad, "agent-controlled values interpolated into HTML without esc():\n" + "\n".join(bad)


# attributes the page fills from its own computed values (coordinates, classes, flags), never from the data
_ATTR_OK = re.compile(r'^(?:[xy]|c[xy]|d|class|style|data-(?:c|col|h|move|pi|pseg|rf|ver)|draggable|id|type|'
                      r'placeholder|title)="\$\{(?:X\(|Y\(|L |H |T |dn|dp|path|scope|c\.(?:id|k)|i\}|k\}|t\}|st\}|m\}|'
                      r'kind\}|id\}|ps\.length|e\.bad|hot|ok|pSeg|raidF|CLOSED|d [>=]|busy|e\.seq <=|x\.key|d ===)')


def test_attributes_escaped():
    bad = [m.group(0) for m in re.finditer(r'[a-z-]+="\$\{(?!esc\()[^}]*\}', PAGE) if not _ATTR_OK.match(m.group(0))]
    assert not bad, "attribute values from data must go through esc():\n" + "\n".join(bad)


def test_links_from_data_are_escaped():
    for m in re.finditer(r'(?:href|src)="\$\{([^}]*)', PAGE):
        assert m.group(1).startswith("esc("), m.group(0)


def test_new_agent_form_offers_every_role():
    from openlab.agents import claude_setup
    form = re.search(r'<select id="fR">(.*?)</select>', PAGE).group(1)
    colours = re.search(r"const ROLEC = \{(.*?)\};", PAGE).group(1)
    for role in claude_setup.ROLES:
        assert f"<option>{role}</option>" in form, f"the new-agent form has no role {role}"
        assert f"{role}:" in colours, f"no colour for role {role}"
