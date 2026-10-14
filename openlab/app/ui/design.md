# Design — openlab app

The locked design system for the web app. Every page and panel reads this file
before changing its look. Extend or amend this file when the system needs to
grow; do not re-pick per page.

## Genre
modern-minimal, the instrument-panel register: cool near-white (or near-black)
paper, one blue accent, visible but thin borders, tight radii, no ornament.
Two voices on one screen: the **chrome** (sans, small, quiet) and the
**document** (prose at reading size, headings in the same face as the math).

## Macrostructure family
- App shell: Workbench. Top bar · tree · document · agent drawer · status
  bar. The document column is the only one the eye should land on.
- Start screen (no document open): two-column — statement, studies and the
  agent left; the agents summary (states, spend, what waits on you, flagged
  agents) and what-you-get right. Collapses to one column under 900px.
- Document views (a study, a file): Long Document. One primary action per
  screen; everything else is secondary or in a menu.
- Top-level views (Agents, Tools): a list and a detail, plain-language empty
  states. Agents is the tree (indented by parent) and the inbox on the left,
  the picked agent's log, brief and budget on the right; the same view sits in
  a study's Agents tab, filtered to that study.

## Theme
The palette lives in `src/styles/tokens.css` and is test-enforced
(`tests/test_ui_tokens.py`): hex only there, light and dark declare the same
names, contrast floors, and the proof/disproof luminance match. Do not add
colours anywhere else. Verdict hues (`--vd-*`) and system hues (`--sys-*`)
never mix; red means "something broke" and nothing else.

## Typography
- Display `--font-display`: KaTeX_Main (Computer Modern), bundled from the
  katex package's fonts (two weights; the app typesets no math). Headings, the
  start-screen statement, view titles. Roman only; weight 400 or 700; never
  under 20px.
- Body/chrome `--font-sans`: IBM Plex Sans 400/500/600, bundled from
  `@fontsource/ibm-plex-sans` (latin). Nothing loads from the network.
- Data `--font-mono`: IBM Plex Mono 400/500, bundled. Numbers, names, paths,
  code, the terminal. Tabular numerals in every table.
- Scale `--fs-1..8` (11–32px). Chrome labels 11–12px; controls 13px; prose
  16px; prose h2 20px, h1 26px; start statement 32px.

## Spacing
`--sp-0..9` (0–48px, 2px granularity at the low end). Named tokens only.

## Motion
≤150ms, `--mo-ease`, on hover/focus/disclosure only, never on data arriving.
Reduced motion collapses everything to 1ms. Focus rings never animate.

## Microinteractions
Silent success (a toast only when nothing on screen changes). Confirm only
irreversible acts (a holdout look, stop, delete). Disabled controls say why in their title.

## CTA voice
One primary button per screen: filled accent, `--r-2` radius, verb-first
label ("Run all", "Run the sample analysis"). Secondary: bordered default
button. Tertiary: ghost. Menus ("More") hold everything that is not a
primary task of the screen.

## Agent states and flags
States and flags are system hues, never verdict hues: working `ok`; idle,
done, stopped neutral; waiting `accent` (it needs you); stuck and lost `warn`;
paused `info`; failed `danger`. Flags: STUCK, LOOP, OVER `warn`; WAITING
`accent`; DEAD-END `info`; BYPASS `danger` (an attempt on a person's act is
the one flag that means something broke). A budget bar fills in the accent
and turns `warn` at 90%. A person's act (pause, stop, resume, an inbox
decision) asks for its reason in a dialog and, when the server asks for one,
the passphrase in a hidden field.

## What every surface shares
The wordmark, the accent, the two fonts, the verdict chip (glyph + word +
hue), the button voice, 11px uppercase section labels in the inspector.

## What may differ
Density: a study and the agent tree are dense; the start screen breathes.
