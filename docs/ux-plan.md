---
title: "UX improvement plan"
description: "Four phases of interface, content and tooling work, written as an implementation brief."
---

# Maiman Studio — UX / product improvement plan, in four phases

This document is an implementation brief. It is written to be handed, one phase at a time, to an
engineer or an AI coding agent working in the `ehsun-sh/maiman-studio` repository. Each phase is
self-contained, ordered by value, and every task states **what to build, where in the code it
lives, exactly how it must behave, what is out of scope, and how it is verified**.

Read **Section 0** before starting any phase. It carries the project's non-negotiable rules and a
map of the code; a task that breaks a rule in Section 0 is wrong even if it works.

---

## 0. Ground rules and code map (read first, every phase)

### 0.1 What the project is

Maiman Studio is an open-source simulator for optical communication links and photonic systems.
The Python engine (`src/maiman/`) is the product; the studio (`src/maiman/studio/index.html`) is a
browser client of the engine's public API, served by `maiman serve` (`src/maiman/server.py`,
standard library HTTP, no framework). Users: practising link engineers coming from OptiSystem or
VPIphotonics (primary), graduate researchers (secondary), students (tertiary). The use scene is
"change a parameter, run, look at the plot, change it again, fifty times an hour".

### 0.2 Rules that are already decided (do not re-litigate)

Taken from `PRODUCT.md`, `DESIGN.md`, `CONTRIBUTING.md`. Violating any of these is a failed task.

1. **The studio is one HTML file with no build step**: `src/maiman/studio/index.html` (~46,800
   lines: CSS at lines ~33–1083, markup ~1096–1760, baked JSON data in
   `<script id="maiman-data">` ~1762–41538, application script ~41540–46812). No npm, no
   bundler, no framework, no TypeScript. Plain ES2020+ JavaScript, hand-drawn SVG canvas, 2-D
   canvas plots. The page must still open straight off disk (`file://`) with no server behind it.
2. **The page fetches nothing from the network.** No CDN scripts, no fonts, no KaTeX. A test
   (`test_formulas_are_drawn_without_fetching_anything`) enforces this. Everything is inline.
3. **Every number on screen comes from the engine.** Never invent, fit, cache, or extrapolate a
   value in the browser. Result data is reduced in the engine; the browser receives binned
   histograms and scalars, never raw sample buffers.
4. **The GUI gets no privileged access.** If a feature is not reachable from the public Python API
   (`maiman.Graph`, `maiman.sweep`, `maiman.load/save`, `maiman.manifests()`), add it to the API
   first, then call it from the server, then from the page.
5. **The palette is generated from `maiman.manifests()`**, never hand-written. Anything the
   inspector needs to know about a parameter (unit, range, choices, `applies_when`, and in Phase 2
   `group`) is declared on the `Param` in Python and carried by the manifest. No tables in the
   page that duplicate model facts.
6. **Never show a control the engine cannot back.** A menu row that does nothing, a field that is
   ignored, a button with no handler: all forbidden. A test
   (`test_every_menu_row_runs_something`) requires every `role="menuitem"` to be named by an
   `onMenu(...)` call. Rows that do not apply are **disabled, not hidden**.
7. **Two grounds, paper (default) and graphite.** Every colour is a token on `:root`; graphite
   tokens appear twice (under `@media (prefers-color-scheme: dark)` guarded as
   `:root:not([data-theme="light"])`, and under `:root[data-theme="dark"]`). No colour may be
   defined only inside a media or `[data-theme]` block. Plots read tokens at draw time via
   `getComputedStyle` and are redrawn on theme change. Text contrast ≥ 4.5:1 (3:1 for large text)
   in both grounds. Port hues are the identity: optical `--optical`, electrical `--electrical`,
   binary `--binary`, symbol `--symbol`, metric `--metric` (plus `soft`); never add a second
   accent palette.
8. **Density over generosity.** No consumer-app whitespace, no large type, no decorative
   animation. The layout (menubar 34px / toolbar 40px / palette 216px | canvas | inspector 260px /
   dock 292px / status bar 24px) is familiar on purpose. Do not move regions.
9. **Keyboard shortcuts live in one table**, `SHORTCUTS` (~line 44347). Both the keydown handler
   and Help → Keyboard shortcuts read it. Never hard-code a shortcut elsewhere.
10. **The `.maiman` project document is the single source of truth.** `PROJECT` (a deep copy of
    `DATA.project`) is what the canvas draws, what the inspector edits, and what Run posts. Any new
    per-project state (Phase 1 autosave, Phase 4 subcircuits) goes into that document, and the
    engine must keep ignoring keys it does not know (`notes` and `annotations` are the precedent).
11. **Opening a file never runs code and never reaches the server.** Files are read locally via
    the browser's file input (`test_opening_a_project_does_not_reach_the_server`).
12. **Tests are mandatory and check the page from its text**, not from a browser. See
    `tests/test_studio.py` (85+ tests, each a sentence-long name describing the claim). Each new
    feature adds at least one test in that style. Where behaviour must be checked in a real
    browser, write a Playwright script under `docs/` or `tools/` the way
    `docs/capture_screenshots.py` is written, but the CI gate stays `pytest`.
13. **The one command that decides** (run it before every commit, whole, no subset):
    ```bash
    pip install -e ".[dev]" && ruff check . && ruff format --check . && mypy && pytest
    ```
14. **After any change to a `Param` or to a component**, regenerate and splice the baked data:
    ```bash
    python examples/python/export_ui_data.py
    ```
    then copy `examples/python/ui_data.json` verbatim into the `<script id="maiman-data">` block of
    `index.html` (tests `test_the_page_carries_the_current_export` and
    `test_the_baked_manifests_are_the_live_ones` fail otherwise). Then
    `python tools/gen_reference_docs.py` for the component reference docs.
15. **Document every decision in `DESIGN.md`**, as a new numbered subsection in the style of §8a–§8h
    (what changed, why, what was measured, what was deliberately not done). A feature without a
    DESIGN.md note is incomplete. User-facing behaviour also goes into `docs/studio.md`.
16. **Commit messages** are plain imperative sentences describing the change (see `git log`).
    One task per commit where practical. Never include model names or tool names in commits.
17. **No physics changes.** Nothing in `src/maiman/components/` changes behaviour in this plan;
    only declarative metadata (`Param.group`, `doc` strings) is touched, and a test must prove
    default outputs are unchanged.

### 0.3 Code map of `index.html` (approximate line numbers at 0.17.0; search by name, not number)

| Thing | Where | Notes |
| :--- | :--- | :--- |
| Colour tokens | CSS `:root` ~34–248 | paper, then graphite twice |
| Menubar markup | ~1098–1340 | `File / Examples / Edit / Simulate / View / Help`, `#theme-btn`, `#kernel-dot`, `#run-bar`, `#project-name` |
| Toolbar markup | ~1345–1465 | `#run-btn #stop-btn #sweep-btn`, tools `#tool-select #tool-pan #tool-note #tool-rect #tool-ellipse #tool-arrow #notes-btn`, zoom, 8 align buttons, `.toolbar-right` run-context readouts |
| Body | ~1468–1520 | `.palette` (`#palette-search`, `#palette-list`), `.canvas-col` (`#graph` SVG, `#canvas-hint`, `<aside id="lesson">`), `.inspector` |
| Dock | ~1520–1700 | `.dock-head` tabs `.tab[data-pane=con|eye|osa|scope|sens|smx|log]`, `.dock-metrics` (`#m-evm #m-snr #m-ber #m-err`, provenance badge), `.dock-body` panes |
| Status bar | ~1700–1760 | `#sb-sel`, counts |
| `DATA`, `PROJECT`, `NODES`, `EDGES`, `VALUES`, `LAYOUT`, `SESSION` | ~41543–41666 | `SESSION.live / hasRun / reference / results / plots / hidden` |
| Palette | `buildPalette()` ~41741 | `CATEGORY_ORDER`, `CATEGORY_PORT` |
| Selection | `selected`, `selection` set, `selectOnly/selectAlso/clearSelection` ~41813 | |
| Undo/redo | `HISTORY`, `FUTURE`, `snapshot()`, `currentState()`, `restore()`, `undo()`, `redo()`, `afterEdit()` ~41887–42010 | state = nodes, edges, layout, annotations, notes, project name |
| Editing | `addComponent`, `removeNode`, `removeEdge`, `connectionProblem`, `connect`, `flashHint`, `svgPoint`, `drawGraph` ~42013–42260 | |
| Inspector | `applies`, `fieldFor`, `conditionText`, `renderInspector`, `bindFields`, `spreadLinkWide`, `LINK_WIDE` ~42279–42515 | |
| Plots | `drawEye*`, `drawConstellation*`, `drawSpectrum*`, `drawScope`, `drawSens`, `drawSmatrix`, `drawSweep`, `drawNothing`, `drawNotRun`, `tilesFor`, `traceBar`, `visibleTraces`, `SERIES` ~42516–43450 | all read tokens at draw time |
| Results | `fillResults`, `clearReadouts`, `LOG`, `logLine`, `showTab`, `logRun`, `describe`, `markProvenance`, `redrawActivePlot` ~43453–43770 | |
| Theme | `applyTheme` ~43770 | sessionStorage key `maiman-theme` |
| Viewport | `VIEW_W/H`, `zoom`, `view`, `setZoom`, `resetView`, `applyView`, `rememberView`, `restoreView` ~43799–43910 | viewBox-based pan/zoom; `VIEW_KEY` in sessionStorage |
| Pointer / tools | `capture`, `release`, `tool`, `setTool`, `alignSelection`, `ALIGN_OPS`, `applyBand`, `endPointer`, `drawPendingWire` ~43918–44340 | |
| Keyboard | `MOD`, `SHORTCUTS`, `onEscape`, keydown listener ~44344–44410 | |
| Teaching | `texToMathML`, `renderRich`, annotations (`ANN_*`, `drawAnnotations`, …), lesson (`renderLesson`, `setLessonOpen`, `revealBlock`, `fitContent`) ~44411–45120 | `fitContent()` at ~45090 already computes a fit-to-content view and is the basis of "Zoom to fit" |
| Files | `projectName`, `currentDocument`, `saveProject`, `openProject`, `projectProblem`, `loadProject`, `newProject`, `saveProjectAs`, `duplicateSelected`, `deleteSelection` ~45174–45450 | Save = `<a download>` of JSON |
| Dialogs, menus | `openDialog`, `showShortcuts`, `showAbout`, `MENUS`, `openMenu`, `closeMenus`, `showMenu`, `onMenu`, `setRow`, `refreshMenuState`, `TEMPLATES` ~45455–45770 | |
| Sweep | `SWEEP`, `numericParams`, `fillSweepForm`, `seedSweepRange`, `sweepValues`, `applySweep`, `defaultMetric`, `sweepSeries`, `fillSweepSummary`, `logSweep`, `drawSweep` ~45778–46320 | POST `/api/sweep` `{project, axis:{node,parameter,values}, runs}` |
| S-matrix | `SMX`, `spectralNodes`, `fillSmatrixForm`, `runSmatrix` ~45826–46045 | POST `/api/spectrum` |
| Run | `RUN`, `setBusy`, `showProgress`, `kernel`, `applyRun` ~46320–46700 | streams `/api/run/stream`; `applyRun(payload, seconds)` fills `SESSION` |
| Helpers | `escapeHtml`, `truncate`, `scalarText`, `fmt`, `si`, `dbm` ~46704+ | |

Server (`src/maiman/server.py`): routes `GET /api/health`, `GET /api/manifests`,
`POST /api/run`, `POST /api/run/stream`, `POST /api/sweep`, `POST /api/spectrum`; functions
`run_project`, `run_sweep`, `run_spectrum`, `_stream_run`, `_execute`. Loopback by default.
Encoding of results: `src/maiman/encoding.py` (`encode`, `scalars`, `encode_results`).
Project format: `src/maiman/project.py` (`graph_to_dict`, `graph_from_dict`, `save`, `load`),
documented in `docs/projects.md`. Parameters: `Param` / `BoolParam` in `src/maiman/component.py`
(`to_dict()` is what the manifest carries). Examples and lessons: `examples/python/*.py`,
`examples/python/lessons.py`, `examples/maiman/*.maiman`, catalogue in `docs/EXAMPLES_ROADMAP.md`.

### 0.4 Working method for each task

1. Read the DESIGN.md sections relevant to the area you are touching.
2. Implement in the smallest number of places; reuse existing helpers (`flashHint`, `logLine`,
   `openDialog`, `snapshot`/`afterEdit`, `setRow`, `onMenu`, `fitContent`).
3. Add the test(s) named in the task to `tests/test_studio.py` (or `tests/test_server.py`,
   `tests/test_project.py` where stated), in the existing style: a function whose name is the
   claim, reading `STUDIO.read_text()` and asserting on the text, or running `run_project` on a
   document.
4. Check both grounds by eye (switch with the sun/moon button) and at 1440×900 and 1180×800.
5. Write the DESIGN.md subsection and the `docs/studio.md` paragraph.
6. Run the full command in 0.2 rule 13. Commit.

---

## Phase 1 — Protect the user's work and complete the editor basics

**Goal.** After this phase, nothing the user does can silently lose work, and every editing gesture
a user of Blender/VS Code/EDA tools reaches for by reflex exists. This is the foundation the later
phases stand on; nothing here needs engine changes except one small server addition (1.12).

**Definition of done for the phase.** All tasks 1.1–1.14 merged with tests; DESIGN.md gains a
section "§9. Phase 1: the editor you can trust"; `docs/studio.md` "Editing" and "Open, save and
examples" updated; full check command green on Python 3.11–3.13.

### 1.1 Dirty state, unsaved-changes guard, and the title marker

**Behaviour.**
- Introduce a single boolean `DIRTY` (module-level `let dirty = false`) with two helpers:
  `markDirty()` and `markClean()`. `markDirty()` is called from exactly one place:
  `afterEdit()` when `record` is true (every real edit already flows through it, including
  parameter edits via `bindFields`, annotation and lesson edits). `markClean()` is called at the
  end of `saveProject()`, `loadProject()`, `newProject()`, and after opening a template.
- Undo back to the saved state does **not** need to clear the flag (keeping it simple is
  acceptable; say so in DESIGN.md).
- The project name in the menubar (`#project-name`) shows a leading `●` when dirty:
  `● coherent_16qam.maiman`. The document `<title>` becomes `● name — Maiman Studio` when dirty,
  `name — Maiman Studio` otherwise (currently the title is static; make it follow the project
  name too).
- `window.addEventListener("beforeunload", …)` sets `event.returnValue = ""` only when dirty.
  Nothing else; no custom text (browsers ignore it).
- `File → New` and opening a template or a file **while dirty** first ask with a native
  `window.confirm("Discard unsaved changes to <name>.maiman?")`. Cancel leaves everything as it
  is. Do not build a custom modal for this in Phase 1.

**Tests.** `test_the_page_asks_before_discarding_unsaved_work`: assert the text contains
`beforeunload`, that `markDirty(` is called inside `afterEdit`, and that `newProject` references
`confirm(` guarded by the dirty flag.

**DESIGN.md.** Note why a single flag at `afterEdit` beats per-call-site bookkeeping (same
argument as §8f's align buttons).

### 1.2 Autosave and recovery

**Behaviour.**
- Every time `afterEdit()` runs with `record: true`, debounce 500 ms and write
  `{ name: projectName, savedAt: Date.now(), document: currentDocument() }` to
  `localStorage["maiman-autosave"]`. Wrap every storage access in `try/catch` (private windows
  throw). Do not write when the document has zero nodes and zero annotations and no notes.
- On page load, after the normal bootstrap and only when the page was opened with a server
  (`SESSION.api`) or off disk alike: if an autosave exists and its `document` differs from the
  document about to be shown (compare `JSON.stringify` of `currentDocument()` without `ui`
  positions), show a one-line bar at the top of the canvas (`#recover-bar`, same visual family as
  `.narrow-note`): *"A project you were editing was not saved: **name.maiman**, 14 minutes ago.
  [Recover] [Discard]"*. Recover calls `loadProject(doc, name)` then `markDirty()`. Discard
  deletes the key. The bar is also dismissed by either button and never reappears for that
  autosave.
- `saveProject()` (a successful save) deletes the autosave key.
- Add `File → Recover autosaved project` row that is enabled only when the key exists (use
  `setRow` in `refreshMenuState`), so recovery is reachable later, not only at load.

**Tests.** `test_an_unsaved_project_survives_a_reload`: text contains `maiman-autosave`, the
write happens inside a debounced function called from `afterEdit`, and the new menu row is named
by `onMenu`.

### 1.3 Save to the same file (File System Access API) with download fallback

**Behaviour.**
- Keep the current download path as the fallback. Where `window.showSaveFilePicker` exists
  (Chromium, Edge, and the Electron desktop build), `File → Save as…` opens the native picker
  with `suggestedName: projectName + ".maiman"` and `types: [{description: "Maiman project",
  accept: {"application/json": [".maiman"]}}]`, writes via `createWritable()`, and keeps the
  returned handle in `let fileHandle = null`. Subsequent `File → Save` writes to that handle
  without a dialog. `File → Open…` likewise uses `showOpenFilePicker` when available and stores
  the handle, so Open → edit → Save overwrites the file the user opened. Permission errors
  (`NotAllowedError`) fall back to the download path and log a line saying so.
- Where the API does not exist (Firefox, Safari, `file://` in some browsers), behaviour is exactly
  today's: `Save as…` prompts for a name and downloads; `Save` downloads under the current name.
- `newProject()` and loading a template set `fileHandle = null`.
- The status bar or log says which path was used: `saved coherent_16qam.maiman (written in place)`
  vs `saved coherent_16qam.maiman (downloaded)`.
- Nothing is ever posted to the server (rule 0.2-11 unchanged).

**Tests.** `test_save_writes_in_place_where_the_browser_allows_and_downloads_otherwise`: text
contains both `showSaveFilePicker` and the existing `link.download`, and `fileHandle = null`
appears in `newProject`.

**DESIGN.md.** Record that the picker is feature-detected, not UA-sniffed.

### 1.4 Zoom to fit

**Behaviour.**
- `fitContent()` already exists (~line 45090) and computes the view that frames all nodes and
  annotations, respecting the lesson panel inset. Expose it:
  - `View → Zoom to fit`, shortcut `shift+1` (add to `SHORTCUTS`; label "Zoom to fit").
  - A toolbar button after `#zoom-in` with a "fit" icon (four corner brackets, 13×13, same stroke
    style as existing icons), `title="Zoom to fit (shift+1)"`.
  - Double-clicking the canvas bed (not a block, not an annotation) also fits.
- With no content, fit calls `resetView()` (already does).
- Opening a project or template from the Examples menu calls `fitContent()` instead of
  `resetView()` when the project has layout (it currently fits only when a lesson is present;
  make it unconditional, but keep the rule "never zoom past 100 %" that is already inside
  `fitContent`).

**Tests.** `test_zoom_to_fit_is_reachable_from_the_menu_the_toolbar_and_the_keyboard`.

### 1.5 Select all, copy, cut, paste, duplicate with wires

**Behaviour.**
- `Edit → Select all` (`mod+A`): selects every node (via `selectAlso` for each) and every
  annotation. Disabled when the canvas is empty.
- Clipboard model, documented in DESIGN.md: a copied subgraph is `{ nodes: [...], edges: [...],
  annotations: [...], layout: {...} }` containing the selected nodes, **only the edges whose both
  ends are inside the selection**, and the selected annotations. It is kept in a module-level
  `CLIPBOARD` variable **and** written to the system clipboard as JSON text (`navigator.clipboard
  .writeText`, wrapped in try/catch) with a top-level `{"maiman-clipboard": 1, …}` marker so a
  paste into another tab or another Maiman window works, and a paste of arbitrary text is
  ignored.
- `Edit → Copy` (`mod+C`), `Edit → Cut` (`mod+X`, = copy then `deleteSelection()` in one undo
  snapshot), `Edit → Paste` (`mod+V`). Paste: reads the system clipboard if permitted, else the
  in-memory copy; assigns fresh labels via `freeLabel(type)` for every node, remaps edge ends and
  annotation `[[label]]` references accordingly; places the pasted group offset by (+24, +24)
  canvas units from the original, or at the pointer's last canvas position if a paste follows a
  right-click there; selects the pasted group; one undo snapshot. Paste is disabled when the
  clipboard is empty.
- `Edit → Duplicate` keeps its current semantics (no wires) **and** gains `Duplicate with wires`
  (`mod+shift+D`): a copy that also duplicates the internal edges of the selection (same rule as
  copy). Update the DESIGN.md §8a paragraph that explained why Duplicate drops wires: the reason
  stands for single blocks; for a selected subgraph, internal wires are unambiguous.
- Keyboard shortcuts must not fire while typing in an input/textarea/select (the existing
  `typing` guard in the keydown handler applies); `mod+C/X/V` inside text fields must keep their
  native meaning.

**Tests.** `test_a_copied_subgraph_keeps_only_the_wires_inside_it` (pure function
`clipboardFrom(selection)` extracted so it can be exercised via a Node `vm` run of the function
text, the way other tests slice functions out of the page; if that is impractical, assert the
rule in the function body text and add a Playwright check script under `tools/`).
`test_paste_assigns_fresh_labels_and_one_undo_step`.

### 1.6 Context menu on the canvas

**Behaviour.**
- Right-click (or `shift+F10` / the context-menu key) on the canvas opens a menu using the
  existing `.menu-pop` visuals, positioned at the pointer, closed by Escape/click-away like the
  menubar menus (reuse `openMenu` plumbing).
- On a **block**: `Run` ·—· `Cut`, `Copy`, `Duplicate`, `Duplicate with wires`, `Delete` ·—·
  `Disconnect all wires` ·—· `Sweep a parameter…` (opens the Sweep form pre-filled with this
  block, see Phase 2.7) ·—· `Show its S-matrix spectrum` (only for `spectral` blocks; otherwise
  absent, because it would never be enabled) ·—· `Open documentation` (link to
  `https://maimanstudio.com/docs/components/<Type>`, opened via `openExternal`).
- On a **wire**: `Delete wire`.
- On an **annotation**: `Edit text` (notes), `Duplicate`, `Delete`.
- On the **bed**: `Paste` (enabled when the clipboard holds something), `Select all`,
  `Zoom to fit`, `Reset view`, `Add block…` (focuses the palette search).
- Right-button **drag** still pans (DESIGN.md §8e): open the menu on `contextmenu` only if the
  pointer did not move more than 3 px between `pointerdown` and `pointerup`.
- Every row goes through `onMenu(id, action)` so the existing test covers it.

**Tests.** `test_the_canvas_has_a_context_menu_and_every_row_runs_something` (extend the menuitem
scan to the new menu ids).

### 1.7 Keyboard nudge and focusable blocks

**Behaviour.**
- With a selection and focus on the canvas (`#graph` gets `tabindex="0"`), arrow keys move the
  selection by 1 canvas unit, `shift+arrow` by 10, with one undo snapshot per key-repeat burst
  (snapshot on the first keydown, none until keyup).
- `Tab` / `shift+Tab` while the canvas has focus cycles the selection through blocks in reading
  order (by y then x); the focused block gets the existing selection ring. `Enter` on a focused
  block focuses the first inspector field.
- Add to `SHORTCUTS` so they are listed in Help.
- `<html lang="en">` is added. The run completion (`applyRun`) and failure messages are announced
  through an `aria-live="polite"` region (reuse `#canvas-hint`, which already has `role="status"`).

**Tests.** `test_the_canvas_is_reachable_from_the_keyboard`: `tabindex="0"` on `#graph`, `lang=`
on `<html>`, arrow-key rows in `SHORTCUTS`.

### 1.8 Collapsible side panels at any width, and the promised narrow-width overlays

**Behaviour.**
- Two toggles: `View → Palette` (`mod+B`) and `View → Inspector` (`mod+shift+B`), ticked rows
  (use the `data-ticks` pattern of the View menu). Also two small chevron buttons at the outer top
  corners of each panel (inside `.panel-head`). Collapsed state hides the column (`display:none`
  on `.palette` / `.inspector`) and the canvas takes the space; `fitContent()` is **not** called
  automatically (the view must not jump under the user).
- Persist both in `sessionStorage` beside the theme (`maiman-panels`).
- Below 940 px, replace today's "collapse and show a note" with: both panels start collapsed and
  the toggles still work, showing the panel as an **overlay** (`position:absolute`, full height of
  `.body`, `z-index` above the canvas, with a subtle shadow). Remove `.narrow-note` and its CSS.
  Update the DESIGN.md §5 paragraph that promised this.

**Tests.** `test_the_side_panels_can_be_hidden_and_the_narrow_note_is_gone`.

### 1.9 Port legend becomes a status-bar item

**Behaviour.**
- The floating "Ports — optical electrical …" box on the canvas (it overlaps toasts and frames,
  e.g. in `ring.maiman`) moves into the status bar as a compact legend on the right side, before
  the sample count, rendered from `PORT_TYPES` and the port tokens. The canvas-hint toast keeps
  its position. A `View → Port legend` tick row shows/hides it (default shown). Remove the canvas
  element and its CSS. Update the geometry note in DESIGN.md §8 ("the Ports legend must not
  overlap a block") to say it no longer lives on the canvas.

**Tests.** Adjust any existing test that looks for the legend on the canvas;
`test_the_port_legend_is_in_the_status_bar`.

### 1.10 Pre-run validation on the Run button

**Behaviour.**
- Compute `graphProblems()` on every `drawGraph()`: list of `{ id, port, message }` for every
  **required** input with no source (the hollow-port rule that already exists), plus "graph is
  empty". The Run button stays enabled (the engine's refusal is still the authority) but shows a
  small count badge (`2`) in `--warn` when problems exist, with `title` listing them, one per
  line: `fib.in has no source`. Clicking Run with problems runs anyway (unchanged); the log
  already prints the engine's answer.
- The status bar's selection slot shows `2 unconnected inputs` when nothing is selected and
  problems exist.

**Tests.** `test_the_run_button_counts_the_inputs_the_engine_will_refuse`.

### 1.11 Metric strip adapts to the link

**Behaviour.**
- The dock's metric strip (`#m-evm #m-snr #m-ber #m-err` with `#k-*` labels) is rebuilt from the
  run's results, not fixed. Rule: collect, in this priority, up to four metrics that exist in
  `SESSION.results`: constellation analyser → EVM, SNR/MER, BER, SYM ERR; BER analyser →
  Q, Q dB, BER, BIT ERR (this mapping partly exists; make it the single `metricStrip()` function);
  OSNR meter → OSNR; power meter (last one in graph order) → POWER; if nothing measured, show the
  strip empty except the provenance badge. Never show a `—` cell for a metric no block on the
  graph produces. Before any run, derive the strip from the **blocks present** (an OOK link with
  a BER analyser shows `Q — · Q dB — · BER — · BIT ERR —`, so the user sees what will be measured).

**Tests.** `test_the_metric_strip_shows_only_what_the_graph_measures` using `run_project` on
`examples/maiman/wdm_osa.maiman` (no EVM anywhere) and asserting the function's selection logic
via the text or an extracted function.

### 1.12 Sweep form: remove the stale reference readout

**Bug.** In the Sensitivity pane, with no sweep in the session, the side panel still prints
`QPSK at 1e-3  −40 dBm` from the bundled reference sensitivity data while the plot says
"No sweep in this session". Find where the sensitivity side readouts are filled from
`DATA.sensitivity` (search `at 1e-3`) and make them obey the same provenance rule as every pane
(DESIGN.md §8b): shown only when `SESSION.reference` is true and the reference project is open,
cleared by `clearReadouts()`, by `newProject()`, and by any `loadProject`.

**Tests.** `test_the_sensitivity_readouts_are_cleared_with_the_reference_they_describe` modelled on
`test_the_readings_are_cleared_with_the_trace_they_describe`.

### 1.13 Examples menu: numbering and gaps

- `4.6 400ZR over one span` and `4.6 800ZR over one span` share a number. Relabel as
  `4.6a 400ZR over one span` and `4.6b 800ZR over one span` in the menu markup and in
  `docs/EXAMPLES_ROADMAP.md` (the roadmap row 4.6 stays one row; add "(a)/(b)" to the status cell).
- `3.7` and `3.8` exist as scripts (`edfa_gain_tilt.py`, `edfa_transient.py`) but ship no
  `.maiman`, so the menu jumps from 3.6 to 3.9. Make each script write a project with a layout
  and a lesson in `lessons.py` (same four-part pattern as every example in
  `docs/EXAMPLES_ROADMAP.md` "How an example is built"), add the two menu rows, and extend
  `test_every_template_runs` coverage automatically (it is parametrised over templates).
- `test_the_examples_menu_is_in_roadmap_order` must keep passing.

### 1.14 Right-click panning no longer swallows the context menu — regression check

After 1.6, verify in a Playwright run that right-drag pans and right-click opens the menu, in both
grounds. Record the measured drag threshold (3 px) in DESIGN.md.

---

## Phase 2 — The inspector and the parameter loop

**Goal.** The inspector is where the fifty-times-an-hour loop happens. After this phase: long
parameter lists are grouped, numbers are edited with the keyboard and the wheel, a changed value
is visible as changed, a sweep starts from the parameter itself, and the run can follow an edit
without a trip to the toolbar.

**Definition of done.** Tasks 2.1–2.9 merged; `Param` gains `group`; every component with more
than 8 parameters declares groups; `DESIGN.md` "§10. Phase 2: the inspector"; `docs/studio.md`
"Editing" rewritten; `docs/writing-components.md` documents `group`.

### 2.1 `Param.group` in the engine and the manifest

**Engine (`src/maiman/component.py`).**
- Add keyword `group: str = ""` to `Param.__init__` and `BoolParam.__init__`; store as
  `self.group`; `to_dict()` emits `"group"` only when non-empty (the manifest shape stays
  identical for ungrouped parameters, so baked data for untouched components does not change).
- Group names are free text but the convention is Title Case, short: `Linear`, `Dispersion`,
  `Nonlinear`, `Polarization`, `Noise`, `Geometry`, `Advanced`. Group `Advanced` has a special
  meaning in the page (2.2).
- Declare groups on every component with **more than 8 parameters**; at 0.17.0 that is at least
  `Fiber` (29), `GratingCoupler` (29), `DirectlyModulatedLaser` (23), `FiberBraggGrating` (18),
  `LongPeriodGrating` (16), and any other the test below finds. For `Fiber`, the intended grouping
  (adjust names to the actual parameter list):
  - *(no group — always first)*: `length`, `attenuation`
  - `Dispersion`: `dispersion`, `dispersion_slope`, `reference_wavelength`, `beta3`-type terms,
    `phase_index`, `group_index` if present
  - `Nonlinear`: `nonlinearity`, `max_nonlinear_phase`, `cross_phase_modulation`, Raman terms,
    `max_walkoff_slip`, `carry_walkoff`, `carry_carrier_phase`
  - `Polarization`: `pmd`, `cross_polarization`, `coherent_polarization`, `interleave_pmd`,
    birefringence terms
  - `Advanced`: solver tolerances and anything a link engineer sets once per project
  Order inside a group is declaration order. Ungrouped parameters always render first, in
  declaration order, with no header.
- Add a test in `tests/test_core.py` (or a new `tests/test_parameter_groups.py`):
  `test_every_long_parameter_list_is_grouped` — for every registered component with > 8
  parameters, at least half of its parameters carry a group and no group has a single member.
  And `test_grouping_changes_no_default_output`: run `examples/maiman/ook_link.maiman` and
  `coherent_16qam` through `run_project` before and after (compare against the numbers the
  existing example tests already pin; it is enough that those tests keep passing, but state this
  in the commit).
- Regenerate and splice `ui_data.json`; regenerate reference docs; `docs/writing-components.md`
  gains a paragraph on `group`.

### 2.2 Grouped inspector with a collapsible Advanced section

**Page (`renderInspector`, `fieldFor`).**
- Render ungrouped fields, then one `<section class="insp-group">` per group in order of first
  appearance, each with a `<h3 class="side-title">` header. All groups open by default **except
  `Advanced`**, which is collapsed; a header click toggles (`aria-expanded`), and the open/closed
  state per `(component type, group)` is remembered in `sessionStorage["maiman-groups"]`.
- Density rule from DESIGN.md §1 stands: headers are the same 10–11 px uppercase tracking as
  existing `.side-title`; no extra padding beyond 4 px above a header.
- A collapsed group whose fields contain a **non-default value** shows a small count on the
  header (`Advanced · 2 changed`) so a hidden edit is never invisible.
- The `applies_when` greying (DESIGN.md §8g) keeps working inside groups; toggling a flag still
  rebuilds the panel, and the rebuild must restore scroll position and group states.

**Tests.** `test_the_inspector_groups_what_the_manifest_groups` (page text references
`spec.group`, no page-side table of groups: extend
`test_the_page_keeps_no_list_of_which_parameters_are_gated` to also forbid a hard-coded
group table); `test_a_collapsed_group_says_when_it_hides_an_edit`.

### 2.3 Real numeric inputs: step, arrows, wheel, drag

**Behaviour (float parameters, `fieldFor`/`bindFields`).**
- Inputs become `type="text" inputmode="decimal"` (keep text, because `type=number` rejects
  `1e-3` in some locales and strips trailing zeros); add a stepping layer:
  - `↑`/`↓` changes the value by one **step**; `shift+↑/↓` by 10 steps; `alt+↑/↓` by 0.1 step.
  - Step is derived, never declared: `10^floor(log10(|value|)) / 10` for `|value| ≥ 1`
    (80 → step 1; 1550 → 10 — then clamp to at most 1 for wavelengths? No: keep the rule and
    let `alt` give fine steps; document it), `0.01` for `|value| < 1`, and `1` for a value of
    0. Choices-typed parameters (`spec.choices`) keep their `<select>` and arrows move through
    the list.
  - Wheel over a **focused** input only (never on hover, which would hijack page scroll) steps the
    value the same way.
  - Values are clamped to `min`/`max` from the manifest; a clamp flashes the field border
    `--warn` for 300 ms and `flashHint("length cannot go below 0 km")`.
- `Enter` in a field commits the edit (already) **and**, when "Run after edit" (2.4) is off,
  triggers a run if `mod+Enter`. Plain `Enter` moves focus to the next field.
- Each edited value still goes through the existing validation path (engine wins).

**Tests.** `test_a_parameter_can_be_stepped_from_the_keyboard` (text contains the step helper and
arrow handling inside `bindFields`; the derived-step function is extracted as `stepFor(value)`
and unit-tested by evaluating its text in Node via `subprocess` the way the page's other pure
functions are tested, or by a Playwright script if none exist — check `tests/test_studio.py` for
precedent first).

### 2.4 Changed-from-default indicator, reset, and "Run after edit"

**Behaviour.**
- A field whose value differs from `spec.default` gets a 4 px dot in `--optical` before its label
  and a `Reset` affordance: a small `↺` button inside the field row (`title="Reset to default
  (0.2 dB/km)"`), which sets the value to default through the same edit path (undoable).
- The inspector header gains `n changed` next to the version chip when any field differs.
- `Simulate → Run after each edit` (tick row, default **off**, persisted in sessionStorage
  `maiman-autorun`): when on, every committed edit (blur/Enter/select change/toggle) schedules a
  run 300 ms later, coalescing edits; a run already in flight is not interrupted (the next one is
  queued, at most one queued). The toolbar's `Run` button shows `auto` in small text beside it
  when the mode is on. Never auto-run on structural edits (add/remove block or wire).

**Tests.** `test_an_edited_parameter_shows_that_it_was_edited`,
`test_run_after_edit_coalesces_and_never_stacks_runs`.

### 2.5 Cleaner field text: doc de-duplication, bounded ranges only, typical values

**Behaviour.**
- If `spec.doc` is empty, or equals the parameter name ignoring case/underscores (`attenuation` /
  "Attenuation"), render no doc line.
- The `range a … b` line is shown **only** when at least one bound is finite and not the trivial
  `0 … ∞`; for `min: 0` with no max, render `≥ 0` inline after the unit, not a separate line.
- **Content pass on `doc` strings** (engine, no behaviour change): every float `Param` on a
  component in categories Fiber, Amplifiers, Optical Sources, Receivers, Modulators gets a `doc`
  of the form *"What it is. Typical: value unit (context)."* e.g. `attenuation`: "Attenuation
  coefficient. Typical: 0.2 dB/km for SSMF at 1550 nm, 0.35 at 1310 nm." Where a typical value
  depends on a standard, cite it (G.652, G.655). A test
  `test_every_main_line_parameter_names_a_typical_value` checks that each float param in those
  categories has a `doc` containing "Typical" or a numeric value. Regenerate and splice.

### 2.6 Learn-more link and port tooltips

- Inspector header gets a `Docs ↗` link (`openExternal`) to
  `https://maimanstudio.com/docs/components/<Type>` (the URL pattern is in `docs/nav.yml`),
  and the `.chip` label gets a tooltip with the block's class path from the manifest.
- Each port row in the inspector and each port circle on the canvas gets a `title`:
  `in · optical — accepts an optical field` using a fixed description per port type kept in
  one object `PORT_DESCRIPTION` beside `PORT_TYPES`.

### 2.7 Sweep starts from the parameter

**Behaviour.**
- Every float field row in the inspector gets a small `⇅ sweep` icon button (visible on row
  hover and on keyboard focus, always in the accessibility tree) that: opens the Sensitivity pane,
  fills the Sweep form with this node and parameter (`fillSweepForm` already takes the state via
  `SWEEP.axis`; add an argument), seeds the range from the current value (`seedSweepRange`: for a
  value v ≠ 0 use `[v/2, 2v]` with 9 steps, for v = 0 use `[0, 10×step]`; respect min/max), and
  focuses the `From` field. The context-menu row `Sweep a parameter…` (1.6) opens the same form
  with the block selected and the parameter dropdown focused.
- The Sweep form moves out of the 200 px side panel into a **form row across the top of the
  Sensitivity pane** (Block · Parameter · From · To · Steps · Repeats · [Run sweep]), with the plot
  beneath and the summary (`fillSweepSummary`) in the side panel. Reuse the existing controls'
  ids so `test_the_sweep_form_offers_the_axis_the_server_expects` keeps passing.
- Results table: under the plot, a `<table>` of x, y per point (and min/max/mean across repeats
  when `runs > 1`), monospace, with a `Copy as CSV` button (Phase 3 adds file export).

**Tests.** `test_a_sweep_can_be_started_from_the_parameter_it_varies`.

### 2.8 Multi-selection editing

**Behaviour.**
- When the selection contains two or more blocks of the **same type**, the inspector shows that
  type's form with each field showing the common value, or `mixed` placeholder when values
  differ; an edit writes to all selected nodes in one undo snapshot. When types differ, the
  inspector says `4 blocks selected — 2 CW Laser, 2 Mach-Zehnder Modulator` and offers only
  Delete / Align / Duplicate. The link-wide parameter rule (`LINK_WIDE`, `spreadLinkWide`) still
  applies.

**Tests.** `test_blocks_of_one_type_can_be_edited_together`.

### 2.9 Palette improvements that serve the inspector loop

- Hovering a palette row shows a tooltip after 400 ms with: name, category, one-line doc (the
  class docstring's first sentence — add `"summary"` to the manifest from
  `cls.__doc__.split(".")[0]`, engine side), and its ports as coloured chips (`in: electrical ·
  out: optical`).
- A **Recent** section at the top of the palette listing the last 6 types added in this session
  (sessionStorage `maiman-recent-types`), collapsed when empty.
- Palette search also matches category and port type words (`optical`, `dsp`).

---

## Phase 3 — Results you can keep, compare and take away

**Goal.** The results dock turns from a viewer into an instrument: runs are kept and compared,
plots have cursors and readouts, and everything can leave the tool as a file or as code.

**Definition of done.** Tasks 3.1–3.9 merged; `DESIGN.md` "§11. Phase 3: results"; `docs/studio.md`
sections "Runs", "Sweeps", "Export"; new server/API functions documented in `docs/api.md` (via
`tools/gen_reference_docs.py`).

### 3.1 Run history

**Behaviour.**
- `SESSION.history`: an array (max 20, oldest dropped) of
  `{ id, at: Date.now(), seconds, document: currentDocument(), results, plots, captions, label }`
  pushed at the end of `applyRun`. The `label` defaults to `run 7 · 2.75 s`.
- A new dock tab **History** (`data-pane=hist`, add to View menu with a tick row like the
  others). It lists runs newest first as rows: label · time · the metric strip values of that run
  (reuse `metricStrip()` from 1.11 on that run's results) · `[Pin] [Restore] [Diff] [×]`.
- **Restore** puts that run's `document` back on the canvas (through `loadProject`-like logic
  without resetting the view, one undo snapshot) and its plots in the dock, with the provenance
  badge reading `history — run 7` in the same style as `stale`/`reference`.
- **Diff** opens a dialog (`openDialog`) listing parameter differences between that run's
  document and the current document: `fib.length: 80 → 120 km`, plus added/removed blocks and
  wires. The diff function `diffDocuments(a, b)` is pure and tested.
- Memory: each history entry holds reduced data only (histograms ≤ 96×96, spectra ≤ 4096 points
  as they already arrive), so 20 entries stay well under a few MB; state the measured size in
  DESIGN.md.
- History is session-only (not persisted), and `newProject()` clears it after a confirm if it
  is non-empty.

**Tests.** `test_every_run_is_kept_and_can_be_told_from_the_next`;
`test_a_document_diff_names_what_changed` (pure JS function evaluated in Node, or mirror the
algorithm in Python inside the test and compare on fixtures — pick the precedent used in the
file).

### 3.2 Pin and compare

**Behaviour.**
- **Pin** marks one history entry as the reference (`SESSION.pinned`). When a pinned run exists:
  - Curve panes (Spectrum, Sensitivity, Scope, S-matrix) draw the pinned run's traces **under**
    the live ones, dashed, in `--ink-4`, with the trace bar listing them as `pinned · osa`.
    Lines can overlay; density plots cannot (DESIGN.md §8e), so:
  - Histogram panes (Constellation, Eye) show the pinned run as a **second tile** beside the live
    one, captioned `pinned · run 4`.
  - The metric strip shows the delta beneath each value in small type: `EVM 7.39 %  (+0.21)`,
    coloured `--good` or `--bad` by whether lower is better for that metric (define a
    `LOWER_IS_BETTER` set: evm, ber, symbol errors, bit errors; higher is better: Q, SNR, MER,
    OSNR, power).
- `View → Compare with pinned run` tick row toggles the overlay without unpinning.

**Tests.** `test_a_pinned_run_is_drawn_beside_or_under_the_live_one_but_never_mixed`.

### 3.3 Export plots as SVG and PNG

**Behaviour.**
- Each pane's side panel gets `Export ▾` with `SVG`, `PNG (2×)`, `Copy PNG`. Implementation: the
  plots are drawn with the 2-D canvas; add a thin drawing adapter so the same `draw*Into(ctx, w,
  h, …)` functions can draw onto a minimal SVG-writing context object implementing the subset of
  CanvasRenderingContext2D the plots use (`fillRect`, `strokeRect`, `beginPath/moveTo/lineTo/
  stroke/fill`, `fillText`, `save/restore`, `translate`, `setLineDash`, `fillStyle/strokeStyle/
  font/textAlign/textBaseline/lineWidth`). Enumerate the subset by grepping the draw functions and
  assert it in a test (`test_the_plots_use_only_what_the_svg_writer_implements`). PNG uses an
  offscreen canvas at `devicePixelRatio × 2`. Both exports use the **paper** tokens regardless of
  the current ground, because the plots leave for papers (PRODUCT.md), with a `Use current
  ground` checkbox for the occasional dark export.
- Filenames: `<project>-<pane>-<block label>-<yyyymmdd-hhmm>.svg`.
- Exported SVG carries a `<desc>` with the project name, block, run time and engine version from
  `/api/health`.

### 3.4 Export data: CSV / JSON, copy metrics

- Spectrum, Scope, Sensitivity/Sweep, S-matrix: `Export ▾ → CSV` writes the trace(s) as columns
  with a header row naming units (`wavelength_nm, power_dbm_per_12.5GHz`); `JSON` writes the raw
  reduced payload for that block as received from the engine.
- Eye and Constellation: `CSV` writes the histogram matrix with axis vectors in the first row
  and column.
- `Copy metrics` on the metric strip copies `EVM 7.39 %\tSNR 22.6 dB\tBER 5.30e-10` to the
  clipboard; the Log gets `copied 4 metrics`.
- The full run: `File → Export run results (JSON)` writes `{ document, context, results,
  engine_version, at }`.

**Tests.** `test_every_pane_can_leave_as_a_file` (each pane's panel markup contains the export
control and each kind has a writer).

### 3.5 Export as Python

**Engine.** Add `maiman.project.to_python(document: dict) -> str` producing a runnable script:
imports, `SimulationContext(**context)`, one `link.add(Type(**params))` per node with the label as
the variable name (sanitised to an identifier, collisions suffixed), `link.connect(...)` per edge
using the real port names, `results = link.run()`, and a print per metric port found in the
graph (`print(f"{pm.power_dbm:.2f} dBm")` style, using the result attributes the encoder knows).
Parameters are written in the units the manifest quotes (as the document stores them). Add
`GET`-less route `POST /api/export/python` returning `{ "source": "..." }` and a CLI entry
`maiman export-python project.maiman` for scripts. Tests in `tests/test_project.py`:
`test_an_exported_script_runs_and_reproduces_the_run` — export `ook_link.maiman`, `exec` it in a
namespace, compare the printed/returned numbers to `run_project`'s within 1e-9.

**Page.** `File → Export as Python…` opens a dialog with the source in a `<pre>` (monospace,
selectable), `Copy` and `Download .py`. Works only with a server (disabled row without one,
with the reason in `title`).

### 3.6 Resizable and maximisable dock

- A 6 px drag handle between the canvas and the dock changes `--dock-h` (CSS variable, default
  292 px, min 160, max 70 % of viewport); persisted in sessionStorage `maiman-dock`. Plots resize
  via the existing `sizeCanvas` on `ResizeObserver`.
- A `⤢` button at the right end of the dock head toggles **maximise** (`.dock.max`: the dock
  takes the whole body height; canvas hidden), shortcut `mod+shift+M`, Escape restores. Only one
  pane is visible either way; this is for looking at an eye, not a second layout.

**Tests.** `test_the_dock_can_be_resized_and_maximised`.

### 3.7 Cursors and hover readouts on plots

- **Eye**: two vertical and two horizontal draggable cursors (default placed at the decision
  instant and ±½ UI, and at the inner eye edges as the engine reports them if available — check
  `eye` payload fields; if the engine does not report eye height/width, add them to the eye
  encoder as `eye_height`, `eye_width_ps` computed in the engine, with a closed-form test on a
  noiseless NRZ eye). The side panel shows ΔT (ps) and ΔI (µA or mV per the trace unit) between
  cursor pairs, plus the engine's eye height / width / jitter when present.
- **Spectrum / Scope / Sensitivity / S-matrix / Sweep**: hovering shows a crosshair and a readout
  `1548.40 nm · −3.5 dBm` in the plot corner; clicking drops a marker (max 2) and the panel shows
  Δ between markers. Shift+wheel zooms the x-axis; double-click resets. Everything redraws from
  tokens on theme change (reuse `redrawActivePlot`).
- Keyboard: when a plot canvas is focused (`tabindex=0`), `←/→` move the active marker by one
  bin.

**Tests.** `test_eye_height_and_width_are_measured_by_the_engine_not_the_page` (engine encoder
test + page reads the fields); `test_a_plot_readout_is_the_engine_s_sample_not_an_interpolation`
(the readout takes the nearest bin, never interpolates).

### 3.8 Standard curves: BER vs OSNR and theory overlay

- Sweep pane: when the swept axis is a noise/OSNR-controlling parameter (an EDFA noise figure, an
  OSNR-setting block, or received power) and the metric is BER, offer a `Theory` tick that draws
  the closed-form BER for the detected format (the engine already has these relations in the
  validation tests and reference-rate code; expose one function
  `maiman.analysis.theoretical_ber(format, osnr_db | q)` and a route `POST /api/theory`). Never
  compute the theory in the browser.

**Tests.** `test_the_theory_curve_comes_from_the_engine`.

### 3.9 Two-parameter sweep (surface) and goal seek

- Sweep form gains a second optional axis (`Axis 2: Block · Parameter · From · To · Steps`);
  `/api/sweep` accepts `axes: [ {...}, {...} ]` (keep the single `axis` key working); the engine's
  `sweep()` already takes multiple axes. The pane draws a heatmap (density ramp from DESIGN.md §6
  is **not** reused: a heatmap value is a measurement, so use a sequential ramp from `--ink-4` to
  `--optical`, documented in DESIGN.md) with the best point marked, and the results table shows
  the grid.
- `Goal seek…` (Simulate menu): pick a block parameter, a metric, and `maximise / minimise /
  reach value`; the server runs a bounded golden-section search (`maiman.sweep.goal_seek`, max 24
  evaluations, tolerance from the parameter's step) and returns the path of points; the pane
  plots the path over the sweep plot and the inspector gets the found value applied through the
  normal edit path (undoable). Tests in `tests/test_sweep.py` on a convex synthetic metric.

---

## Phase 4 — First five minutes, visual identity, teaching, and the desktop

**Goal.** A student opens the studio and builds a working link in five minutes without reading
docs; a block is recognisable at a glance; lessons can act on the canvas; the desktop build
behaves like a desktop application.

**Definition of done.** Tasks 4.1–4.10 merged; `DESIGN.md` "§12. Phase 4"; README screenshots
regenerated with `docs/capture_screenshots.py`; `desktop/README.md` updated.

### 4.1 Welcome panel

- On load with an empty canvas (a fresh `maiman serve` opens the flagship today — **change the
  default**: the page still carries the flagship as a template, but a first visit opens the
  Welcome panel over an empty canvas; a return visit in the same session restores the last
  document via 1.2's autosave). The Welcome panel is a `<dialog>` in the style of `openDialog`,
  800 px wide, three columns:
  - **Start**: `New empty project`, `Open a file…`, `Take the five-minute tour` (4.2).
  - **Recent** (from a `maiman-recent` list of up to 8 `{name, at}` kept in localStorage; with the
    File System Access API, the handles are stored in IndexedDB so they reopen in place).
  - **Examples**: the seven phases as rows, each row the examples as small cards with a
    64×36 px thumbnail of the schematic. Thumbnails are **generated at export time** by
    `examples/python/export_ui_data.py` as inline SVG strings of the block boxes and wires (no
    text, port colours only), embedded in `DATA.templates[key].thumb`; a test asserts every
    template has one and that none exceeds 2 kB.
- `Help → Welcome` reopens it; a `Show at start` checkbox at the bottom (default on).

### 4.2 Five-step interactive tour

- A guided tour engine in the page (no library): a sequence of steps, each `{ target: selector,
  text, done: () => boolean }`, drawn as a spotlight mask over the page with a card beside the
  target. Steps: (1) click `CW Laser` in the palette → done when a node of that type exists;
  (2) add `Optical Fiber`; (3) add `Optical Power Meter`; (4) wire laser → fiber → meter → done
  when both edges exist; (5) press Run → done when `SESSION.hasRun`; closing card: "The meter
  reads −16.00 dBm: 80 km at 0.2 dB/km costs 16 dB. Set the fibre to 200 km and run again."
  Escape exits; `Help → Take the tour` restarts; progress is not persisted.
- The tour must work with the keyboard (focus moves to the target).

**Tests.** `test_the_tour_ends_on_a_number_the_engine_produces` (the closing text's number is
checked against `run_project` of the three-block graph).

### 4.3 Category glyphs on block cards

- Each manifest category gets one 16×16 monochrome glyph (stroke icons in the existing toolbar
  style), drawn at the top-left of the node card before the title, in `currentColor` of the
  category's port hue (`CATEGORY_PORT`). Define them once in a `GLYPHS` object of SVG path
  strings; the palette rows use the same glyphs. Categories at 0.17.0: Optical Sources, Electrical
  Sources, Modulation, Modulators, Fiber, Amplifiers, Passive, Photonic IC, Receivers, DSP,
  Measurements, Electrical, Signal Flow, Free Space. Standard schematic cues: laser = diode with
  two arrows out, fiber = coil, EDFA = triangle, PIN = diode with two arrows in, filter = sine
  in box, DSP = `Σ`, measurements = meter arc.
- Test `test_every_category_has_a_glyph` (extend
  `test_every_category_has_a_place_in_the_palette_and_a_colour`).

### 4.4 Level of detail on zoom, and long names

- Below 70 % zoom, `drawGraph` omits the caption line and draws glyph + title only; below 45 %,
  title only, in a larger relative font so it stays legible (measure at 1440 px and record the
  thresholds in DESIGN.md). Hit areas do not change.
- Node titles wrap to two lines when the name exceeds the card width at 100 % (`NODE_H` grows to
  64 for those cards only; `portPos` must use the per-node height). The `truncate` tooltip rule
  stays for anything longer than two lines.

### 4.5 Wires: hover emphasis, orthogonal routing option, bridges

- Hovering a wire or a block highlights the wire(s) (stroke 2.5 px) and dims every other wire to
  40 % opacity; selecting does the same persistently. Hovering a port shows its name and type
  (2.6).
- `View → Wire routing → Curved | Orthogonal` (default Curved, persisted). Orthogonal routes each
  wire as horizontal → vertical → horizontal with the bend placed at the midpoint, rounded 6 px
  corners; backward wires loop below both blocks (keep the §9 "backward wire" argument).
- Where two wires cross in orthogonal mode, the later-drawn one gets a 6 px half-circle bridge.
  Curved mode gets no bridges (state why: crossings in curves are oblique and legible).

### 4.6 Auto-layout

- `Edit → Arrange by signal flow` (`mod+shift+L`): a layered layout — rank each node by longest
  path from a source (nodes with no inputs), place ranks left-to-right at the existing 138 px
  pitch, order within a rank by the barycentre of predecessors, rows at the existing 190 px pitch,
  then collapse rows to the minimum. Annotation frames are moved with the bounding box of the
  nodes they enclose. One undo snapshot. Then `fitContent()`.
- Test `test_arrange_places_every_block_and_keeps_every_wire` on the flagship document (a pure
  JS function `arrangeLayout(nodes, edges)` evaluated in Node, or the algorithm mirrored in Python
  for the test — pick the precedent).

### 4.7 Lessons that act

- Extend the `[[label]]` syntax in `renderRich`: `[[label.param=value]]` renders as a button-like
  link that **sets** the parameter through the normal edit path and reveals the block;
  `[[run]]` runs; `[[sweep:label.param=from..to]]` fills and runs a sweep. The lesson editor's
  help line documents the three forms. Convert at least the "Try this" sections of phases 1 and 2
  in `examples/python/lessons.py` to use them (regenerate projects).
- Test `test_every_lesson_action_names_a_block_and_a_parameter_that_exist` over all templates.

### 4.8 What's new, documentation links, i18n scaffold

- On first load of a new engine version (compare `/api/health` version to
  `localStorage["maiman-seen-version"]`), show a small "What's new in 0.18" dialog with the
  bullets taken from `docs/releases.md` for that version: `export_ui_data.py` extracts the
  section for the current version into `DATA.whatsNew` (test: it is the section for
  `maiman.__version__`).
- All user-facing strings in the page move into one `STRINGS` object (keys in English, values
  English for now); `t(key, vars)` helper; Persian/RTL readiness is confirmed by a test that no
  template literal in markup-producing code contains a hard-coded English sentence outside
  `STRINGS` (allow-list the log lines and the engine's messages, which are facts). Do **not**
  translate yet; this task only makes it possible.

### 4.9 Subcircuits (hierarchical blocks) — scoped minimum

- `Edit → Group into subcircuit` on a selection of ≥ 2 blocks creates a node of type
  `Subcircuit` in the document: `{ id, type: "Subcircuit", subgraph: { nodes, edges, layout },
  ports: { inputs: {...}, outputs: {...} } }` where the exposed ports are every port of an inner
  block that was wired to something outside the selection (or unwired required inputs), named
  `<inner label>.<port>`. **Engine first**: `maiman.project.graph_from_dict` flattens subcircuits
  on load (prefixes inner labels with the subcircuit id), so the engine runs the flat graph and
  needs no new component; results come back keyed by the flattened labels and the page maps
  them back. Double-click enters the subcircuit (breadcrumb at the top of the canvas,
  `Esc`/breadcrumb leaves); `Ungroup` is the inverse. Copy/paste (1.5) carries subcircuits.
- Tests in `tests/test_project.py`: `test_a_subcircuit_runs_as_its_flat_graph` (same numbers
  grouped vs ungrouped on `ook_link`), `test_a_subcircuit_round_trips_through_save_and_load`.

### 4.10 Desktop build

- `desktop/main.js`: native `Save`/`Open` dialogs via IPC (the page calls
  `window.maimanDesktop.save(doc)` when that bridge exists, before trying the web picker; expose
  through a `preload.js` with `contextIsolation: true`), a native **Recent files** submenu on all
  platforms (the page's Welcome panel reads the same list via the bridge), `.maiman` file
  association in `electron-builder` config (open-with and double-click), window title following
  the dirty state (1.1), and `Cmd/Ctrl+Q` asking about unsaved changes through the same confirm.
- `desktop/smoke.js` extended to open a `.maiman` from the command line and assert the window
  title. `desktop/README.md` updated.

---

## Appendix A — Priority if time runs short

Do, in this order, and stop anywhere: 1.1 → 1.2 → 1.4 → 1.12 → 1.6 → 1.5 → 2.1 → 2.2 → 2.4 →
2.7 → 3.1 → 3.3 → 3.4 → 3.5 → 1.8 → 1.11 → 3.6 → 3.7 → 4.1 → 4.2 → 4.3 → the rest.

## Appendix B — What is deliberately not in this plan

- Touch and mobile layouts (desktop tool; PRODUCT.md).
- A rewrite onto React Flow or any framework (PRODUCT.md constraint; revisit only if the canvas
  outgrows SVG, which 20-block links do not).
- Real-time collaboration or cloud storage (opening a file never reaches the server; keep it so).
- New physics or components beyond declarative metadata and the small engine helpers named in
  3.5, 3.7, 3.8, 3.9 and 4.9, each with its closed-form or round-trip test.
