# Design

The visual and interaction record for Maiman Studio's interface. It exists because
[`src/maiman/studio/index.html`](src/maiman/studio/index.html) is a *build*, and a build without a
written rationale is a set of numbers nobody can argue with later. Everything
here was decided against the audience and use scene in [PRODUCT.md](PRODUCT.md);
where a decision was made by measurement rather than judgement, the measurement
is given.

---

## 1. What the surface is

An **Operate** surface: a schematic editor, used to complete a task rather than
to be admired or read through. Scanability, consistency and the real usage scene
outrank expression. Brand lives in precise details, not in gestures.

The engineer is *iterating* — change a parameter, run, look at the constellation,
change it again — dozens of times an hour. Three consequences follow, and they
decide most of what is below:

1. **Density beats generosity.** Screen space spent on decoration is space not
   spent on the schematic or the plot.
2. **Recognition beats novelty.** The layout is the one this audience already
   knows from Blender, VS Code and EDA tools. Nothing is moved to be different.
3. **Nothing on screen may be invented.** Every number, every parameter, every
   port comes from a real engine run. See §7.

---

## 2. Two grounds

The interface ships **paper** and **graphite**, and defaults to paper.

A schematic is a document before it is a screen. Its plots leave the tool for
reports, theses and papers, where a screenshot off a black canvas is the wrong
artefact. Graphite is one click away for the night session beside a terminal.

Neither is a skin over the other:

| | paper | graphite |
| :--- | :--- | :--- |
| ground | `#e9edf3` bed under white panels | `#0b0e13` under `#11151c` panels |
| a block | white card on a grey bed | raised panel on a dark one |
| hover | adds **ink** (darker) | adds **light** |
| plot bed | `#ffffff` — a plot is on paper | `#080b10` |

The port hues are the identity and survive both, but their **values are
re-derived per ground rather than inverted** — on paper the spectrum darkens to
hold contrast against a light bed.

**Structure.** The bare `:root` block carries the complete paper palette. The
graphite tokens appear twice: once under
`@media (prefers-color-scheme: dark)` guarded as `:root:not([data-theme="light"])`,
and once under `:root[data-theme="dark"]` so an explicit choice wins in both
directions. No colour is ever defined *only* inside a media or `[data-theme]`
block — that is the classic unreadable-page bug, and it is checked rather than
assumed (§8).

---

## 3. The spectral port identity

**Every port colour is a wavelength, not a preference.** A wire's colour tells
you what travels down it.

| port type | paper | graphite |
| :--- | :--- | :--- |
| optical (C-band) | `#0a8279` | `#22d3c5` |
| electrical | `#a35f00` | `#f0a030` |
| binary | `#5a6474` | `#7c86a0` |
| symbol | `#6539cc` | `#9b7cf6` |
| metric | `#bc2668` | `#e85d9b` |

This is the one place the interface is allowed to be memorable, and it earns its
place by being *information*: a typed-port system that refuses invalid wiring at
edit time is meaningless if the types are invisible.

The shown project is chosen so that **all five types appear on the canvas at
once**. That is why it is a coherent link rather than an on-off-keyed one: only
a coherent chain carries binary into symbols, symbols into two electrical
drives, an optical field, two photocurrents back, and symbols out again. It runs
over 80 km of real fibre, because a simulator whose demonstration link has no
fibre in it is showing the wrong thing.

Accent is `--optical`, because the tool is an optical simulator. Semantic colour
(`--good` / `--warn` / `--bad`) is kept separate from it and does not count as
the accent.

---

## 4. The ink ladder

Four layers, spaced so hierarchy comes from the **distance between them** rather
than from pushing the weakest below readability.

| token | paper | vs white | graphite | role |
| :--- | :--- | ---: | :--- | :--- |
| `--ink` | `#0f141c` | 18.5 | `#e6ecf5` | values, headings |
| `--ink-2` | `#333d4b` | 11.0 | `#a8b4c6` | labels, body |
| `--ink-3` | `#4c5666` | 7.4 | `#8e9bb0` | section titles, axes |
| `--ink-4` | `#5f6a79` | 5.5 | `#7a869b` | units, ranges, disabled |

The bottom rung is set by the **tinted grounds it actually lands on** — a unit
inside an input well, a disabled toolbar label — not by white, where it would
look too pale. Against the darkest surface it still clears 4.5:1.

---

## 5. Layout

Four regions, top to bottom:

```
menu bar            34px   File / Edit / Simulate / View / Help, ground switch, engine state
action toolbar      40px   Run, Stop, Sweep · select/pan/align · zoom · run context
body                1fr    palette 216px | node canvas 1fr | inspector 260px
results dock        292px  tabs + metric strip, plot + side readouts
status bar          24px   block and link counts, selection, sample count, precision
```

**Run sits leftmost in the toolbar**, where the eye lands first, because it is
the action taken dozens of times an hour.

**The dock is 292px** because the constellation's plot is square and therefore
sized by the dock's *height*, not its width. A shorter dock wastes the width it
has.

**The canvas viewBox is 1000 x 530**, seven columns of 138px at a node width of
116, three rows at y = 30 / 220 / 410. The seventh column arrived with the span
and its compensator; the alternative was to keep six and squeeze the spacing to
120, which leaves four pixels between adjacent blocks and reads as cramped.

The *height* is set by the region the SVG actually lands in, and that number is
worth measuring rather than estimating. At 1440px the canvas region is
**964 x 510 — an aspect of 1.89**. With `preserveAspectRatio="xMidYMid meet"`
any box wider than that is letterboxed top and bottom: the original 880 x 420
(2.10) wasted 50px, and widening to 1000 x 420 (2.38) wasted **105px** — it made
the fit worse, not better, which is the opposite of what a first pass here
claimed. Matching the box to 1.89 wastes none, and the rows spread into the
height they gained.

Below 940px the palette and inspector collapse and a note says so. This is a
desktop tool; the real build would dock them as overlays rather than dropping
them, and the note says that too rather than pretending the narrow layout is the
intended one.

---

## 6. Plots

Four plots, in two families, and which family a result belongs to is decided by
what one mark on it means rather than by what block produced it.

**Histograms.** Two of them, sharing one visual language because they mean the
same thing: **distance from the page is how many landed there**.

- **Eye diagram** — density ramp bed → cyan → far stop.
- **Constellation** — the same ramp; axes through the origin, because a
  constellation is read relative to zero and not to the corner of a box; ideal
  points marked with **crosses rather than filled dots**, so a cluster's own
  centre stays visible underneath and a bias offset reads instead of being
  covered by the marker meant to locate it.

**Curves.** Two of them, and they deliberately do *not* borrow the ramp. A point
on a curve is one measurement, not a count, and shading under it would promise a
density that is not there.

- **Sensitivity** — one curve per format, each labelled where it crosses the FEC
  threshold rather than in a legend box. The threshold is 1e-3, not Q = 6:
  nobody operates an error-free channel any more, they operate one a
  soft-decision code can close.
- **Spectrum** — wavelength across, ascending left to right because that is how
  an OSA is read; the engine's grid ascends in *frequency*, so the axis maps the
  array backwards rather than the trace being reversed. Up is dBm **per
  resolution bandwidth**, labelled in the analyser's own setting because the
  trace is measured in it: widen the resolution and the ASE floor climbs decibel
  for decibel while the channels, already narrower than either setting, do not
  move. A bare-dBm label would hide the one asymmetry the instrument exists to
  show. The vertical range is 80 dB below a decade above the peak, **fixed
  rather than fitted** to the trace's own minimum — the floor of an unamplified
  link is the encoder's floor and not a measurement, and letting it set the
  scale would plot two runs of the same link at different heights.

The ramp's mid stop is placed by **luminance**, at roughly the same fraction of
each ground's span, which is what keeps sparse outliers subordinate to the dense
regions on both.

One thing the spectrum pane does *not* draw is an OSNR of its own. The figure
beside the trace comes from an OSNR meter on the graph and is blank when there
is none: read off the displayed curve it would move with the resolution knob,
and it would not be the number anyone means by OSNR.

A canvas has no cascade, so the plots read the same tokens as everything else at
draw time and are redrawn when the ground changes. Nothing below the token block
names a colour.

---

## 7. Real data, always

The palette is generated from `manifests()` — the same call the real GUI will
make. The inspector shows true parameters, units, ranges and docstrings. The
constellation, eye and sweep are a real run, exported by
[`examples/export_ui_data.py`](examples/export_ui_data.py).

This is not a purity exercise. Building the mockup against real data has twice
found engine defects that the test suite missed:

- `eye_histogram` accepted any `time_bins` and rendered a 32-sample trace across
  96 columns as **vertical banding**. Time resolution is now capped at one
  column per sample.
- Placing an analyser after the differential decoder reported **EVM of exactly
  zero** however bad the link was, because that block emits decisions. The
  mockup now carries two analysers — soft measurement before the decoder, error
  count after — which is also how a bench does it.
- The sensitivity sweep's 256-QAM curve **never crossed the FEC threshold** in
  the swept range, so it was the one curve with no label — quietly contradicting
  the rule in section 6 that each is labelled where it crosses. The range now
  extends far enough that all four cross.

---

## 8. What is verified, and how

Not by looking at downscaled screenshots. Each publish is checked by
measurement, in **both grounds**:

- **Contrast** — every text node's computed colour against its resolved
  background; ~110 elements per dock tab. Target 4.5:1 (3:1 for large text).
  Current status: **zero failures**, both grounds, all four tabs.
- **Token completeness** — every `var(--…)` the stylesheet references must
  resolve from bare `:root` alone. Current: 34/34.
- **Theme-block drift** — the two graphite blocks are compared key by key.
- **Canvas repaint** — plot beds are sampled after switching, in both
  directions.
- **Geometry** — no SVG element may stray outside the viewBox; the viewBox
  aspect must match the canvas region's, or the difference is letterboxed away;
  the Ports legend must not overlap a block; no horizontal page overflow.
- **Screenshots** — `python docs/capture_screenshots.py` regenerates
  [`docs/images/`](docs/images/) in both grounds at the same 1440px the audits
  use, so what the README shows is what the current build renders rather than an
  older one. It refuses to run if the mockup's theme bootstrap has changed,
  because the failure mode otherwise is two paper captures, one of them labelled
  graphite.

Two findings worth keeping:

- A contrast audit run while the browser pane was not compositing reported a
  clean result that **meant nothing** — the page had laid out at its narrow
  breakpoint, where the palette and inspector are `display: none` and were never
  checked. It happened a second time on a freshly opened tab, which does not
  inherit the previous tab's emulated size. The audit now asserts
  `innerWidth === 1440` and throws otherwise, and reports how many nodes it
  inspected so a zero is readable as a pass rather than as silence.
- The design detector once ran degraded and returned `[]`. An empty result from
  a degraded tool is an undercount, not a pass.

---

## 8a. The menubar

For most of this project's life the bar read **File · Edit · Simulate · View ·
Help** and only the first of them opened anything. The other four were labels
over nothing — which is precisely the failure [PRODUCT.md](PRODUCT.md) names as
the thing a polished result must never be, sitting in the most-read strip of the
window. A user who had installed it went looking for *Save as* and found the gap.

All five open now, and **every row runs something**. That is held by a test
rather than by care: each `role="menuitem"` must be named by an `onMenu(...)`
call, so a row added without a handler fails CI naming itself.

### What is in them, and why those

| | |
| :--- | :--- |
| **File** | New, Open, Save, Save as |
| **Edit** | Undo, Redo, Duplicate block, Delete |
| **Simulate** | Run, Stop, Parameter sweep |
| **View** | three zooms, the five result panes, the two grounds |
| **Help** | Keyboard shortcuts, Documentation, Report an issue, About |

Two of those are new capability rather than a new way to reach an old one.
**Redo** did not exist — `undo()` popped a stack and the state came off it went
nowhere, so an undo was a one-way door. There is a `FUTURE` stack beside
`HISTORY` now, cleared by the next real edit, which is what every editor does
and the only consistent answer once the timeline branches. **Duplicate** is new
too, and deliberately does *not* copy the block's wires: which of them a copy
was meant to inherit is a guess, and a duplicate that silently joins the signal
path is worse than one that waits to be wired.

### What is deliberately not in them

**Cut, copy and paste.** They would need a clipboard model — what a copied
subgraph is, what happens to wires that leave the selection, whether a paste
into another project is allowed — and none of that exists. Duplicate covers the
case those three are usually reached for. Three rows that looked familiar and
did nothing would have recreated exactly the problem this section is about.

**Select all.** There is no multiple selection to have, so it would be a row
that succeeded at nothing.

**Zoom to fit.** The canvas has no measured content extent to fit *to* — the
number the ceiling review in the last section established is a
fixed grid, not a bounding box of what is drawn. It can be added when there is
something real behind it.

### How it behaves

Like a desktop menubar, because §1 says recognition beats novelty and this
audience has one in their hands already: one menu open at a time, hovering the
bar while one is open moves to its neighbour without a second click, arrows walk
the rows, left and right move between menus, Escape closes and returns focus to
the button it came from.

**Rows that do not apply are disabled, not hidden.** A menu whose contents move
around is a menu you have to read every time, and an unavailable row still tells
you the action exists. Undo and Redo are greyed when their stacks are empty,
Stop when nothing is running, Zoom to 100% when the zoom is already there. The
state is computed *as the menu opens* — there is no second copy of "can this be
undone" kept in sync and able to fall out of step.

### The two things worth taking from the build

**One shortcut table.** `SHORTCUTS` is read by both the keydown handler and by
Help → Keyboard shortcuts, so the list a user reads is the list the page obeys.
A hand-written shortcut panel is a second copy, and a second copy eventually
promises a key that was renamed. `mod` renders as ctrl or cmd by platform and
matches either, because a Mac user pressing cmd+S and a Windows user pressing
ctrl+S are doing the same thing.

**About reports the engine, not the page.** The version comes from
`/api/health`, which grew a `version` field for it. The page is a file and the
engine is what is running; they are the same in a release and differ the moment
somebody runs a checkout of the studio against an installed engine, which is
exactly when knowing would matter. With no server behind the page it says so
rather than inventing a number.

One bug the build surfaced and fixed: the undo snapshot never carried the
**project name**. Nothing had changed it before — New does, so undoing a New
restored twenty-two blocks under the title `untitled.maiman`, which is a true
canvas under a false label. The name is part of the state now.

## 8b. Four things a real user found

The menubar shipped and somebody installed it and used it. All four of these came
back from that, and none of them would have come from reading the code.

**The status text slid under the control beside it.** Nothing in
`.menubar-right` was `flex: none`, so when the bar got tight every child shrank
equally, the text overflowed a box narrower than itself and the first letter of
"engine ready" went behind the theme button. Measured: the two halves of that bar
collide below **769px**. Exactly one thing is allowed to give now — the project
name, because a project can be called anything and an ellipsis says it was
shortened where an overlap says the page is broken.

**A run on a new link showed the old link's plots.** This is the serious one. The
constellation pane read `SESSION.plots.constellation || DATA.constellation` — the
reference histogram baked into the page. Before a run that is right and the dock
badge labels it "reference"; *after* one it is a result claimed for a link that
never produced it. A laser and a power meter ran and drew a 16-QAM cloud. The eye
had guarded against this since it was written and the spectrum too; the
constellation and the sensitivity plot had not. All four now share one
`drawNothing` empty state, and `File → New` repaints — clearing the data a canvas
reads does not clear the canvas.

**The canvas could not be moved.** A schematic larger than the window had parts
you could not reach. Dragging the bed now pans it and the grid carries
`cursor: grab` so that is visible before it is tried; middle-drag pans from
anywhere including over a block, which is the gesture this audience has from
Blender and every EDA tool. Pan and zoom are written into **one** transform, so
they cannot disagree, and `svgPoint` needed no change at all because
`getScreenCTM()` already accounted for whatever the element's transform was.
Panning leaves the selection alone; a press on the bed that never moved is still
a click and still clears it.

**About set its facts at body size**, so `maiman 0.2.0` read as a heading it is
not. They are a readout: 11px, monospaced on the value side, the size the status
bar uses for the same kind of thing.

What did *not* change: the Log keeps its lines across a `New`. It is a session
journal, every line is timestamped and the load writes one saying what happened,
so the history is legible rather than stale. That is a judgement rather than an
oversight, and it is written here so it can be argued with.

## 9. Decisions worth not re-litigating

- **Paper is the default**, and it is stamped before first paint so a dark host
  never flashes through. The stylesheet still answers `prefers-color-scheme` on
  its own for the no-script case.
- **A backward wire is the schematic wrapping**, not a mistake. It needs a
  *tighter* bezier control offset than a forward one; scaling the offset with
  the span throws the curve outside the canvas on exactly the wire that already
  travels furthest.
- **Node labels truncate with the full name in a tooltip.** A fixed-width box
  that lets long names spill across neighbours is worse than one that clips.
- **Hover on paper adds ink, not light.** Lightening a hover on a white ground
  moves it towards invisible.

---

## 10. Not done

- **The block-count ceiling was wrong, and this is what replaced it.** This
  section used to say that "at ~20 blocks the node text stops being readable at
  this canvas size". That claim conflated two things and neither of them is a
  block count.

  Node text does not depend on how many blocks there are. It depends on the
  zoom, and the zoom is a control. What *can* force zooming out is the layout's
  **extent** against the viewport — and extent is set by the grid, not by
  occupancy. The canvas is a 138 x 190 pitch, seven columns by four rows, which
  is **28 slots**. The link went from eighteen blocks to twenty and the extent
  did not move at all: x from 30 to 858, y from 30 to 600, both times, because
  the new blocks went into slots that were already empty.

  So the honest constraint is: *keep the layout inside one screen at 100 %, and
  keep the wires followable.* Twenty blocks in twenty-eight slots does both.
  Thirty would not, and the number that matters then is rows, not blocks.

  What the old rule got right is that a schematic nobody can read is not a
  better demonstration. What it got wrong is where the limit comes from — and it
  cost a real capability, because soft-decision FEC was kept off this canvas
  partly on the strength of a number that was never measured.

- **Three links still get canvases of their own**, all openable from the File
  menu, and now for one reason rather than two: they cannot be *this* link. The
  eye comes from [`examples/ook_eye.maiman`](examples/ook_eye.maiman) — a
  coherent receiver has no eye. The spectrum comes from
  [`examples/wdm_osa.maiman`](examples/wdm_osa.maiman) — a single-carrier link
  has nothing for an OSA to show. Tests keep both off this canvas rather than
  trusting that nobody wires one on.

  [`examples/coherent_sdfec.maiman`](examples/coherent_sdfec.maiman) is the
  third, and it stays because it isolates the code from the carrier: it declares
  an ideal laser and no fiber, so what it measures is the decoder. The flagship
  now carries soft-decision FEC too, and measures something different — see
  below.

- **What carrying a block code costs this canvas, which is not nothing.** A
  staircase block is 16384 coded bits, so at four bits per symbol the window is
  quantised to 4096 symbols. The shipped project can no longer be run at 256 for
  a quick look; it refuses, and names a length that works. That is a real
  reduction in what the sequence-length control can be set to on the link people
  open first, and it is the honest price of the capability rather than an
  oversight. It is also not an implementation wart: a block code quantises a
  frame in hardware too, and nobody runs an OTN link at 256 symbols either.

- **The post-FEC number on this canvas is dominated by the window's edges, and
  says so.** The link filters circularly — the pulse shaping and the dispersion
  compensator both wrap — so the first and last few symbols carry a fold from
  the far end of the sequence. The analyser is allowed to discard them and does,
  through `ignore_edges`. **A decoder is not**: it has to decode every bit it is
  given. Measured on the shipped graph, all seven pre-FEC bit errors sat in the
  first 0.05 % and the last 0.07 % of the window, and the interior was clean.

  That is an artefact of simulating a finite window, not a property of the link —
  a real stream has no edges. It is left visible rather than hidden behind a
  larger window, because a post-FEC rate that quietly measures the simulation
  boundary is exactly the kind of number this project exists not to print.

- No motion beyond the run pulse and the control transitions.
- Progress on a long run is **done**, and done the way this section said it
  would have to be: a real fraction from the engine rather than an animation
  that only signals nothing has crashed. The run streams, the bar in the menubar
  is drawn from components-behind-this-point plus the fraction *within* the one
  running, and the block whose name is on screen is the block currently working.
  A 16-second span reports 268 times on the way and lands on exactly 1.0. What
  it is not is a *time* estimate: a run has no estimate before it starts, and
  the honest quantity is how much of the work is behind it.
- Stop aborts the response, not the run. Block-mode execution has no safe place
  to stop halfway — killing a component mid-array leaves the server holding a
  torn result — so the engine finishes and its answer is discarded. The log says
  that in those words rather than implying a cancellation that did not happen,
  and the engine dot stays green, because stopping is not failing.
- The running, failed and stale states are now real, because the page calls the
  engine. A run in flight dims the canvas and swaps Run for Stop; a failure
  turns the engine dot red, names the kind of failure from the status the
  server chose (400 unreadable, 422 readable but unrunnable, 413 too large) and
  opens the log with the engine's own message in it; a graph edited since the
  last run marks the results dock stale rather than leaving numbers on screen
  that answer a question no longer being asked.
- The empty state exists now that the canvas can be emptied. It names the next
  action rather than describing the situation: someone looking at a blank canvas
  has already worked out that it is blank, and what they need is where to start
  and that ctrl+Z brings back what they just deleted.
- A refused connection is answered on the canvas, in one line at the bottom,
  rather than in the log. The log is for what a run did; this answers something
  the hand is still doing, and by the time you have looked at a panel to find it
  the gesture is over.
