---
title: "The studio"
description: "The schematic editor: build a link by dragging blocks and wires, run it, sweep it, save it."
---

# The studio

A working schematic editor, served by the engine it drives: build a link by dragging blocks and
wires, press Run, sweep a parameter, save the project. Every number on the screen comes from the
engine.

```
maiman serve
```

then open `http://127.0.0.1:8765/`. The interface is a file inside the package, so installing the
engine installs it: there is nothing to build and no checkout to be standing in. From a checkout,
`python -m maiman.server` is the same command by another name.

![The Maiman Studio schematic editor on its graphite ground](images/studio-graphite.png)

## The canvas draws the document

The blocks, the wires and the parameter values are not written into the page. They are read from
the same `.maiman` document the server accepts, and Run posts that object straight back. So the
picture on the canvas, the values in the inspector and the graph the engine executes are one
description instead of three kept in agreement by hand.

## Editing

- **Add a block** by dragging it from the palette, or clicking it there.
- **Wire it** by dragging from an output port to an input port.
- **Move** by dragging; select several and align them from the toolbar or the menu.
- **Remove** a selected block or wire with `Delete`, and take it back with `Ctrl`+`Z`.
- **Edit parameters** in the inspector. A field that has no effect under the current settings is
  greyed out rather than silently ignored.

A connection is refused *while it is being dragged*, not when Run is pressed, and the refusal says
why: `binary cannot drive optical`, `fib.in already has a source`, `a block cannot feed itself`.
An input with nothing feeding it is drawn hollow, because the engine will refuse to run the graph
and showing which port is the problem beats reporting it afterwards.

Every wire colour is a signal type, not a preference (optical C-band cyan, electrical amber,
binary slate, symbol violet, metric magenta), so a glance at a link says what travels down it. A
typed-port system that refuses invalid wiring is worth nothing if the types are invisible.

## Lessons and notes

Every project the Examples menu opens comes with a lesson in a panel on the left of the canvas:
what the link is for, the formulas behind each stage, and what to try next. A block named in the
lesson is a link: click it and the block is selected on the canvas. The toolbar's note, frame and
arrow tools put your own explanations on the canvas, and **Notes → Edit** writes a lesson of your
own; both are saved in the `.maiman` file and never change what the engine runs. See
[Project files](projects.md#the-format).

## Runs

Press Run and the page posts its graph to the engine and draws what comes back: the constellation,
the measurements, the block captions, and a log in which every line is a fact from the response.
Change a parameter and run again, and the numbers move because the physics moved: set the fibre to
200 km and the received power drops by exactly the 24 dB the extra loss costs.

**A run says how far along it is.** Not a spinner: the response streams, and the bar is drawn from
the fraction the engine has reported, the components behind it plus the fraction within the one
running. In a link with a long span one block is the entire run, so the split step reports after
every step and a 16-second span moves the bar 268 times. It is deliberately not a time estimate; a
run has no honest one.

**Stop aborts the response, not the run.** Block-mode execution has no safe place to stop halfway,
so the engine finishes and its answer is discarded, and the log says so in those words.

**Numbers say where they came from.** A badge in the results dock reads *live* when they came from
this session's last run, *stale — graph edited* when the graph has changed since, and *reference*
when they came from the run baked into the file. The page also opens straight off disk with nothing
running, which is how it should be read if you only want to look.

## Sweeps

Pick a block and a parameter, give it a range, and the curve appears beside the form that made it.
Repeats draw the spread at each point, because one BER estimate at a marginal operating point is a
sample and not an answer. The plot opens on the metric that moved most, which is usually the one
the sweep was run to watch, and which stops it opening on an analyser downstream of the decoder
whose EVM is zero at every point.

The endpoint sends back numbers, not pictures: seven points of a coherent link is 6 kB. The server
runs four points at a time; see [Sweeps](sweeps.md) for why the answer does not depend on that.

## The spectrum dock

An Optical Spectrum Analyzer anywhere on the graph puts its trace in the dock: wavelength across,
dBm per resolution bandwidth up, the peak marked where it was found. The axis is labelled in the
analyser's own resolution because the trace *is*: widen the setting and the ASE floor lifts decibel
for decibel while the channels stay where they were. That asymmetry is why an OSNR figure is
meaningless without its bandwidth, and it is one parameter and one Run away from being seen.

The OSNR beside the trace comes from an OSNR meter on the graph, and is left empty when there is
none: a figure read off the displayed curve would move with the resolution knob.
`examples/maiman/wdm_osa.maiman` opens four channels on the 100 GHz grid through two amplified
spans.

## A block's own spectrum

Select a grating, a ring, a coupler or any other photonic block and its properties carry *Show its
S-matrix spectrum*: the S-matrix tab draws every entry of its scattering matrix across a window, as
power, phase or group delay, without a link run through it. The window starts where the device is
worth looking at (a grating's line, four of a ring's free spectral ranges) and can be moved
anywhere. The engine does the work: `POST /api/spectrum` builds the block from the same node a run
would and calls `spectrum()` on it, which a script can call just the same.

Group delay is a local derivative, `S(f + δ)` against `S(f)`, not a differenced unwrap of the
sampled phase, which turned a straight waveguide's 14 ps into −1 ps as soon as the grid was coarser
than its delay. Where an entry is 40 dB below its own peak its phase means nothing, and the plot
leaves a gap rather than drawing the spike.

## Open, save and examples

**File → Save** writes a `.maiman` file: the same document the canvas draws and the server runs,
with the block positions folded in. **File → Open** reads one back. Both go through the browser:
the file is chosen in your operating system's own picker and read locally, and nothing is posted.
A file that is not a project says so and leaves the canvas alone.

The Examples menu opens complete projects, one submenu per phase of
[the examples roadmap](EXAMPLES_ROADMAP.md): first light, direct-detection links, amplified and
WDM links, and coherent links with the full DSP chain and soft-decision FEC. Each is a `.maiman`
file in `examples/maiman/`, written by a script in `examples/python/`.

## Paper and graphite

The editor ships two grounds and defaults to paper, because a schematic is a document before it is
a screen and its plots leave the tool for reports and papers. Graphite is one click away.

![The same editor on its paper ground](images/studio-paper.png)

## How it is built

The studio is one HTML file with no build step: an SVG canvas, no React Flow and no bundler, so it
still opens straight off disk with nothing installed. It talks to the
[session server](server.md), which is an ordinary client of the [Python API](api.md): nothing in
the interface knows any physics, and nothing in the engine knows the interface exists. See
[DESIGN.md](../DESIGN.md) for why it looks the way it does.
