<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-wordmark.png">
    <img src="assets/logo-wordmark-dark.png" alt="Maiman Studio" width="420">
  </picture>
</h1>

**An open-source, modular simulator for optical communication links and photonic systems.**

*Named for **Theodore Maiman**, who fired the first working laser in May 1960 — the event every
link in this simulator descends from.*

[![CI](https://github.com/ehsun-sh/maiman-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/ehsun-sh/maiman-studio/actions/workflows/ci.yml)
![status](https://img.shields.io/badge/status-pre--alpha-orange)
![license](https://img.shields.io/badge/license-Apache--2.0-blue)
![python](https://img.shields.io/badge/python-3.11%2B-blue)

---

> ### Project status: 0.4.1 — released, and still moving.
>
> `pip install maiman`, then `maiman serve` — or [start from no Python at
> all](#installing-and-running-it). Phases 0 through 4 are done: **56 components, more than 1250 tests, and
> every physics block checked against a closed-form result in CI.**
>
> **Links run end to end.** Direct detection — PRBS → NRZ → laser → MZM → fiber → PIN → filter →
> eye/Q/BER. Coherent — Gray-coded M-QAM at every whole number of bits per symbol to 256, RRC
> shaping, IQ modulator, 90° hybrid with balanced detection, and a receiver that recovers the
> sampling instant, the carrier frequency and the carrier phase *blind*, over 1000 km of fiber
> that leaves nothing recoverable at the photodiode. Dual polarization at 256 Gb/s with a blind
> butterfly equaliser.
>
> **Channels interact.** The split-step propagates them coupled, so a neighbour's power modulates
> each channel's phase at twice the rate its own does, sliding past under walk-off derived from
> the dispersion, and triplets mix to put light where nobody launched it. EDFAs emit ASE into the
> noise-bin model, saturate on total power so WDM channels share one inversion, and give an OSNR
> **and a Q-factor that follows from it**.
>
> **Errors get corrected.** RS(255,239) per ITU-T G.709, and a soft-decision braided BCH staircase
> decoded on log-likelihood ratios — which clears a line at 7.4e-3 that the hard code returns
> untouched.
>
> **Photonic circuits solve** as bidirectional S-matrices, cross-validated against SAX to 7e-15,
> with PDK import that reads a foundry's fitted numbers and refuses to extrapolate past the window
> they were fitted in. Each guided polarization carries its own indices through the same
> reduction, so a ring resonates at **two** sets of wavelengths on two free spectral ranges. And a
> netlist a layout tool wrote is read as a document and solved against a kit — gdsfactory's own
> shipped sample included, unmodified, without importing it.
>
> **The receiver acquires, and the amplifier has a clock.** Coarse carrier acquisition covers the
> whole sampled band where the M-th power estimator folds at ±4 GHz and returns a wrong offset at a
> right-looking confidence. Erbium gain dynamics are solved on their own time axis, because the
> transient is twenty-five thousand simulation windows long — a channel drop swings a survivor
> +3.21 dB through one amplifier and +9.94 through a chain of eight.
>
> **The interface is a working application**: `maiman serve`, build a link by dragging blocks and
> wires, press Run, sweep a parameter, save the project. Every number on screen comes from the
> engine.
>
> **0.x means the API is not stable yet.** That is the honest reading of the version and not a
> formality — the core is still small enough that changing it is cheap, which makes now the most
> useful time to argue with it. **[The architecture document](docs/ARCHITECTURE.md)** is where the
> expensive decisions are set out, and criticism of them is still worth more than any feature.
> What is *absent* is listed as plainly as what works: see the [roadmap](#roadmap) and the
> per-section notes on what each model does not do.

---

## Installing and running it

**This section assumes you have never used Python or a terminal.** If you have, the whole thing is
`pip install maiman` then `maiman serve` — skip to [The interface](#the-interface).

Everything below was run on a clean machine and the outputs are what actually appears.

### 1. Install Python

Maiman needs **Python 3.11 or newer**. It is free, and installing it changes nothing else on your
computer.

**Windows** — download the installer from [python.org/downloads](https://www.python.org/downloads/).
On the installer's **first screen, tick "Add python.exe to PATH"** before pressing Install. That
single checkbox is the difference between the commands below working and saying *"not recognized"*,
and it is unticked by default.

**macOS** — download from [python.org/downloads](https://www.python.org/downloads/) and run the
`.pkg`. (macOS ships its own Python, but it is older and Apple asks you not to use it for your own
work.)

**Linux** — you almost certainly have it. If not: `sudo apt install python3 python3-pip` on
Debian/Ubuntu, `sudo dnf install python3 python3-pip` on Fedora.

**Now open a terminal.** This is the window you type commands into:

| | How to open it |
| :--- | :--- |
| Windows | Press <kbd>⊞ Win</kbd>, type `powershell`, press <kbd>Enter</kbd> |
| macOS | Press <kbd>⌘ Cmd</kbd>+<kbd>Space</kbd>, type `terminal`, press <kbd>Enter</kbd> |
| Linux | <kbd>Ctrl</kbd>+<kbd>Alt</kbd>+<kbd>T</kbd>, or find *Terminal* in your applications |

Type this and press <kbd>Enter</kbd> — `python` on Windows, `python3` on macOS and Linux:

```
python --version
```

You should see something like `Python 3.13.1`. If the number is **3.10 or lower**, install a newer
one. If you get an error instead, see [When something goes wrong](#when-something-goes-wrong).

### 2. Install Maiman

One command. On Windows:

```
pip install maiman
```

On macOS and Linux:

```
python3 -m pip install maiman
```

It takes a few seconds and ends with a line like:

```
Successfully installed maiman-0.4.1 numpy-2.3.5
```

That is the whole installation. NumPy is the only thing it brings with it, and the interface is
inside the package — there is nothing else to download and nothing to build.

### 3. Start it

```
maiman serve
```

You will see:

```
Maiman Studio session server
  56 components
  http://127.0.0.1:8765/
```

**Now open that address in your browser** — copy `http://127.0.0.1:8765/` into the address bar, or
<kbd>Ctrl</kbd>-click (<kbd>⌘</kbd>-click on a Mac) the link in the terminal.

Leave the terminal window open. It is running the simulator; closing it stops the program. The
address is on your own machine only — `127.0.0.1` never leaves your computer, and nothing is sent
anywhere.

### 4. The first five minutes

A complete coherent link is already on the canvas when the page opens. You do not have to build
anything to see it work.

1. **Press Run.** The engine simulates the link and the page fills in: a constellation diagram, the
   measurements beside it, and a log where every line is a number that came back from the engine.
2. **Click a block** — the fiber, say. Its parameters appear in the inspector on the right.
3. **Change one and press Run again.** Set the fiber's length to 200 km and the received power
   drops by exactly the amount its attenuation says it should. Every number on screen is computed,
   not drawn.
4. **File → Open** has other links to look at: an eye diagram on a direct-detection link, a WDM
   spectrum on an optical spectrum analyser, a coded link.
5. **File → Save** writes a `.maiman` file — a plain text description of your link that you can
   keep, re-open, or send to someone.

To add blocks of your own: drag one from the palette on the left, then drag from one block's output
dot to another's input dot to wire them.

### Stopping it

Click the terminal window and press <kbd>Ctrl</kbd>+<kbd>C</kbd>. It prints `stopping` and exits.
Closing the terminal window works too.

To use it again another day, you do **not** reinstall — just open a terminal and type
`maiman serve` again.

### Updating

```
pip install --upgrade maiman
```

### When something goes wrong

These are the real failures, in the order they happen to people.

| What you see | What it means | What to do |
| :--- | :--- | :--- |
| `'python' is not recognized…` (Windows) | Python is installed but Windows cannot find it | You missed *"Add python.exe to PATH"*. Re-run the Python installer, choose **Modify**, and tick it. Or use `py` instead of `python` |
| `command not found: python` (macOS/Linux) | The command is `python3` here | Type `python3`, and `python3 -m pip` instead of `pip` |
| `'pip' is not recognized` | pip is there but not on the path | Use `python -m pip install maiman` (Windows) or `python3 -m pip install maiman` |
| `error: externally-managed-environment` | Your Linux or Homebrew Python protects itself from stray installs. This is normal and not a fault | `pipx install maiman` — install pipx first with `sudo apt install pipx` or `brew install pipx` |
| `'maiman' is not recognized` **after** a successful install | The program installed, but its folder is not on the path | `python -m maiman.server` does exactly the same thing |
| `OSError: [Errno 98] Address already in use` | Something is already on port 8765 — usually a copy you left running | `maiman serve --port 8766`, and open that number instead |
| Windows Defender Firewall asks for permission | It asks about any program that opens a port | **Cancel** is safe. The server only listens on your own machine and does not need network access |
| The browser says it cannot connect | The server is not running, or you typed the address before it started | Check the terminal still shows the `http://127.0.0.1:8765/` line and no error under it |
| The page loads but Run does nothing | The browser cannot reach the engine | Make sure the address is `127.0.0.1` and not `0.0.0.0`, and that the terminal is still open |

### If you would rather write Python

The interface and the library are the same engine. Anything the page does, a script can do — and a
sweep of two hundred cases is easier written than clicked.

Put this in a file called `first.py`:

```python
from maiman import Graph, SimulationContext
from maiman.components import CWLaser, Fiber, PowerMeter

ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=16, sequence_length=64)
link = Graph(ctx)

laser = link.add(CWLaser(power=0.0, wavelength=1550.0))
fiber = link.add(Fiber(length=80.0, attenuation=0.2))
meter = link.add(PowerMeter())
link.chain(laser, fiber, meter)

print(f"received power: {link.run()[meter].power_dbm:.2f} dBm")
```

Run it with `python first.py`, and it prints:

```
received power: -16.00 dBm
```

0 dBm launched, 80 km at 0.2 dB/km, so −16.00 dBm out. Nothing in that number is a lookup: the
laser makes a field, the fiber attenuates it, the meter integrates it.

The [`examples/`](examples/) folder has a dozen more, each one runnable with
`python examples/<name>.py`.

### Checking your machine

```
maiman devices
```

reports what the propagation kernels can run on:

```
Array back-ends
  numpy        available
  cupy         not installed

Only NumPy here, so there is nothing to cross-check.
A GPU needs `pip install cupy-cuda12x` and a CUDA device.
```

NumPy is enough for everything in this README. A GPU only matters for long split-step runs.

---

## The interface

**It runs.**

    pip install maiman
    maiman serve

then open `http://127.0.0.1:8765/`. The interface is a file inside the package, so installing the
engine installs it — there is nothing to build and no checkout to be standing in. From a checkout,
`python -m maiman.server` is the same command by another name. Press Run and the page posts its graph to the engine and draws
what comes back: the constellation, the measurements, the block captions, and a log in which every
line is a fact from the response. Change a parameter in the inspector and run again, and the
numbers move because the physics moved — set the fiber to 200 km and the received power drops by
exactly the 24 dB the extra loss costs, while the constellation collapses, because the dispersion
compensator is still set for the old span.

**The canvas draws the project document itself.** The blocks, the wires and the parameter values
are not written into the page. They are read from the same `.maiman` document the server accepts,
exported from the graph that produced the reference numbers, and Run posts that object straight
back. So the picture on the canvas, the values in the inspector and the graph the engine executes
are one description instead of three kept in agreement by hand — the hand-written copy that used
to live in the page had already drifted, and was missing four of the fiber's parameters.

**It is an editor.** Drag a block to move it. Drag from an output port to an input to wire them.
Click a component in the palette to add it, select a block or a wire and press Delete to remove it,
ctrl+Z to take it back. Then Run, on a graph that did not exist a minute ago.

A connection is refused while it is being dragged rather than when Run is pressed, and the refusal
says why — `binary cannot drive optical`, `fib.in already has a source`, `a block cannot feed
itself`. Port types are what make that possible, and it is the reason they have been in the signal
model since the first commit. An input with nothing feeding it is drawn hollow, because the engine
will refuse to run the graph and showing which port is the problem beats reporting it afterwards.

Built without React Flow, on the SVG canvas that was already there. That was a judgement call
against what PRODUCT.md had written down, taken because React Flow means npm and a bundler and this
project has no build step at all — the page still opens straight off disk with nothing installed,
which is how the README asks you to read it. The interaction layer is the part that would have had
to be written either way.

**A run says how far along it is.** Not a spinner: the response streams, and the bar in the menubar
is drawn from a fraction the engine has already reported — components behind this point, plus the
fraction *within* the one running. That second part is what makes it useful, because in a link with
a long span one block is the entire run: the split step reports the distance it has travelled after
every step, so a 16-second span moves the bar 268 times instead of resting on `fib` for the whole
of it. It is deliberately not a time estimate. A run has no estimate before it starts, and the one
honest quantity available is how much of the work is behind it.

Stop aborts the response rather than the run. Block-mode execution has no safe place to stop
halfway, so the engine finishes and its answer is discarded — and the log says that in those words,
because an interface that implies a cancellation it did not perform is worse than one that cannot
cancel. The button was enabled for the whole of every run and wired to nothing until there was
something real to abort.

**Sweeps.** A single run answers *what does this link do*; a sweep answers *how far can it go*,
which is the question that gets asked more often. Pick a block and a parameter, give it a range,
and the curve appears beside the form that made it. Repeats draw the spread at each point, because
one BER estimate at a marginal operating point is a sample and not an answer. The endpoint sends
back **numbers, not pictures** — a curve is made of scalars, and shipping a 96×96 histogram at
every point would be megabytes to draw a line. Seven points of a coherent link is 6 kB.

The plot opens on the metric that *moved*. A sweep is run to watch something change, so the one
that changed most is the one worth showing — which is also what stops it opening on the analyser
downstream of the decoder, whose EVM is exactly zero at every point because it is looking at
symbols that have already been decided.

**Points run at the same time, and the answer does not know it.** `sweep(..., workers=4)` — or
`workers=None` for one per core — runs four points at once, each on its own deep copy of the graph.
Measured 3.2× on an eight-point nonlinear span sweep, and **bit-identical** to the sequential
result, in sweep order, however many workers ran it. That last part is the whole claim: a sweep
whose numbers moved with the thread count would be worthless as a measurement, and the failure
would not look like a crash — it would look like a curve with the right shape and the wrong values.

Threads rather than processes. Measured, the two are the same speed on this work — 3.25× against
3.14× on twelve workers — because the time goes into numpy's FFTs, which release the GIL, and
because a split step is memory-bound long before it is core-bound. Threads then win on everything
else: nothing to pickle, no interpreter to start, and no objection to a component you defined in a
notebook. The copies are what make it safe — a run *writes* a point's overrides onto the components
and rolls them back after, so two threads sharing one graph would produce numbers belonging to
neither point.

The default is one worker, because a sweep of four cheap points is slower to parallelise than to
run. The session server uses four: a browser is holding a connection open for the whole of a sweep,
which is the case this was built for, but taking all twenty-four cores for a curve somebody asked
for by clicking a button is not a trade a design tool gets to make on its own.

**The spectrum is drawn, not summarised.** An Optical Spectrum Analyzer anywhere on the graph puts
its trace in the dock: wavelength across, dBm per resolution bandwidth up, the peak marked at the
wavelength it was found at. The axis is labelled in the analyser's own resolution because the trace
*is* — an OSA reports power within its resolution, so widening the setting lifts the ASE floor
decibel for decibel and leaves the channels, already narrower than either setting, exactly where
they were. That asymmetry is the whole reason an OSNR figure is meaningless without the bandwidth
it was quoted in, and it is one parameter and one Run away from being seen rather than described.
The OSNR beside the trace comes from an OSNR meter on the graph and is left empty when there is
none: a figure read off the displayed curve would move with the resolution knob and would not be
the OSNR anyone means. `examples/wdm_osa.maiman` opens four channels on the 100 GHz grid through
two amplified spans, which is what the trace baked into the page is a run of.

**Open and save.** `File → Save` writes a `.maiman` file: the same document the canvas draws, the
same one the server runs, with the block positions folded in. `File → Open` reads one back. Both
go through the browser — the file is chosen in the operating system's own picker and read locally,
and nothing is posted. A server that opened or wrote a path it was handed would be a different and
much worse program, and it would gain nothing: the file belongs to the person at the keyboard.
A file that is not a project says so and leaves the canvas alone rather than emptying it first and
explaining second.

The page also still opens straight off disk with nothing running, which is how it should be read
if you only want to look. It says which mode it is in rather than leaving it to be inferred: a
badge in the results dock reads **live** when the numbers came from this session's last run,
**stale — graph edited** when the graph has changed since, and **reference** when they came from
the run baked into the file. Numbers from four days ago and numbers from the last click must never
look alike. See [DESIGN.md](DESIGN.md) for why the rest of it looks like this.

![The Maiman Studio schematic editor on its paper ground](docs/images/studio-paper.png)

It ships two grounds and defaults to paper, because a schematic is a document before it is a
screen and its plots leave the tool for reports and papers. Graphite is one click away:

![The same editor on its graphite ground](docs/images/studio-graphite.png)

Every wire colour is a wavelength rather than a preference — optical C-band cyan, electrical amber,
binary slate, symbol violet, metric magenta — so a glance at a link tells you what travels down it.
A typed-port system that refuses invalid wiring at edit time is worth nothing if the types are
invisible.

---

## Try it

```bash
pip install maiman
maiman serve
```

Step by step, assuming neither Python nor a terminal:
**[Installing and running it](#installing-and-running-it)**.

Or, to work on it:

```bash
pip install -e ".[dev]" && pytest
```

```python
from maiman import SimulationContext, Graph
from maiman.components import CWLaser, Combiner, Fiber, PowerMeter

ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=16, sequence_length=64)
g = Graph(ctx)

laser = g.add(CWLaser(power=0.0, wavelength=1550.0))  # 0 dBm
fiber = g.add(Fiber(length=80.0, attenuation=0.2))  # 80 km, 0.2 dB/km
meter = g.add(PowerMeter())
g.chain(laser, fiber, meter)

print(g.run()[meter])  # PowerReading(-16.000 dBm; 1550.00nm=-16.000dBm)
```

Two carriers stay two independently sampled bands, which is the point of the signal model:

```python
g = Graph(ctx)
ch1 = g.add(CWLaser(wavelength=1550.0, label="ch1"))
ch2 = g.add(CWLaser(wavelength=1551.0, label="ch2"))
mux = g.add(Combiner(2))
fiber = g.add(Fiber(length=80.0, attenuation=0.2))
meter = g.add(PowerMeter())

g.connect(ch1, mux["in0"])
g.connect(ch2, mux["in1"])
g.chain(mux, fiber, meter)

print(g.run()[meter])
# PowerReading(-12.990 dBm; 1551.00nm=-16.000dBm, 1550.00nm=-16.000dBm)
```

Each band carries its own centre frequency, so channel spacing never enters the sample rate.
Put those two lasers 6 THz apart instead of 125 GHz and nothing about the run changes — which is
exactly what a single-carrier signal model cannot do.

## Results

`python examples/ook_link.py` builds a 10 Gb/s OOK link and characterises it. Abridged output:

```
Receiver sensitivity (back to back)        Dispersion-limited reach (0 dBm launch)
  launch      Q    BER (from Q)  counted     distance     Q    BER (from Q)
  -22 dBm   1.59     5.61e-02    460/8184        0 km   94.75    0.00e+00
  -20 dBm   2.51     6.09e-03     49/8184       40 km    9.26    1.03e-20
  -19 dBm   3.15     8.17e-04      7/8184       60 km    6.48    4.63e-11
  -16 dBm   6.25     2.11e-10   none counted    80 km    3.76    8.58e-05
  -14 dBm   9.83     3.99e-23   none counted   120 km    0.63    2.65e-01
```

Two things are worth reading off that. **Sensitivity is −16 dBm** for a Q of 6 — the right figure
for a PIN into a plain 50 Ω load. **Reach is ~62 km**, and it is set by dispersion, not by loss:
with dispersion switched off the same 60 km span gives Q = 15.4 instead of 6.5. That is the
textbook result for uncompensated 10 G NRZ on standard fiber.

The two columns are also a cross-check on each other. 120 km of 0.2 dB/km is 24 dB, and launching
0 dBm through it gives the same Q as launching −24 dBm back to back. Modulator, fiber, detector,
filter and analyzer all have to agree for that to hold; it is
[a test](tests/test_ber.py), not a coincidence.

Both curves come from `sweep()`, and the same script writes the schematic to
[`examples/ook_link.maiman`](examples/ook_link.maiman) — versioned JSON, diffable, runnable headless.

### Coherent

`python examples/coherent_link.py` runs the same treatment on a 32 GBd coherent link —
PRBS → Gray-coded M-QAM → IQ driver → IQ modulator → 90° hybrid with balanced detection —
and finds the received power each format needs for a BER of 1e-3 — a **soft-decision** FEC
threshold, not the hard-decision one; see [What FEC is for](#what-fec-is-for) for what
RS(255, 239) actually carries from there:

```
format     rate      launch for BER 1e-3     SNR there   EVM there
BPSK          32 Gb/s      -38.0 dBm received        6.9 dB     45.2%
QPSK          64 Gb/s      -35.0 dBm received        9.9 dB     32.0%
8-QAM         96 Gb/s      -29.7 dBm received       15.2 dB     17.5%
16-QAM       128 Gb/s      -27.5 dBm received       17.4 dB     13.5%
32-QAM       160 Gb/s      -23.8 dBm received       21.1 dB      8.8%
64-QAM       192 Gb/s      -21.7 dBm received       23.2 dB      6.9%
128-QAM      224 Gb/s      -18.3 dBm received       26.6 dB      4.7%
256-QAM      256 Gb/s      -16.3 dBm received       28.7 dB      3.7%
```

The required-SNR column is the one to check against a textbook: 9.9 / 17.4 / 23.2 / 28.7 dB are
the standard figures for QPSK through 256-QAM at 1e-3. The step from BPSK to QPSK costs exactly
3 dB — the same energy per bit for twice the rate, which is why coherent systems start at QPSK
and never look back.

**The odd orders are rectangular, not cross.** 8-QAM is 4×2, 32-QAM is 8×4, 128-QAM is 16×8: the
extra bit goes on I. A cross — a square with its corners cut off — is the better constellation, 2.30
dB peak-to-average against the rectangle's 3.48 at 32 points, and worth about a decibel of required
SNR. Two things argued against it. A cross **cannot** be exactly Gray coded, which is a proved
result rather than a gap in the construction, so its labelling is a published heuristic that would
have to be transcribed rather than derived — and this project does not transcribe models it cannot
check. The rectangle, being a product of two Gray PAMs, extends everything already written for
square QAM by arithmetic: the levels, the slicer, and an error rate that **is** the square formula,
term for term, with `M_I` and `M_Q` put in separately. If cross QAM arrives it belongs beside this,
chosen by a parameter, not instead of it.

A rectangle is not invariant under a quarter turn, and two things here had assumed every
constellation was. The blind phase search now covers the constellation's **own** rotational
symmetry — `[0, π)` for a rectangle, `[0, π/2)` for a square — because an 8×4 grid turned a quarter
is a 4×8 grid, a different alphabet, and a search that never looked past π/2 could not find an
offset of 0.6π at all. Measured: 0.0022 mean square error with the widened search against 0.1176
with the old one. And differential *quadrant* encoding is refused on an odd order rather than
quietly mis-encoding it — a rectangle's blind ambiguity is a half turn, which quadrant differencing
does not address.

Nothing here is configured to come out right. The shot-noise-limited SNR is asserted against
`R·P/(2qB)`, the counted symbol errors against
[`ser_qam()`](src/maiman/modulation.py), and the modulator's 3 dB against `10·log10(2)`.

### What FEC is for

A link is not specified at the bit error rate it delivers. It is specified at the rate its
*decoder* delivers, and every sensitivity figure in this project was quoted at a pre-FEC threshold
with nothing behind the word.

The code is **RS(255, 239) over GF(2⁸)**, the generic FEC of ITU-T G.709 — the one in the OTN
frame overhead. It is specified completely enough to build from the standard: the field is
generated by `x⁸+x⁴+x³+x²+1`, sixteen parity symbols correct up to `t = 8` symbol errors anywhere
in the codeword, and the overhead is 6.69 %. Syndromes, Berlekamp-Massey, a Chien search and
Forney's magnitudes; the generator polynomial is *derived* and its sixteen roots asserted rather
than a coefficient table being transcribed.

**A byte is a byte.** Eight bad bits in one symbol cost the same as one — which is why this code
was chosen for a bursty channel, and why every expression below starts from the *symbol* error
rate. A test puts 64 bad bits in eight bytes (correctable) beside 9 bad bits in nine (not).

**On a real link, counted end to end.** 10 Gb/s OOK, a real modulator, a photodiode with its shot
and thermal noise, a filter, and a blind slicer deciding — no formula anywhere in the path:

```
received   pre-FEC BER   post-FEC BER   symbols repaired   codewords failed
-20.0 dBm    5.99e-03      5.73e-03            22              37/40
-19.5 dBm    2.39e-03      1.18e-04           182               1/40
-19.0 dBm    7.72e-04      0.00e+00            63               0/40
-18.5 dBm    1.72e-04      0.00e+00            14               0/40
```

That cliff is about a decibel wide, and the top row is the honest half of it: **below threshold the
decoder makes things worse.** A bounded-distance decoder does not degrade gracefully — past `t` it
can find a wrong error pattern and apply it confidently, so 5.99e-3 in comes out at 5.73e-3 with
miscorrections mixed in. Clipping the output rate at the input rate would look tidier and would
flatter the code. A test asserts the post-FEC rate is *higher* there.

**A correction this work forced.** The sensitivity tables above are quoted at a pre-FEC BER of
1e-3. RS(255, 239) takes 1e-3 to about **1.1e-6** — nowhere near the 1e-15 a transport system is
specified at. To reach 1e-15 it needs roughly 1e-4 at its input. So **1e-3 is a soft-decision
threshold, and this is a hard-decision code.** The word "FEC" was covering both; it no longer does.

The closed-form curve and the decoder are independent and are checked against each other. They
disagreed by a factor of four the first time and **the expression was the one that was wrong**: a
wrong byte is wrong in *one* bit at these error rates, not four. Conditioning on "at least one of
eight flipped" gives `p/p_s`, which is about an eighth at small `p`, not the half it would tend to
at `p = 0.5`. The Monte Carlo caught it.

**Two things the engine did not have, and now does.** A coder makes more bits than it is given, and
the run window here is a fixed number of symbols *on the line* — so the window is the line rate and
coding makes the payload smaller, not the line faster. `FECEncoder` is therefore a source: it makes
its own payload, codes it, fills the window exactly, and emits the uncoded payload on a second port
for the decoder to check against. That is what an OTN framer does.

And there was no digital output from a direct-detection receiver at all. Every bit in this library
used to be decided inside `BERAnalyzer`, which decides *with the transmitted sequence in hand* and
emits a number — the right way to measure a link and no way to build one. `Slicer` is the missing
block: blind in both choices a decision circuit makes, with the sampling instant at maximum eye
variance and the threshold from Lloyd's two-means. It converges on the midpoint of the rails rather
than the optimum, which sits nearer the zero rail because a one carries more noise — a real penalty,
stated rather than smoothed over.

### What soft-decision FEC is for

The hard-decision code above decides each bit, then corrects. This one never decides until the end:
it works on **log-likelihood ratios** all the way through, and how near each symbol sat to a
decision boundary is exactly the information the other one throws away. That is worth two to three
decibels, and here is what it looks like on the same engine.

32 GBd 16-QAM coherent, staircase code at 16.4 % overhead, four blocks, eight iterations — a real
IQ modulator, a real hybrid with balanced detection and its own shot noise, a max-log demapper, and
an iterative decoder working on what that produced:

```
received     pre-FEC BER   post-FEC BER  |  RS(255,239) at the same input
-26.0 dBm      1.49e-02      6.04e-04    |    1.49e-02  — corrects nothing
-25.0 dBm      7.36e-03      0           |    7.21e-03  — corrects nothing
-24.0 dBm      3.20e-03      1.78e-05    |    1.02e-03
-23.0 dBm      1.34e-03      0           |    8.75e-06
-22.0 dBm      4.73e-04      0           |    3.31e-09
```

**At −25 dBm the line is running at 7.4e-3 and the payload comes out exact**, while the classic
G.709 code at that same input returns essentially what it was given — every one of its codewords is
far past `t = 8`. Reading across, the hard code needs about −22 dBm to reach 1e-9 and this one is
already clean at −25: roughly three decibels of sensitivity, which is the number the whole
soft-decision argument is about.

**The code is a braided BCH staircase** — Smith, Farhood, Hunt, Kschischang and Lodge,
*J. Lightwave Technol.* 30(1), 2012 — with Chase-II component decoding (Chase, 1972) and Pyndiah's
soft-output variant (Pyndiah, 1998). It is the family OIF's **oFEC belongs to and it is not oFEC**:
the 400ZR Implementation Agreement is normative about an interleaver and a framing this does not
reproduce, and a block that carried the name while being a reconstruction would be the one thing
this project refuses. The parameters are stated as a choice, not as a standard.

**Three things had to exist before any of this could be wired up**, and each is its own block:

`SoftDemapper` turns symbols into LLRs, max-log, with the noise variance estimated blind because a
receiver has no other option. `PortType.SOFT` is a sixth port type, so a soft output physically
cannot be wired into a block expecting bits — that mistake does not crash, it runs and quietly
gives back the decibels this whole path exists to recover, so the type system is what makes it
unrepresentable. And `SoftFECEncoder` is a *source* for the same reason the hard one is: a coder
makes more bits than it is given, the window is a fixed number of symbols on the line, so coding
shrinks the payload rather than speeding the line.

**The last block of a stream is provisional.** Every bit is checked twice — once by its own
stripe's rows, once transposed as a column by the next stripe — so the final block has half its
protection missing until the next one arrives. With two blocks in a window that is half the payload
and it sets a floor near 1e-4 that has nothing to do with the channel. It is why real decoders run
a sliding window and never emit the newest block, and there is a test on it rather than a footnote.

Getting the iteration right took three attempts, and all three failures are now tests. A
no-competitor reliability *below* the input's own made the extrinsic negative and destroyed bits the
decoder had never touched — 268 errors out of zero corrections, which is the signature that located
it. A loop that let each stripe overwrite the previous one's result discarded half of what the
braiding exists to produce. And a component decoder fed its own previous extrinsic agreed with
itself, so its output collapsed to zero and its corrections quietly unwound two passes later.

### How the flagship resolves its quarter turn

Every blind stage in a coherent receiver gets the phase right *modulo* the
alphabet's own symmetry — rotate a square QAM constellation by 90° and nothing about the received
samples changes, so no amount of data distinguishes the two. Something has to break the tie, and
there are exactly two ways: difference the quadrant out, or know some of the symbols.

The link used to do the first. `DifferentialDecoder` has to **slice** in order to difference, and
that is fatal to anything downstream that wants a measurement rather than a decision — the distance
from its output to the nearest constellation point is *exactly zero*, and a demapper reading it
returns LLRs of order 1e29.

It now does the second. `PilotInserter` overwrites every 64th symbol with one the receiver already
knows; `PilotPhaseRecovery` estimates `angle(Σ r·conj(p))` over those positions and removes one
constant angle from the whole window. It decides nothing.

```
                      differential            pilots
overhead              0 % of the rate         1.6 % of the rate
cost near threshold   2× the errors           none
EVM                   7.34 %                  7.31 %
counted BER           4.20e-10                3.56e-10
soft output           LLR ~1e29 (useless)     LLR ~121, real spread
blocks on the canvas  19                      18
```

**That table is the measurement taken when the change was made, and the canvas has moved since**:
soft-decision FEC arrived afterwards and took it to twenty. The EVM and BER columns are of that
link, not of the one the flagship is now — which reads 7.39 % and 5.27e-10, because it carries a
coder, a demapper and a decoder that the eighteen-block version did not. What the table is
evidence for is the *comparison*, both halves of which were measured on the same canvas on the same
day.

**It was cheaper in blocks, not just in errors.** Two disappeared with the differential
arrangement: a second mapper that existed only to hand the error analyser a non-differential copy
of the same bits, and a second analyser that existed only because an EVM taken after a decision
device reads zero however bad the link is. Neither had a reason to exist any more, so the canvas
lost a block while gaining a capability.

**Pilots replace payload, they do not add to it** — the window is a fixed number of symbols on the
line, so a pilot costs a data symbol. They are drawn from the alphabet's own **outermost** points,
which carry the most energy a phase estimate can be made from and, unlike an off-grid pilot, are
legal symbols: unit-power QPSK pilots inside a 16-QAM link sit exactly between four points, and the
error count then measures the pilots rather than the channel. The sequence is not constant either,
because a repeated symbol puts a line in the spectrum at the pilot rate.

The diagnostics report the **residual** — how far the removed angle sat from an exact multiple of
90°. Everything upstream is supposed to leave only an ambiguity, so a few milliradians says that
held. On the shipped link it is 5.4 mrad over 128 pilots. A large one would mean the constant
removed here was covering for a stage that did not do its job.

### What the flagship's post-FEC number measures

The shipped coherent link now runs end to end on soft-decision FEC: a staircase coder as the
source, pilots for the quadrant, a max-log demapper, and an iterative decoder. Twenty blocks, and
about a second.

**Its post-FEC error rate is dominated by the window's edges, and that is worth knowing before
reading it.** This link filters *circularly* — the root-raised-cosine shaping and the dispersion
compensator both wrap — so the first and last symbols carry a fold from the far end of the
sequence. The analyser is allowed to discard them and does, through `ignore_edges`. **A decoder is
not**: it has to decode every bit it is handed.

Measured on the shipped graph: all seven pre-FEC bit errors sat in the first 0.05 % and the last
0.07 % of the window. The interior was clean. A real stream has no edges, so this is an artefact of
simulating a finite window rather than anything about the link — and it is left visible rather than
buried under a longer window, because a post-FEC rate that quietly measures the simulation boundary
is the kind of number this project exists not to print.

**Pilots are erased, not believed.** A pilot overwrote whatever the coder put in that symbol, so
those bits say nothing about the codeword — and zero is exactly what that means. Not a confident
value: the receiver knows the pilot, but it is not the codeword bit, so anything confident is
confidently false. The difference is not subtle, measured on a 9.9e-3 channel:

```
pilots treated as       post-FEC BER
erasures (LLR = 0)        4.9e-04
ordinary bits             2.8e-02   ← worse than the decoder's own input
```

A code corrects roughly twice as many erasures as errors, and that is the whole of why.

**Carrying a block code quantises the window.** A staircase block is 16384 coded bits, so at four
bits per symbol the link runs in multiples of 4096 symbols and refuses anything else, naming a
length that works. That is a real reduction in what the sequence-length control accepts on the link
people open first — and it is not an implementation wart: a block code quantises a frame in
hardware too, and nobody runs an OTN link at 256 symbols either.

### A coherent link that runs on soft decisions

`examples/coherent_sdfec.maiman` — open it from the File menu. 32 GBd 16-QAM, the staircase code at
16.4 % overhead, decoded on log-likelihood ratios from a max-log demapper rather than on bits:

```
pre-FEC BER   post-FEC BER   blocks
  7.36e-03         0            4
```

At −25 dBm the line is running at a few times 1e-3 — past what RS(255,239) can touch — and the
payload comes out exact.

**The flagship carries this too, now.** This project stays because it measures a different thing:
an ideal laser and no fiber, so what it reports is the decoder. The flagship has a real span and
real carrier recovery, and its post-FEC number is dominated by the simulation window's edges —
see [What the flagship's post-FEC number measures](#what-the-flagships-post-fec-number-measures).

So the coded link declares an **ideal carrier** — both lasers at 1550 nm, no linewidth — which
removes the ambiguity rather than resolving it. A real system pairs soft FEC with pilot symbols,
which this library does not have; that absence is why the idealisation is declared here rather than
worked around quietly. Carrier recovery is demonstrated on the flagship, where it belongs.

**Four blocks, not two.** Every bit is checked twice — by its own stripe's rows and, transposed, by
the next stripe's — so the last block of a stream has half its protection missing until the next
one arrives. At two blocks that is half the payload, and it floors the error rate near 1e-4 for
reasons that have nothing to do with the channel.

### What EDFA saturation is for

Until now the EDFA's docstring said, in its own words, that **saturation was a clamp and not a
model**: `max_output_power` held the output at a ceiling and the gain fell to whatever achieved
it. That has a kink in it. Below the ceiling the amplifier was perfectly ideal; above it the gain
fell as `1/P_in`. Real amplifiers do neither — they compress from the first photon, smoothly, and
never hit a wall.

The replacement is the standard steady-state description (Saleh, Jopson, Evankow and Aspell,
*IEEE Photon. Technol. Lett.* 2(10), 1990), which is implicit in the gain:

```
G = G₀ · exp( −(G − 1) · P_in / P_sat )
```

solved exactly — Newton on `ln G`, to 1e-12 — rather than approximated. The root is bracketed by
`[0, ln G₀]` and the function is increasing and convex, so Newton started at the upper end descends
onto it monotonically: no bisection fallback and no iteration cap that could quietly hand back a
half-converged gain. A test puts the answer back into the equation, because that is the only honest
check on a solver for an implicit relation.

**The parameter is the one a datasheet quotes** — `saturation_power` is the *output* power at 3 dB
of gain compression — and is converted to the model's own `P_sat` by `P_3dB·(G₀−2)/(G₀·ln2)`, which
follows from setting `G = G₀/2` above. For a large gain that is `1.443 · P_3dB`; the `(G₀−2)/G₀`
factor is carried in full because it is 0.98 at 20 dB of gain and 0.80 at 10, and a version that
dropped it as "large gain" passes a 20 dB test and fails a 10 dB one. Both are in the suite.

A 20 dB amplifier with a +17 dBm saturation power:

```
in dBm   out dBm   gain dB   compression
   -40    -20.00     20.00      0.00 dB
   -20     -0.06     19.94      0.06 dB
   -10      9.46     19.46      0.54 dB
     0     16.99     16.99      3.01 dB   ← the declared point, by construction
    10     21.65     11.65      8.35 dB
```

**Saturation is driven by total power, ASE included**, which is what makes an EDFA behave like a
fixed power source rather than a fixed gain. Feed one 20 dB amplifier more channels at −5 dBm each
and watch them share:

```
channels   total out   per channel
    1       13.61 dBm   13.61 dBm
    2       15.75 dBm   12.74 dBm
    4       17.58 dBm   11.56 dBm
    8       19.15 dBm   10.12 dBm
```

Eight times the input for 5.5 dB more output. Every doubling of the channel count costs the
channels already there about a decibel, which is the arithmetic a WDM operator plans around, and it
falls out of the model rather than being put in — the amplifier never sees channels, only the
power they add up to. A version that saturated each band against its own copy of the amplifier
would pass a single-channel test and be wrong here, so the suite checks it through a real graph
with a real combiner and not only through the gain function.

**It is off by default.** An amplifier that never compresses is an idealisation, and it is now
declared as one the way a zero-linewidth laser is: `saturate` is a flag and it starts false. That
keeps every existing result in this repository exactly where it was — the change moved the
manifests and not one number — and makes turning it on a decision someone takes. What it replaces
was worse than an idealisation, because a clamp claims to be a saturation model and is not.

`max_output_power` is retired. Old projects still open, with the parameter dropped — and unlike the
other retirements in this library that **does** change the link they describe, which is stated in
the code rather than glossed. The value it carried was fitted to a fiction.

### What happens between two steady states

The gain above is the one the erbium *settles* into. Getting there takes milliseconds, and what a
line system actually fears is the trip rather than either endpoint: an erbium amplifier shares one
inversion between everything passing through it, so take channels away and the survivors inherit
the gain those channels were using.

**This is an analysis, not a block, and the reason is a measurement.** The relaxation constant is
`τ / (1 + P_out/P_sat)` — for the 20 dB amplifier here, between 3.3 and 9.9 ms. A simulation window
of 4096 symbols at 32 GBd is **128 ns**. The fastest transient is twenty-five thousand windows
long, so within one window the gain is a constant, which is exactly what `EDFA` already assumes and
is right to. A per-sample block could not show any part of this.

So there is no `metastable_lifetime` parameter on the amplifier. **A parameter that cannot change
the result of a run is a control the interface cannot back**, and this library does not ship those.
The lifetime belongs to [`maiman.transient`](src/maiman/transient.py), which can use it.

**It is the dynamic form of the equation already in the component, not a second model beside it.**
Writing the Saleh reservoir in terms of `g = ln G`:

```
τ · dg/dt  =  ln G₀ − g − (e^g − 1)·P_in/P_sat
```

Set `dg/dt = 0` and it returns `G = G₀·exp(−(G−1)·P_in/P_sat)`, which is what `effective_gain`
solves by Newton. Same `G₀`, same `P_sat`, exactly one quantity added — so a test integrates the
ODE from a long way off and checks it lands on the component's own solver. Two code paths sharing
no arithmetic, meeting to **1e-9**.

Linearising about that rest point gives the effective constant, which is why a saturated amplifier
answers in a fraction of its own lifetime: the stimulated emission draining the reservoir is itself
proportional to how full it is. That closed form is tested against a step response measured from
the integrator, to a part in a thousand.

`python examples/edfa_transient.py` drops seven channels of eight from a −6 dBm comb:

```
                      one amplifier      down a chain of 8
gain before                15.63 dB
gain after                 18.84 dB
survivor excursion         +3.21 dB               +9.94 dB
settles in                 47.9 ms
```

**The chain is the number that matters**, and it accumulates *sub*-linearly rather than as eight
times the first: each amplifier down the chain starts less saturated than the one before, so it has
less compression left to give back. Eight times 3.21 dB would be 25.7; the measured figure is 9.94.
Chaining is exact rather than approximate — there is no feedback from a later amplifier to an
earlier one, so a span integrates in order, each taking the previous one's output on the same grid.

**What is absent.** One reservoir means one gain for every wavelength, and a real erbium transient
*tilts* the gain spectrum as the inversion changes, so a survivor's excursion depends on where in
the band it sits. The pump is implicit in `G₀` and does not respond — a deployed amplifier has a
control loop pushing back on exactly this excursion, and what is computed here is the uncontrolled
case, which is the one such a loop is sized against. Both are real and both are missing rather than
approximated.

### What RIN is for

Every noise in this project until now got *quieter*, relative to the signal, as the launch power
went up. Shot noise grows as the square root of the current and thermal noise does not grow at all,
which is why every sensitivity curve here keeps improving to the right. **Relative intensity noise
does not work like that.** It is a fluctuation *of the power itself*, so it scales with the signal,
and the ratio it fixes is one that no power budget moves.

`CWLaser` now takes a `rin` parameter in dB/Hz — the single-sided spectral density of the
fractional power fluctuation, `S_δP(f) / P̄²`, which is the quantity a laser datasheet quotes.
A sampled window carries a one-sided bandwidth of `fs/2`, so the fractional fluctuation per sample
is drawn from `N(0, rin·fs/2)` and the field is the square root of what is left (Agrawal,
*Fiber-Optic Communication Systems*, 4th ed., §4.6.2).

**The closed form is exact and the test is an equality.** Detect a CW laser with shot and thermal
noise off, and what is left in the photocurrent is the laser's own intensity noise:

```
RIN         measured SNR   1/(RIN·B_n)   the same, at ten times the power
-155 dB/Hz    56.264 dB      56.278 dB          56.264 dB
-145 dB/Hz    46.265 dB      46.278 dB          46.265 dB
-135 dB/Hz    36.268 dB      36.278 dB          36.268 dB
```

`B_n` is the receiver filter's *noise* bandwidth, which for the Gaussian shape used here is
`1.0645 × B` in closed form rather than approximately — which is why this can be checked as an
equality (0.014 dB) instead of an order of magnitude. The third column is the point: **+10 dBm
gives bit-identical SNR**. A model that got the variance right and this wrong would pass a
variance check and be useless for the one question RIN is asked.

**On a link it shows up as a penalty that grows with power.** 10 Gb/s OOK, PIN with shot and
thermal noise on, Q in dB against an otherwise identical ideal laser:

```
received      ideal Q  |  penalty at -155  |  at -145  |  at -135 dB/Hz
 -20 dBm       8.06 dB |      0.00 dB      |  0.00 dB  |    0.01 dB
 -15 dBm      17.98 dB |      0.00 dB      |  0.01 dB  |    0.13 dB
 -10 dBm      27.75 dB |      0.01 dB      |  0.12 dB  |    1.04 dB
  -5 dBm      36.49 dB |      0.08 dB      |  0.77 dB  |    4.30 dB
   0 dBm      41.54 dB |      0.24 dB      |  1.99 dB  |    7.63 dB
```

At −20 dBm the columns are indistinguishable — thermal noise is a hundred times larger and the
laser's intensity noise is invisible. Every 5 dB of extra power makes it more visible, not less,
until at 0 dBm a −135 dB/Hz laser has given back 7.6 dB of the Q the same link would have had.
The penalty is quoted as a difference at equal power on purpose: it does not depend on whatever
else limits the ideal column at the top of the sweep.

`rin = 0` is a **sentinel for an ideal laser, not 0 dB/Hz of noise** — 0 dB/Hz would be a
fractional intensity variance of one per hertz of bandwidth, which is not a laser. Every real
device sits between about −110 dB/Hz for a cheap Fabry-Perot and −165 dB/Hz for a good DFB, so the
top of the scale is free to carry the sentinel. It is the default, and the shipped links keep it:
turning it on by default would move every number in this file, which is a decision for whoever
wants the noise and not for the parameter's default.

The phase and intensity noises are drawn from **separate streams**, so sweeping the linewidth does
not move the intensity samples. Otherwise a plot of one against the other shows a coupling that is
not there.

### What timing recovery is for

Everything downstream of the receiver used to sample on a grid it was *given*. `IQSampler` takes
one column out of every symbol and the column is a parameter — exact when the transmitter's symbol
grid and the receiver's sample grid are the same one, which in a simulation they are by
construction, and wrong the moment anything in the path holds a delay that is not a whole number of
samples.

**One shipped block is enough to do it.** A silicon waveguide at a group index of 4.2 holds 14 ps
per millimetre. At 32 GBd a symbol is 31.2 ps, so a quarter of a millimetre is a ninth of one:

```
waveguide     delay | EVM off   errors off | EVM on    errors on     moved
      0 um   0.00 ps |   6.44%     0/1920  |   6.44%     0/1920    +0.00 ps
    250 um   3.50 ps |  18.39%    64/1920  |   6.43%     0/1920    -3.52 ps
    500 um   7.00 ps |  37.06%   675/1920  |   6.45%     0/1920    -7.00 ps
   1115 um  15.61 ps | 110.20%  1527/1920  | 2993.36%  1806/1920  +15.62 ps
```

The method is **one FFT bin**. A modulated signal is cyclostationary, so `|A|²` carries a line at
the symbol rate; the phase of that line *is* the timing — Oerder and Meyr, 1988. Nothing is
searched and nothing iterates. The correction is a phase ramp, which is an exact fractional delay
for a periodic band-limited window rather than an interpolation with a passband to argue about,
and the test asserts that forward-then-back returns the input to 1e-12.

**Half a symbol is the cliff, and past it the failure is a different one.** At 1115 µm the delay is
exactly half a symbol: both directions are the same distance, so whichever it takes puts the
instant in the right place and the *sequence* one place out. The constellation is fine and the
errors are total, which is the signature of a slip rather than of bad timing. No estimator reading
`|A|²` can do better — delay by a whole symbol and the intensity waveform is **identical**, so the
line has the same phase. That is framing, not timing, and it is resolved against a known reference
exactly as the quadrant ambiguity is.

The estimate carries the **magnitude** of the line it came from, because the phase of nothing is a
number like any other: a signal shaped to exactly the Nyquist bandwidth carries no symbol-rate line
at all, and a heavily dispersed one nearly none.

### What frequency recovery is for

A transmitter laser and a local oscillator are independent oscillators, and they are not on the
same frequency. This library already modelled that honestly — bands carry their own centre
frequency, so an LO tuned off the transmitter beats against the signal inside `CoherentReceiver`
exactly as two real lasers do. Nothing removed it blind until this block; the analyser removed it
*data-aided*, with the transmitted sequence in hand, which is what a bench instrument does and not
what a receiver can.

**How little it takes.** On the shipped 32 GBd 16-QAM link, 10 MHz — three hundredths of one per
cent of the symbol rate — is enough:

```
  offset | errors, no block | errors, block |  found offset | confidence
       0 |      0/1920      |    0/1920     |    +0.000 MHz |    21.7
  10 MHz |   1677/1920      |    0/1920     |   +10.254 MHz |    22.1
 100 MHz |   1796/1920      |    0/1920     |  +100.098 MHz |    21.6
   1 GHz |   1791/1920      |    0/1920     | +1000.000 MHz |    22.3
 3.9 GHz |   1790/1920      |    0/1920     | +3899.902 MHz |    22.0
```

EVM is left out of that table on purpose: once the constellation is spinning, the analyser's
common-gain fit has nothing to fit and the percentage it reports is not a measurement. The error
count still is.

For scale: a laser on the ITU grid is specified to ±2.5 GHz and a good tunable holds ±100 MHz, so
the offset a receiver actually meets is two to three orders of magnitude past the point where the
link stops working.

**The method is one FFT**, of the symbols raised to their own alphabet's rotational symmetry —
Leven, Kaneda, Koc and Chen, *IEEE Photon. Technol. Lett.* 19(6), 2007. Raise a symbol to that
power and every point that differed only by one of those turns lands in the same place: the
modulation becomes a constant and the offset becomes a tone at M times itself. The power is taken
from the geometry by the same `rotational_symmetry()` that bounds the blind *phase* search, which
is the only way the two agree — 4 for a square constellation, 2 for a rectangular one or for BPSK.

That makes the acquisition range fall out of the format, and a rectangular format wins twice:

```
format   M   unambiguous range   errors at 1 GHz off/on   confidence
QPSK     4        ±4 GHz              1448  ->  0            197
8-QAM    2        ±8 GHz              1671  ->  0             34
16-QAM   4        ±4 GHz              1791  ->  0             22
64-QAM   4        ±4 GHz              1888  ->  80            16
```

64-QAM's 80 is not residual offset — it is that link's own noise floor, which is 74 errors with no
offset at all. The confidence column falls with order because a higher-order alphabet strips less
cleanly, and it is reported rather than hidden: it is the spectral peak over the median, so a value
near 1 says the argmax found no line and the megahertz beside it are an accident with decimal
places.

**This is not the phase problem, and the order is not a preference.** `CarrierRecovery` removes a
phase that *walks*; this removes one that *ramps*. A phase search covers a quarter turn and averages
over a window, so a ramp steep enough to cross that quarter turn inside the window makes it slip
rather than track — and a slip looks like noise. Every deployed coherent receiver puts frequency
before phase.

**Past half the stripped bandwidth it aliases rather than degrades**, which is the failure worth
knowing because it does not look like one. Beyond `symbol_rate / (2M)` the tone wraps, and the
estimate comes back wrong by exactly `symbol_rate / M` **with the same confidence as a correct
one**: 4.1 GHz is reported as −3.9 GHz, at a confidence of 23.1 against 24.0 for a correct
estimate. A test asserts the aliasing explicitly, rather than leaving it as a range nobody checks.

Something with no rotational symmetry at all has no power that strips it, so it is refused at run
time rather than answered with an argmax over noise.

### Acquisition, and why it cannot be a sweep

The obvious way to widen that ±4 GHz window is to try candidate offsets and keep whichever looks
best. **It cannot work, and the reason is worth stating because it is not an implementation
limit.** An alias of `symbol_rate / M` is a phase advance of exactly one `2π/M` turn per symbol —
and `M` is *defined* as the turn that maps the alphabet onto itself. So the aliased constellation is
not merely similar to the correct one, it is identical, symbol for symbol. A sweep would be
searching for a difference that is not there. The ambiguity belongs to the alphabet, not to the
estimator.

So [`CoarseFrequencyRecovery`](src/maiman/components/dsp.py) does not look at the symbols at all. It
takes the **first circular moment of the received waveform's power spectrum** — where the band
*is*, before any of the modulation has been stripped:

```
offset = angle( Σ S(f)·exp(2πjf/fs) ) / 2π · fs
```

Circular rather than an ordinary centroid, and both reasons carry weight. A band straddling the
`+fs/2` wrap has no meaningful arithmetic mean and an exact circular one. And **a flat noise floor
contributes nothing to a circular moment**, because a constant spread evenly around the circle sums
to zero — so no floor has to be estimated, thresholded or subtracted, and *the block never has to
be told the signal's bandwidth or roll-off*. That last part is not a convenience. The obvious
alternative — slide a window of the signal's width and take the position holding the most energy —
must be told that width, and any window wider than the band sits on a plateau of equal energy whose
argmax is arbitrary: measured, assuming `1.6·R_s` for a band that was `1.2·R_s` put the answer
**3.9 GHz out**.

```
                       fine (M-th power)        coarse (circular moment)
reads                  symbols                  the waveform
unambiguous over       ±symbol_rate/(2M)        ±fs/2 — which is Nyquist, not the method
                       = ±4 GHz at 32 GBd       = ±256 GHz on the shipped link
accuracy               sub-MHz                  ~±150 MHz
past its range         confidently wrong        correct — the alias *is* the sampling
confidence figure      peak/median              resultant length, 0…1
```

**The two failure modes are opposites, and that is the whole argument for having both blocks.** The
fine stage's alias destroys the link and reports a healthy confidence while doing it. The coarse
stage's "alias" past `fs/2` is not an error at all: beyond Nyquist an offset *is* the wrapped value
in sampled data, and correcting to it is correct. Measured on the shipped link at 512 GHz sample
rate, +258 GHz reads as −254 GHz and the link still recovers to 7.50 % EVM with zero symbol errors.

**It has to run before dispersion compensation and before the matched filter**, and its port types
make it impossible to wire anywhere else — it takes a waveform and returns one. Both of those
stages are built around baseband: the compensator applies a quadratic phase measured from zero
frequency, and the matched filter is a root-raised-cosine centred there. A band sitting 20 GHz away
gets the wrong phase curve and is then largely filtered off.

Measured end to end on the shipped coherent link, detuning the LO and running the whole chain:

```
LO offset    coarse stage lands    fine stage residual    EVM      symbol errors
+20 GHz      +19.915 GHz           +85 MHz                7.41 %   0
+60 GHz      +59.890 GHz           +109 MHz               7.37 %   0
+200 GHz     +199.903 GHz          —                      7.41 %   0
```

Back to back is 7.39 %. Without the block the chain is broken past 4 GHz — at 4.1 GHz the same link
reads 6002 % EVM.

**It is deliberately coarse**, and the accuracy is set by the data's own spectral asymmetry rather
than by noise: a finite draw of random symbols does not have an exactly symmetric spectrum, and
±150 MHz over a 4096-symbol window is whatever that asymmetry is. That is ~25× finer than the range
the fine stage needs, which is the entire requirement — this block does not have to be accurate, it
has to be *unambiguous*. It also outlives the link: at −14 dBm launch the constellation is at 117 %
EVM and the acquisition estimate is still right to 226 MHz.

The **concentration** figure it reports is the resultant length of that same moment, and unlike the
M-th power's peak-to-median it is informative about the thing that actually goes wrong here, since
what degrades a circular mean is the spectrum ceasing to be a band. Walking the launch power down:
0.98 at +10 dBm, 0.88 at 0, 0.66 at −6, 0.43 at −10, 0.23 at −14.

`python examples/acquisition_link.py` runs one link with a matched filter in it through both
arrangements and prints the two tables side by side. Its middle rows are the interesting ones: the
fine stage reports −3.900 GHz for a +4.100 GHz offset at a confidence of 19.6, where a correct
estimate on the same link scores 23.2. Further out — at 20 and 100 GHz — the confidence does
collapse, to 4.4 and 3.3, but that is the matched filter having already destroyed the signal rather
than the estimator noticing anything. **Near the fold, which is where it matters, confidence does
not separate a right answer from a wrong one.**

**A large offset is not automatically a hard one**, which a test records so nobody re-reads it as
one. On a link that samples at the symbol rate with no matched filter, the sampler itself folds the
offset by `symbol_rate` before the fine stage sees it — at 60 GHz that fold lands on −4 GHz, inside
the fine stage's range, and at 64 GHz, exactly twice the symbol rate, it lands on zero. Both come
back with zero symbol errors and no acquisition stage at all. What makes an offset hard is a stage
downstream that is built around baseband.

### Both are in the reference design

The coherent link the studio opens with carries both stages, and its LO is tuned 200 MHz below the
transmitter. That is not an injected impairment — it is the removal of an idealisation. Two
free-running lasers are never on the same frequency to fifteen decimal places, and a good tunable
holds about this much; modelling them as exactly co-tuned was the unrealistic choice, in the same
way that modelling a zero-linewidth laser would be. The link already runs a realistic 100 kHz
linewidth for exactly that reason.

200 MHz is six thousandths of one per cent of the symbol rate. Here is what each stage is worth on
that link, with the detuning and without it — every row a real run of the shipped graph:

```
   LO      timing  frequency |    EVM      SNR      symbol errors
co-tuned     off      off    |   7.318%   22.71 dB      0/3968
co-tuned     on       on     |   7.387%   22.63 dB      0/3968
detuned      off      off    | 285.219%   -9.10 dB   3334/3968
detuned      on       off    | 292.993%   -9.34 dB   3473/3968
detuned      off      on     |  13.006%   17.72 dB      0/3968
detuned      on       on     |   7.342%   22.68 dB      0/3968
```

Three things are worth reading off it. **Frequency recovery is the difference between a link and
no link** — 3334 errors to none. **Timing recovery is worth 5.7 points of EVM, but only once the
offset is gone**: on its own it does nothing for a detuned link, because a spinning constellation
is not a timing problem. And **the two together return the detuned link to the co-tuned one**,
7.342 % against 7.318 %.

That last gap is the honest cost of the stages: on a link with an ideal LO they find 2.12 ps and
+0.000 MHz and correct them, and the correction is very slightly worse than leaving it alone —
0.07 points of EVM. A blind estimator has variance, and paying it is what buys the 278 points in
the row above. A receiver carries these stages because the impairment is normally there, not
because they are free when it is not.

### What carrier recovery is for

With ordinary 100 kHz lasers and no phase recovery, 16-QAM at 32 GBd does not close — and, more
tellingly, **launching more power stops helping**:

| launch | without recovery | with recovery |
| ---: | :--- | :--- |
| −14 dBm | 14.6 dB, BER 6e−3 | 22.1 dB, BER 4e−9 |
| −10 dBm | 15.1 dB, BER 4e−3 | 25.7 dB, BER 4e−18 |
| −6 dBm | 15.3 dB, BER 4e−3 | 28.7 dB, BER 1e−34 |
| −2 dBm | 15.9 dB, BER 2e−3 | 30.9 dB, BER 2e−56 |

Twelve dB of extra power buys 1.3 dB. Laser phase noise is a random walk, so it is not removable by
subtracting a constant or a line, and it puts a ceiling on SNR that no power budget lifts.
[`CarrierRecovery`](src/maiman/components/coherent.py) removes the ceiling using the blind phase
search of Pfau et al. Both halves of that claim are [asserted](tests/test_dsp.py) — the second
would be meaningless without the first.

### Reaching past the bench

Every coherent example above was back to back. `python examples/dispersion_link.py` puts the same
32 GBd 16-QAM link through real fiber, with loss and nonlinearity switched off so that chromatic
dispersion is the only thing acting:

```
            accumulated   spread          uncompensated              compensated
  span          [ps/nm]  [symbols]      EVM       SNR    errors      EVM       SNR
       0 km          0       0.0      1.68%    35.51 dB      0/3968     1.68%    35.51 dB
       5 km         85       0.8     17.04%    15.37 dB      5/3968     1.67%    35.52 dB
      20 km        340       3.3    217.64%    -6.75 dB   3336/3968     1.67%    35.56 dB
      80 km       1360      13.4   2542.45%   -28.11 dB   3698/3968     1.68%    35.52 dB
     400 km       6800      67.0   2686.86%   -28.58 dB   3692/3968     1.67%    35.55 dB
    1000 km      17000     167.4   3390.48%   -30.61 dB   3697/3968     1.68%    35.51 dB
```

Read the uncompensated column first, because it is the reason this block exists. **Five kilometres
— a metro hop — costs twenty decibels.** At 80 km the link is not degraded, it is gone: 3698 of
3968 symbols wrong is 93%, and blind guessing on 16-QAM gives 93.75%. The whole coherent phase had
been validated without ever meeting the impairment that dominates every real span.

The compensated column is the argument for coherent detection in one line. **Back-to-back quality
at every distance, with no penalty that grows with it.** Dispersion is an all-pass phase — it
rearranges the field in time and removes nothing — so a receiver that measures the field still
holds all of it, and one static filter puts it back. That is not error correction; it is inverting
an invertible operation. A direct-detection receiver squares the field at the photodiode, destroys
the phase, and can never do this at all, which is why it has to carry dispersion-compensating fiber
in the line instead.

The setting is sharp, and that is worth seeing rather than being told:

| compensator set to | error | EVM | SNR |
| ---: | ---: | ---: | ---: |
| 76 km | −68 ps/nm | 13.64% | 17.30 dB |
| 79 km | −17 ps/nm | 3.78% | 28.46 dB |
| **80 km** | **0** | **1.68%** | **35.52 dB** |
| 81 km | +17 ps/nm | 3.77% | 28.47 dB |
| 84 km | +68 ps/nm | 13.64% | 17.30 dB |

Being one kilometre out costs 7 dB. The symmetry of those flanks is also the sign check: a
compensator applying its correction the wrong way round would put the nominal setting at *twice*
the span, and the two sides would not match to three digits.

**Why this is a separate stage from the butterfly equaliser.** Both are linear filters, so one
adaptive filter could in principle do both jobs. It does not work. On the dual-polarization link
over 80 km, growing the butterfly from 7 taps to 65 — the longest the block allows, and nine times
the cost — leaves the link just as dead, because a blind modulus criterion has no gradient to
follow once the constellation is smeared into a Gaussian blob. The static block ahead of a 7-tap
filter restores back-to-back quality outright. Dispersion is static and *long*; polarization
mixing is fast and *short*; one filter serving both would have to be both, which is the worst of
each. That ordering is [asserted](tests/test_cd_compensation.py), not quoted.

### Dual polarization

`python examples/dualpol_link.py` puts two independent 16-QAM tributaries on orthogonal
polarizations of one wavelength — **256 Gb/s** — and rotates the state the way a fibre does:

```
rotation    without equaliser              with equaliser
    0 deg  EVM    2.5 /    2.5 %      0 err      EVM 2.50 / 2.57 %    0 err
   15 deg  EVM   28.1 /   29.2 %   1416 err      EVM 2.51 / 2.58 %    0 err
   30 deg  EVM  218.0 /  123.5 %   6217 err      EVM 2.52 / 2.51 %    0 err
   45 deg  EVM  278.5 /  365.2 %   6763 err      EVM 2.48 / 2.55 %    0 err
   72 deg  EVM   85.7 /  165.6 %   5523 err      EVM 2.54 / 2.56 %    0 err  (swapped)
   90 deg  EVM    2.5 /    2.5 %      0 err      EVM 2.54 / 2.48 %    0 err  (swapped)
```

Past a few degrees the unequalised branches are not degraded — they carry no recoverable data at
all, because each is a *mixture* of both tributaries. The
[butterfly equaliser](src/maiman/dsp.py) separates them blind, with no training sequence anywhere in
the link. Read the two end rows together: 90° is a clean swap rather than a mixture, so it needs no
equaliser at all and simply delivers the tributaries the other way round — which is also why
nothing blind can label them, and why a real link recovers the pairing from framing.

This is the increment that finally exercises `Ey`, which has been in the signal model since the
first commit for exactly this purpose.

### Shaping and differential encoding

Two transmitter refinements that close out the coherent phase.

**Root-raised-cosine shaping** bounds the spectrum. A held symbol has a sinc spectrum that never
ends — fine for one channel alone, useless once neighbours are packed onto a grid. At a 0.2
roll-off, **99.5%** of the shaped power falls inside ±0.6 symbol rates, against 83.6% for a held
symbol. The shaping is split into a root at each end, so the cascade is Nyquist (zero at every
symbol instant but its own) *and* the receiver's filter is matched to the transmitted pulse. One
end alone gives neither.

A shaped waveform overshoots between symbols, so its peak-to-average ratio is higher than the
constellation's and a full-swing drive clips — about 7% EVM at 16-QAM, falling to 1% backed off.
That is real, and backing off is what a transmitter does about it.

**Differential quadrant encoding** closes the quarter-turn ambiguity that every blind stage leaves
behind: the phase search cannot resolve it, and neither can the butterfly equaliser. Under plain
Gray labelling a quarter turn permutes the bits differently for every point; under a
quadrant-relative labelling it does exactly one thing, so differencing the quadrant makes it cancel.
A quarter turn that destroys **>75%** of absolutely-labelled symbols costs a differentially encoded
link nothing but its first symbol.

```python
result = sweep(graph, {("laser", "power"): [-24.0, -21.0, -18.0]}, runs=8)
q = result.metric(analyzer, lambda m: m.q_factor)     # shape (points, runs)
```

Repeats matter more than they look. At −20 dBm, eight runs of the same link give error counts of
37 to 58 — a 50% spread on the thing being measured, while Q itself is stable to ±1%. A single
BER at a marginal operating point is one sample, not an answer.

### Amplified and nonlinear

`python examples/amplified_link.py` runs a chain of 80 km spans, each amplified back to transparency:

```
  spans   reach     OSNR      vs. one span            Sech pulse over 4 soliton periods
      1     80 km   36.95 dB    +0.00  (theory -0.00)   configuration      width   peak
      2    160 km   33.94 dB    -3.01  (theory -3.01)   soliton (N = 1)    x1.00   x1.00
      4    320 km   30.93 dB    -6.02  (theory -6.02)   half the power     x2.62   x0.40
      8    640 km   27.92 dB    -9.03  (theory -9.03)   no nonlinearity    x4.12   x0.23
     16   1280 km   24.91 dB   -12.04  (theory -12.04)  no dispersion      x1.00   x1.00
```

The left table tracks `10·log10(N)` to a hundredth of a dB over sixteen spans, and nothing in the
model is written in those terms — the noise bins just accumulate. The single-span figure of
36.95 dB is `58 − 16 − 5`: the quantum floor, the span loss the amplifier has to make up, and its
noise figure.

The right table is the soliton. At N = 1 the chirp the Kerr effect imposes cancels the one
dispersion imposes and the pulse is unchanged after 29 km; halve the power and the balance breaks.
The last row is the honest caveat — self-phase modulation *alone* also preserves `|A(T)|`, so
shape invariance proves nothing by itself. It is invariance with both effects active that is the
result.

### What ASE beat noise is for

An amplifier's OSNR is only half an answer. A photodiode squares the field, so ASE arriving with
the signal *beats* against it rather than merely adding its power — and on any amplified link that
beat term is the noise floor. Without it this project could compute OSNR to a hundredth of a dB
over sixteen spans and then report a Q that had almost nothing to do with it:

```
8 x 80 km, each amplified to transparency, 10 Gb/s OOK

   NF       OSNR    Q before    Q now   Q from OSNR
  4 dB   25.96 dB     118.73    22.79         25.08
  8 dB   21.96 dB     113.72    14.50         15.59
 14 dB   15.96 dB      94.83     6.83          7.51
```

The middle column is the old model. Ten decibels of OSNR cost it **nothing at all** — Q moved from
118.7 to 94.8 while the link's optical margin collapsed — because the only thing ASE contributed
was mean power and its shot noise. The right-hand column is the textbook relation
`Q = 2·√(B_ref/B_e)·OSNR/(1+√(1+4·OSNR))`, checked first against its own known point: 14.5 dB must
give Q ≈ 6, which is the industry figure for 10 Gb/s at 1e-9.

The model now sits **just below** that limit at every point, which is the right side to be on — it
carries shot, thermal and finite-extinction effects the closed form omits.

Coherent detection has the same term, with ASE beating against the local oscillator instead, and
there the check is `SNR = 2·OSNR·B_ref/R_s`. The gap closes as ASE takes over, which is what makes
it a test of the beat term rather than of one operating point:

| noise figure | OSNR | SNR | optical limit | gap |
| ---: | ---: | ---: | ---: | ---: |
| 4 dB | 22.08 dB | 18.82 dB | 21.01 dB | 2.18 dB |
| 10 dB | 16.08 dB | 14.32 dB | 15.01 dB | 0.69 dB |
| 16 dB | 10.08 dB | 8.78 dB | 9.01 dB | **0.23 dB** |

Only ASE co-polarized with the signal beats with it, which is why a polarizer helps a receiver and
why a coherent front end needs no optical filter at all: it is filtered by its own electrical
bandwidth, so the ASE-ASE term that forces a direct-detection receiver to carry one is absent by
construction.

### Wavelength selection, and what a filter is really for

`python examples/wdm_demux.py` puts four channels on a 100 GHz grid through four amplified spans
and demultiplexes one. Because every band carries its own centre frequency, that is a real
wavelength-selective operation and not a choice of array index — a filter tuned *between* two
channels attenuates both.

The second job is the one that surprises people. An amplifier emits ASE across four terahertz and
every hertz of it reaches the photodiode and beats there:

| link | OSNR | ASE power | Q | vs OSNR limit |
| :--- | ---: | ---: | ---: | ---: |
| no demultiplexer | 29.93 dB | 0.3250 mW | 7.43 | 0.19× |
| 50 GHz demux | 26.92 dB | 0.0040 mW | 25.04 | **0.89×** |

Eighty times less ASE reaches the diode, and the demultiplexed link lands just under its own OSNR
limit — which is where a real receiver sits. **The OSNR figure barely moves**: it is quoted in a
fixed 12.5 GHz reference bandwidth, so it cannot see ASE removed outside that band. The cheapest
improvement available to a receiver is invisible to the number everyone quotes.

The filter's skirts are floored at a declared `extinction`, because a super-Gaussian's are not. A
third-order 50 GHz passband is `exp(-2838)` one channel spacing away — not a small number but
exactly zero in double precision, which would make rejection infinite and a chain of filters
accumulate no crosstalk at all. Real hardware specifies 30–50 dB and it is that floor, not the
shape, that decides what leaks through a long line of them.

The **[OSA](src/maiman/components/filters.py)** finally makes the signal model visible: bands and
noise bins rendered onto one grid, the way an instrument shows them. Its resolution bandwidth is
not cosmetic — widening it raises the ASE trace decibel for decibel and leaves a carrier exactly
where it is, which is the clearest demonstration of why OSNR needs a stated reference bandwidth.

**It sweeps full span by default, because an instrument you have to aim is no use for finding
something.** It used to start at 1550 nm with a 1000 GHz window and drop everything outside, so a
laser at 1560 nm produced an empty trace and the only way to see it was to already know the
wavelength and type it in here as well — which is backwards, since the wavelength is usually the
thing being measured. Measured, before and after:

```
laser at   fixed window (1550, 1000 GHz)   automatic
1550 nm    1.0000 mW, peak 1550.00         1.0000 mW, peak 1550.00
1560 nm    0.0000 mW  — nothing at all     1.0000 mW, peak 1560.00
1310 nm    0.0000 mW  — nothing at all     1.0000 mW, peak 1310.00
```

Automatic covers what the signal actually occupies — every band's sampled bandwidth and every noise
bin, plus 5 % margin — so two carriers forty nanometres apart are both in the trace rather than the
loudest one deciding. `auto_span` off restores the declared window exactly, which is what you want
once the channel is known: `points` is fixed either way, so span buys coverage and costs
resolution, and that trade is the reason both modes exist.

**A default changed here, and it changes numbers.** An OSA in an existing project now sweeps a
different window, so its trace is not the trace it was — the shipped WDM demo went from a 6.40 nm
slice to the whole 35.27 nm comb. Set `auto_span` to false to get the old one back.

## A circuit is not a chain

A fibre link is a chain: each block takes a waveform and returns one, and the scheduler runs them in
order. A photonic integrated circuit is not. Light in a ring goes round, comes back to the coupler it
entered by, and interferes with itself; a resonance *is* that feedback at steady state. There is no
order to run the blocks in, because every port's answer depends on every other port's at once.

So [`maiman/circuit.py`](src/maiman/circuit.py) does a different kind of solve. Every device is a
**scattering matrix** — for unit amplitude into port *j*, `s[i, j]` is what leaves port *i* — and a
circuit is which ports are wired to which. Split the ports into external and internal, write the
wires as a permutation `C` that hands each internal port's outgoing wave to its partner, and

```
a_int = C S_ii a_int + C S_ie a_ext        S = S_ee + S_ei (I - C S_ii)^-1 C S_ie
b_ext = S_ee a_ext + S_ei a_int
```

That is the whole solver. It is exact and not iterative: there is no convergence criterion to pick
and no number of round trips to truncate at. A resonance appears as `(I - C S_ii)^-1` growing large,
which is the same statement as the round trip approaching unit gain, reached in one step instead of
summed. The test suite checks it against the summation anyway, because that is a genuinely different
algorithm: **they agree to 1e-13 over three free spectral ranges.**

**The roadmap said to integrate a solver rather than write one, and that was worth checking before
believing.** For the *solver* it does not hold — the identity above is **twelve lines of numpy**,
inside a `solve` that is forty-six lines once the grid check, the port bookkeeping and the dropping
of dangling ports are counted. (This used to say "thirty lines", which was neither of those
numbers and was nobody's measurement.) What is genuinely large in that ecosystem is the PDK side,
and that is now joined from the other direction — see [the netlist
section](#what-a-layout-tool-knows-and-what-it-does-not). The other path was measured rather than
argued about: installing SAX resolves to **37 packages**, including jax
and a 66 MB jaxlib, plus matplotlib, pandas, scipy, sympy, xarray and pydantic — to perform one
`numpy.linalg.solve`. And `klujax`, its sparse back-end, is **LGPL-2.0-only**; this project already
refuses FFTW over exactly that question.

Refusing the dependency is not refusing the reference. SAX was installed in a scratch environment and
given the same two device models, the same wiring and the same grid:

| add-drop ring, 4001 frequencies over 2 THz | max &#124;maiman − sax&#124; |
| :--- | ---: |
| in → through | 7.2e-15 |
| in → drop | 3.2e-15 |

Thirty-three units in the last place of double precision, across a spectrum containing three
resonances. The comparison is not in CI — nothing that costs 70 MB and a licence review should be —
but it is the reason those twelve lines are defensible.

### What comes out of it

`python examples/microring_filter.py`. Three blocks
([`maiman/components/photonic.py`](src/maiman/components/photonic.py)) and two device models
([`maiman/photonics.py`](src/maiman/photonics.py)), and the ring is **assembled rather than written
down** — two couplers and two arcs, wired into a loop and handed to the solver. The closed forms from
Yariv and Bogaerts live in the tests, on the other side of the comparison, where they can disagree;
they agree to 1e-13.

A 100 µm silicon ring at 3 dB/cm, walked across one resonance:

| offset | critical, κ = 0.00688 | under-coupled, κ/4 | over-coupled, 4κ |
| ---: | ---: | ---: | ---: |
| ±20 GHz | −0.01 dB | −0.00 dB | −0.03 dB |
| ±1 GHz | −2.08 dB | −0.57 dB | −3.07 dB |
| 0 | **−134 dB** | −4.42 dB | −4.39 dB |

**Critical coupling is a knife edge and a false friend.** Match the coupling to the round-trip loss
and the field coupled back out of the ring cancels the field that stayed on the bus, exactly. Miss it
and the notch fills in — by the *same* amount on either side, because the depth
`|t − a|² / |1 − t a|²` is symmetric in *t* and *a*. Solving for the partner of a given coupling,
`t₂ = (2a − t₁(1 + a²)) / (1 + a² − 2 a t₁)`, produces pairs that are indistinguishable from the
through port alone: κ = 0.0199 and κ = 0.00237 both notch −6.22 dB.

**The free spectral range is set by the group index and the resonance position by the effective
one**, and in silicon those differ by a factor of 1.7. Measured against `c / (n_g L)` across four
resonances: 713.80 GHz against 713.79. Change `n_eff` and the resonances move without the spacing
changing; double `n_g` and the spacing halves.

### A ring is an ASE gate, and that needs its linewidth

An amplifier emits across terahertz; a ring's comb passes about one linewidth in every free spectral
range. Reading the response at the bin's centre would return whatever that one frequency landed on,
so the noise a ring passes is *integrated* — and how finely is not a free parameter.

A ring 116 MHz wide inside a 714 GHz period is a loaded Q of 1.7 million: good, and buildable.
Averaging its drop response over a 4 THz amplifier bin, against a converged 2.444e-4:

| how the mean was taken | result | |
| :--- | ---: | ---: |
| 4096 points across the whole bin | 4.81e-4 | 97 % high |
| 4096 points across one free spectral range | 3.13e-4 or 2.14e-4 | 28 % high, or 12 % low |
| points set from the linewidth | 2.444e-4 | converged |

The middle row is the one worth staring at. **Which way it is wrong depends on where the amplifier's
band happened to sit** — high when the window landed on the resonance, low when it landed on a real
EDFA's centre — which is what makes an under-resolved integral worse than a merely inaccurate one.
So `RingResonator` hands its own linewidth down to the integrator, and the ceiling that stops the
point count running away is stated in the source rather than discovered.

On an ordinary ring the numbers come out where they should: a 12.44 GHz linewidth in a 714 GHz
period passes **2.40 %** of a flat spectrum, against 2.74 % for a Lorentzian of the same width — the
14 % the shape approximation costs once integrated rather than merely evaluated at half depth.

**What that 2.40 % is, and what it is not.** It is the fraction of an amplifier's *total* ASE the
drop port passes, and it is right. For a long time it was also, silently, the density the ring left
at *every* frequency, because a noise bin was flat and could hold nothing but that average. So an
OSNR meter on resonance read the ASE beside the carrier as having dropped 16 dB, and a test asserted
the ring improved OSNR by more than twelve. It does not. This ring's 12.44 GHz linewidth *is* the
12.5 GHz reference band, so on resonance it passes most of the ASE inside that band along with the
signal:

| ASE beside the carrier, read as | kept | OSNR change |
| :--- | ---: | ---: |
| the flat average over one free spectral range (before) | 2.40 % | +15.6 dB |
| integrated over the reference band (now) | 68.8 % | **+1.06 dB** |

Noise bins now carry the shape of what they went through, normalised to the mean they already held,
so every total-power figure in this section is unchanged and a density read at one frequency is
finally the density at that frequency: about 91 % of the input at a tooth and a few hundredths of a
percent between two. A detector downstream of a ring or an interferometer now beats against the ASE
that is really beside its carrier rather than against an average spread over hundreds of gigahertz —
which, for a ring like this one, had been understating its signal-spontaneous beat noise by some
16 dB.

Straightened out, the same waveguide is a delay line, and the arithmetic is bleak: 1 mm of silicon
holds 14.01 ps and costs 0.2 dB; 10 cm holds 1.40 ns and costs 20 dB. Optical buffering is expensive
and this is why.

### Two modes in one waveguide

This section used to say that one response was applied to both polarizations, and that fixing it
needed "a polarization-resolved scattering matrix, which is a second index on every device". **The
second half was wrong, and it was wrong in a way worth recording**: it had never been measured. The
solver needed *no changes at all*.

`SMatrix` identifies ports by **name**. So a device carrying two modes is not a device with an
extra index — it is a device with twice as many ports, and `Circuit.solve` already solved those.
What was missing was one combinator:

```python
dual_polarization({"te": guide_te, "tm": guide_tm})   # -> ports in@te, out@te, in@tm, out@tm
```

which stacks one matrix per polarization block-diagonally, plus `link_polarizations` and
`expose_polarizations` so a wire is wired on both modes in one call — the fault they exist to
prevent being a circuit connected on TE and not on TM, which yields a perfectly plausible spectrum
on one polarization and silence on the other.

**A strip waveguide's two modes are two different waveguides.** For 500 × 220 nm silicon at
1550 nm:

```
              n_eff    n_group    ring FSR (100 µm)
TE             2.44       4.20         713.8 GHz
TM             1.78       3.80         788.9 GHz
```

27 % apart in phase index. So the ring resonates at **two sets of wavelengths that are nowhere near
each other, on two different free spectral ranges** — and both match `c/(n_g·L)` to a part in a
thousand, each against its own group index. Measured on a graph: at 1558.16 nm the `Ex` component
sits in a notch and `Ey` passes; at 1561.71 nm it is the other way round. A ring like this is a
polarization-selective filter, which is what a real one is.

`python examples/birefringent_ring.py` prints both tables — where each comb sits against
`c/(n_g·L)`, and then the same ring as a block with the light launched at 45° so both axes carry
something for it to treat differently. Using TE's group index for TM would put that row 75 GHz out,
a tenth of a free spectral range, and it would still look like a perfectly plausible ring.

**The flag is off by default**, declared as an idealisation exactly the way `saturate` is on the
EDFA. Turning `birefringent` on changes the numbers of any link with a photonic block in it; that
should be a decision someone makes, not a default that moves under an existing project. A test
asserts that off, the two field components take *exactly* the path they took before the flag
existed — `rel=1e-12`, not approximately.

**Block diagonal is a choice the models make, not a shape the matrix can hold.** A bend, a sidewall
that is not vertical, and a mode converter placed there on purpose all convert TE to TM. Nothing
here produces such a term, so `dual_polarization` takes a `cross` argument for one — because the
solver never needed the block-diagonal assumption, and the one place where quietly acquiring it
would be invisible is here.

**What is still shared: loss and dispersion.** A real strip has a different propagation loss for
TM — it overlaps the sidewalls less and the substrate more — and a different dispersion with it.
Those are per-process numbers and this library will not invent one, so the two indices are what is
offered and a test records the rest as shared rather than leaving it to be discovered. A fitted set
belongs in a PDK, where `maiman.pdk` already refuses to extrapolate past the window it was fitted
in. The same goes for the couplers: a directional coupler's split ratio is polarization-dependent
and here it is not, so the two combs come out with the same notch depth where a real pair would
not.

**Coupling into the die is not modelled either.** `Ex` is taken to be the chip's TE mode and `Ey`
its TM, which says the die is aligned to the signal's own x axis. A real launch goes through a
grating or an edge coupler with its own alignment and its own extinction.

The MMI, the Y-junction, the Mach-Zehnder and PDK import are all downstream of this framework
rather than of new physics — an interferometer is already two couplers and two arms in a `Circuit`,
and the tests build one.

## A channel plan, and the crosstalk it produces

The roadmap said "DWDM MUX/DEMUX with crosstalk" and Phase 3 shipped without one. That was not
quite a gap in the physics — `Combiner`'s docstring calls itself a WDM multiplexer and
`OpticalFilter`'s first line calls itself the demultiplexer, and both are telling the truth. What
was missing was the *grid*: building an eight-channel demux meant a splitter, eight filters, and
channel arithmetic each project did for itself.

### The two ITU grids are not the same kind of grid

**G.694.1 is uniform in frequency. G.694.2 is uniform in wavelength.** That is the whole difference,
and it has a consequence people trip over:

```
   frequency      wavelength        step
  192.9000THz    1554.1340nm
  193.0000THz    1553.3288nm     0.8053nm
  193.1000THz    1552.5244nm     0.8044nm
  193.2000THz    1551.7208nm     0.8036nm
  193.3000THz    1550.9180nm     0.8028nm
```

The dense grid is anchored at **193.1 THz**, which is 1552.5244 nm, and every channel at every
spacing is an exact multiple of the spacing from it — that is what makes a 50 GHz plan a *superset*
of a 100 GHz one rather than something offset by half a channel. But the wavelength steps are not
equal and cannot be, because `Δλ = λ²Δf/c`: 100 GHz is 0.7808 nm at 1530 and 0.8170 nm at 1565.
Stepping a flat 0.8 nm to build a "100 GHz grid" loses most of a channel across the C band.

The coarse grid does the opposite — eighteen channels, 1271 to 1611 nm, exactly 20 nm apart — which
comes out **56 % uneven in frequency**, 3.654 THz at the blue end against 2.339 at the red.

**And the 20 nm is a specification, not a round number.** A CWDM channel is meant for an *uncooled*
DFB, and an uncooled laser walks about 0.1 nm per kelvin: seventy kelvin of case temperature is
7 nm of drift before manufacturing spread. 20 nm spacing is what makes a transmitter with no
thermoelectric cooler, no wavelength locker and no control loop into a legal channel — which is why
CWDM optics cost a fraction of DWDM optics, and why the grid reaches out to 1611 nm where no
amplifier will help it.

`dwdm_frequencies` refuses a spacing the standard does not define and names the function that will
build one anyway. `cwdm_wavelengths` stops at eighteen: 1631 nm is not a CWDM channel, it is past
the end of the grid, and continuing the arithmetic would invent one.

### The blocks are routers, not splitters

`Multiplexer` and `Demultiplexer` sit on that grid. Eight channels out and back:

```
  port      channel       kept    worst neighbour
     0   1550.0000nm    -8.000dB            -40.00dB
     1   1549.1990nm    -8.000dB            -40.00dB
     …
     7   1544.4105nm    -8.000dB            -40.00dB
```

−8 dB is 4 dB through the mux and 4 through the demux, and **it does not grow with the channel
count** — at 2, 4, 8, 16 and 32 channels it is −8.000 dB every time. An arrayed-waveguide grating
*routes*: each channel leaves by its own port. A broadcast-and-select demux really would divide
power N ways and cost 10·log10(N) — 15 dB at thirty-two channels — and that device is a `Splitter`
in front of filters, which is exactly what `Splitter` is for.

Every demultiplexer port is **the same function** the single-filter block is. `apply_passband` was
lifted out of `OpticalFilter` for that reason and a test asserts the two agree sample for sample: a
multi-port device whose per-channel response drifted from the single-filter block's would be the
worst kind of disagreement, since both would look right alone and the crosstalk number would depend
on which one happened to be used.

### Which of two things sets the crosstalk

```
   spacing   passband    with floor    skirt alone
      200G        50G       -40.00dB          -infdB
      100G        50G       -40.00dB          -infdB
       50G        50G       -40.00dB       -192.66dB
       40G        50G       -40.00dB        -50.50dB
       25G        50G        -3.01dB         -3.01dB
```

Nothing in that table was declared. The right column is the super-Gaussian's own skirt, falling as
the sixth power of detuning at order 3 — and `-inf` there is not infinite rejection, it is the field
underflowing `complex64` past about −600 dB, which is a storage limit and not a physical one.

The left column is what a real system sees, and the point of it is that **the skirt is almost never
what matters**: at any sane plan it is already below the extinction floor, and it is the floor that
accumulates down a chain of these. At 25 GHz with a 50 GHz passband the neighbour is simply inside
the channel, and no floor is involved at all — which is a channel plan that does not work rather
than a model that does not.

`python examples/wdm_grid.py` prints all five tables.

## A mirror, and a graph that goes one way

Every optical block up to here is matched at both ends: light enters one port and leaves another,
and a dataflow edge is an arrow that says so. A fibre Bragg grating is the first one that is not.
Its useful output comes back out of the fibre it arrived on, and on a bench the only way to that
output is a circulator.

Two things had to be true for that to work, and one of them was not.

**The solver was already fine.** `circuit.py` has said since it was written that "a non-reciprocal
device (an isolator) and a reflecting one (a facet, a Bragg grating) both solve correctly" — the
reduction never transposes anything and never drops a term. Until now nothing outside a test made
it prove it. `FiberBraggGrating` puts its reflection on the diagonal of a 2×2 matrix and
`Circulator` is a cyclic permutation that is deliberately not symmetric, and both go through
unchanged.

**The scheduler was not.** The canonical drop is

```
signal → circ.in1 ; circ.out2 → fbg.in ; fbg.reflected → circ.in2 ; circ.out3 → receiver
```

which reads as `circ → fbg → circ` and was refused as a feedback loop. **It is not one.** At the
level of ports, `out3` is a function of `in2` alone and `out2` of `in1` alone; no light in that
picture travels backwards in time. The cycle was an artefact of scheduling whole components.

So a component may now declare that its ports fall into independent `PortGroup`s, and the
scheduler makes a node of each. The default is one group holding everything, which is the truthful
answer for every block that computes all its outputs from all its inputs in one call — that is,
every other block in the library, and the sort is bit-for-bit what it was. A real loop is still a
real loop: two attenuators wired to each other are refused, and so is a circulator whose three hops
are wired into a ring, because splitting a component into independent nodes cannot launder a cycle
that runs through all of them.

The cost of getting that claim wrong is a port that never runs or one that runs twice, so the
groups are checked to be a partition before anything executes rather than discovered mid-run.

### The grating is assembled, not written down

Two hundred per-section transfer matrices, multiplied. Erdogan's closed form for a uniform grating
appears nowhere in the model — it is in `tests/test_grating.py`, on the other side of the
comparison, and they agree to 2×10⁻¹¹ across two decades of grating strength. That is what buys
chirp and apodization, neither of which has a closed form:

```
        δn    κL    tanh²(κL)      model     max error
     1e-05  0.203    0.039981   0.039981      2.3e-13
     5e-05  1.013    0.588552   0.588552      3.4e-12
     1e-04  2.027    0.932915   0.932915      7.9e-12
     5e-04 10.134    1.000000   1.000000      2.3e-11
```

The first-null bandwidth lands on `λ²/(π n_eff L)·√((κL)²+π²)` to a per-mille, which is the
resolution of the measurement rather than of the model, and `R + T = 1` everywhere to 10⁻¹⁰ — a
transfer-matrix product that drifts off unity is the classic way this method fails, and it fails
quietly.

### Apodization is not mainly about sidelobes

The textbook reason to shape a grating's strength is crosstalk. The reason that decides whether a
*compensator* is usable is group-delay ripple, and it is an order of magnitude bigger:

```
       profile     peak R   sidelobe   GD ripple
       uniform     0.9329     -7.6 dB     101 ps
 raised-cosine     0.5886    -31.4 dB     6.5 ps
      gaussian     0.5823    -41.4 dB     8.1 ps
```

101 ps is a whole bit at 10 Gb/s. The peak comes down in the same move, because shaping the
coupling lowers its average — there is no free version of this.

### A compensator that needs no receiver

`DispersionCompensator` is DSP. It inverts the fibre's all-pass filter, which means it needs the
field, which means it needs coherent detection; a direct-detection receiver squares the field at
the photodiode and can never do it at all. A chirped grating is glass, and it works in front of a
photodiode.

10 cm chirped across 0.71 nm is −1360 ps/nm — 80 km of standard fibre — in a part the length of a
finger. A 30 ps Gaussian pulse, rms width, no amplifier:

```
     span       bare    + grating
      0 km    21.21 ps    44.81 ps
     40 km    29.46 ps    28.52 ps
     80 km    46.06 ps    21.34 ps
    120 km    64.89 ps    30.54 ps
```

Back to the launched width at the span it is matched to, over-compensated below it and
under-compensated above — which is what a fixed compensator does.

### The circulator charges twice, and says so

Its loss is quoted per hop because that is how a datasheet quotes it, and light reaching a
reflective device pays it going out and coming back. Dropping 1550 nm out of a pair 2 nm apart:

```
                1550 nm     1552 nm    accounting
       drop     -1.702 dB  -47.359 dB  two hops (-1.4) + R (-0.30)
    express    -12.434 dB   -0.700 dB  one hop (-0.7) + T (-11.73)
```

Every figure is arithmetic. With an ideal circulator the neighbour's 45 dB of rejection on the drop
port is **the grating's own sidelobe floor two nanometres out**, not an isolation figure anybody
declared — apodize the grating and the number moves. With a real circulator it is not the grating at
all; see below.

**A real circulator leaks, and this section used to get that wrong.** It said isolation could not be
a parameter because in this configuration the backward leak is a *loop* — light returning to the
grating, reflecting again, coming round once more — and a cavity the engine was right to refuse. It
is not a loop. Reflected light coming back into port 2 leaves at port 3, or back out of port 1
toward the source, and neither returns to the grating. `Circuit.solve`, which sums every bounce there
is, gives a drop port equal to the feed-forward `τ·r·τ + ι` with a difference of exactly zero.

What the leak really broke was `PortGroup`. Port 3 now hears port 2 *and* the direct leak from port
1, while port 2 hears port 1 alone: dependencies that overlap and contain no cycle, which a partition
of ports could not express. Outputs are still a partition; an input may now be read by more than one
group, and `Circulator` has an `isolation` parameter. The engine matches the exact solve to a few
micro-decibels at every setting:

```
    isolation     1550 nm      1552 nm    neighbour rejection
        ideal    -1.702 dB   -47.359 dB        45.7 dB
        60 dB    -1.702 dB   -46.781 dB        45.1 dB
        40 dB    -1.701 dB   -38.713 dB        37.0 dB
```

At 40 dB the floor on the drop port is the circulator's leak from port 1, not the grating — which is
the number a real drop would be specified by. **Return loss is what would be a cavity**: a reflection
back out of the port light entered by, bouncing into the grating again. At 40 dB of it the exact solve
leaves feed-forward by 8e-3 and ripples the drop port by about ±0.07 dB. That one is a real loop, and it has
its own section.

### A loop the engine will run

The scheduler orders a graph, and a cycle cannot be ordered — which for almost everything here is the
truth: a link goes one way, and a cycle in its graph is a wiring mistake. Two things that looked like
cycles turned out not to be, and `PortGroup` handles both without any loop control. Return loss is the
one that is. Light reflected back out of port 2 goes into the grating again, which reflects it again,
and the steady state is the geometric series `τ·r·τ / (1 − ρ·r) + ι` that nothing feed-forward can
sum.

`Feedback` is the explicit, declared way to run it. Put it on any one wire of the loop: its `out` feeds
the loop and its `in` takes what comes back, so the scheduler sees a source and a sink and the graph
has an order again. The whole graph then runs `passes` times. On the first pass `out` is dark; on every
later one it is whatever reached `in` the pass before. Without it, the same wiring is refused with a
`CycleError` that now names the fix — and a graph with no `Feedback` in it runs exactly once, as it
always did.

Against `Circuit.solve`, which sums every bounce exactly — 20 dB of return loss, 40 dB of isolation:

```
   passes   drop, 1550 nm    residual
        1      -40.000 dB         inf   nothing has been round yet: the leak alone
        2       -1.701 dB     9.6e-02   one trip to the grating: the feed-forward sum
        3       -1.671 dB     9.4e-03
        4       -1.752 dB     9.0e-04
        5       -1.752 dB     8.7e-05
        6       -1.751 dB     8.4e-06   the exact cavity, to 4e-7 dB
```

**The residual falls by 0.0966 every pass, and that number is the physics:** the echo's amplitude 0.1
times the grating's reflection 0.966, the loop's round-trip gain. The power does not fall so tidily —
three passes are further off than two, and five than four — because partial sums of a series whose
ratio carries a phase overshoot and undershoot the limit in turn. Watch the residual, not the power.

**What a pass is not.** It is not a lap in time. Every signal is a whole window in its own retarded
frame, so a trip round the loop adds no delay and every lap lands on top of the last. That is right for
a cavity short against the window — a reflection between parts centimetres apart — and wrong for a
recirculating loop a pulse goes round many times, where the laps should arrive one after another. That
needs a delay line, and this is not one.

### The named windows were never the limit

The section loop has always eaten two arrays — a coupling per section and a local Bragg wavelength
per section — and the three named windows and the linear chirp were only ever two ways of filling
them. Handing the arrays over directly costs the loop nothing and turns a grating *model* into
something closer to a grating *design* model. A named window and its array are checked to be the
same device bit for bit, not approximately, so the convenient path and the general one cannot
become two models with one of them tested.

**A sampled grating is one array.** Write the grating and erase it periodically — in practice,
expose it through an amplitude mask — and the single peak becomes a comb. It is the tuning element
of a sampled-grating DBR laser, a multi-channel dispersion compensator, an interrogator that reads
a whole array of sensors at once. Ten sampling periods over 20 mm:

```
  7 peaks, spacing                       0.41478 nm
  predicted λ²/(2·n_eff·Λs)              0.41494 nm
```

Four digits, against the same Fabry–Perot arithmetic as any other cavity of that length — because
the sampling period *is* the cavity. The model has nothing in it that knows about combs. It has a
coupling that is switched on and off.

**A phase-shifted grating is one more.** Break the periodicity once and the stop band acquires a
transmission window in the middle of it: two mirrors facing each other, which is a cavity. At π,
dead centre, on a 2 cm grating:

```
  window at            1550.000000 nm
  width                0.70 pm  =  87 MHz
  unbroken, the same grating transmits 1.8e-4 there
```

87 MHz is the narrowest feature anything in this library produces. It is the distributed-feedback
laser's cavity and the filter a laser locks itself to. Moving the break off centre makes the two
halves unequal mirrors and the resonance dies — 1.000, 0.705, 0.099 at positions 0.5, 0.45 and
0.35 — which is a real design sensitivity rather than a modelling artefact.

The phase rides on the **coupling** and not on the detuning, and that is not a detail. It means the
two off-diagonal terms take it with opposite signs, so a section stays unimodular and the device
stays lossless; and it means a phase constant along the whole length does nothing at all, which is
the physically right answer, since where a grating's fringes start is not observable. A phase on
the detuning instead would shift the entire spectrum and look perfectly plausible doing it.

**Two descriptions of one thing are refused rather than resolved.** `coupling_profile` alongside a
named `apodization`, or `bragg_profile` alongside `chirp`, is an error that names the array the
other one stands for. Profiles of different lengths are an error. A component is the exception and
for a stated reason: a project file and an inspector carry numbers and not arrays, so
`FiberBraggGrating` declares `sampled` and `phase_shifted` as flags that *build* the arrays — and
when a grating is both sampled and apodized it multiplies them, because a mask over an apodized
exposure is what the writing process actually does.

Sampling also raises the section count on its own. What has to be resolved is the mask rather than
the envelope, so the floor is twenty sections per sampling period — measured: the comb spacing is
unchanged from ten sections per period all the way to four hundred. Deriving that rather than
refusing a number it could compute itself is the difference between a component and a form.

### The same device is a strain gauge and a thermometer

A grating's period is a length. Lengths respond to being stretched and to being warmed, and the
reflection reports it — so nothing has to be added to the fibre, there is nothing electrical near
the measurement, and because the reading is a *wavelength* it does not drift when the light gets
dimmer. That is why these go into boreholes, composite spars and bridge decks.

The whole of it is one line, and both coefficients are a material number times λ_B:

```
Δλ/λ = (1 − p_e)·ε + (α + ξ)·ΔT
```

```
  strain        1.209 pm per microstrain     p_e = 0.22
  temperature  11.191 pm per kelvin          α + ξ = 7.22e-6 /K
```

Stretching lengthens the period *and* lowers the index, and the two fight: the index term cancels
22 % of the geometric one. Warming does two things too, and they are not the same size — thermal
expansion is 0.55e-6/K and the thermo-optic coefficient 6.67e-6/K, so this is a thermometer made of
glass rather than one made of geometry. It is also why **coating changes the thermal number a lot**
and the strain number not at all: a jacket adds its expansion to α and nothing to ξ.

**One grating produces one wavelength and there are two unknowns behind it.** Divide one
sensitivity by the other:

```
  9.26 microstrain per kelvin

  +10 K, no load        -> 1550.111910 nm
  92.6 ustrain, no heat -> 1550.111953 nm
  difference             43.4 fm
```

Nothing about the spectrum hints that an interrogator should try to tell those apart. Every real
installation is arranged around this.

**And a second grating at another wavelength does not fix it**, which is the trap worth seeing
measured. Both sensitivities scale with λ, so their ratio does not:

```
   lambda_B   pm/ustrain      pm/K     ratio
   1530.0nm       1.1934   11.0466     9.256
   1550.0nm       1.2090   11.1910     9.256
   1570.0nm       1.2246   11.3354     9.256

   condition number of the 1530/1570 pair: 4.6e+16
```

Two rows differing only by a scale factor invert into noise. What does work is a grating that
carries no load — loose in its tube beside the working one, measuring the temperature alone — which
is a genuinely independent second equation rather than a nearly dependent one.

### Interrogation is a sweep, and it recovers what was applied

The array is one fibre: the probe goes out through a circulator and the gratings sit in series,
each passing what it does not reflect. Stepping a tunable laser across each peak and taking the
centroid of the reflected power is what a real interrogator does, and 121 points over 0.6 nm — a
5 pm grid — recovers a shift to a fraction of a picometre:

```
     grating      nominal      measured       shift
     working   1545.0000nm   1545.64936nm     649.36pm
   reference   1555.0000nm   1555.16841nm     168.41pm

   recovered temperature    15.000 K    (error +0.0000)
   recovered strain        400.000 ue   (error +0.0001)
```

And the same array read as though the temperature were known to be zero, which is what an
uncompensated gauge assumes:

```
              applied   strain read       error
   400 ue and + 0.0 K      400.00ue      +0.00ue
   400 ue and + 1.0 K      409.26ue      +9.26ue
   400 ue and + 5.0 K      446.28ue     +46.28ue
   400 ue and +15.0 K      538.85ue    +138.85ue
```

9.26 microstrain per kelvin, arriving exactly on schedule.

**Why a sweep, and why an analyser now works too.** A tunable laser is the more precise
interrogator, and every sweep point is a real run of a real graph. The other standard instrument — a
white-light source and an OSA — could not be built here when this section was first written:
broadband light was a `NoiseBin` whose density was *flat* by construction, a response reached it as
one averaged number, and the reflection of ASE off a grating came back with no peak in it at all.

Noise bins now carry the shape of what they pass through (see the ring section above), so it can.
The reflection of flat amplifier ASE off a 10 mm grating draws its peak on an analyser at the Bragg
wavelength and at exactly the input density times `tanh²(κL)`; a nanometre away it is 25 dB down;
reflected and transmitted power add back to the input to nine digits; and a thousand microstrain
moves the drawn peak by 1.209 nm, the same shift the sweep reads.

One more thing the array turned up, and the engine was right about it. Wiring every grating's
reflection back onto one return fibre is what the hardware does, and `Combiner` **refuses** it: a
single-wavelength probe means every grating is reflecting a copy of the *same* band, and co-located
carriers have to be added as fields on a common grid rather than multiplexed. So each grating is
metered where it sits, which is the quantity being measured anyway.

`python examples/fbg_sensor.py` prints all five tables.

### And a sign that was wrong, in the fibre

The grating's own physics says a positive chirp puts the short wavelengths at the near end, so they
turn round first and the long ones arrive later: `dτ/dλ > 0`, which is `D > 0`, the sign standard
fibre has. A compensator is therefore the **negative** one, which is what the table above uses.

It did not use to be. For a while this device and `Fiber` disagreed about the sign, because
`kernels.propagate_dispersion` carried the opposite quadratic sign from
`photonics.propagation_constant` — something the latter's docstring had said in as many words since
long before the grating existed, with nothing ever putting the two in one graph to argue about it.

**The grating was that thing, and the kernel was the one that was wrong.** Its β₂ term was
Agrawal's `exp(+iβ₂ω²z/2)`, correct under the `exp(−iωt)` convention that book uses and wrong under
the `exp(+iωt)` that `numpy.fft.ifft` gives this one. A β₂-only model cannot feel the difference —
ω appears squared, so every width came out right — which is why nothing caught it for so long. The
β₃ term was never affected, because it had been *derived* from the expansion of β(ω) rather than
transcribed, and the docstring says so.

What could feel it was anything built on the other convention. Inside `propagate_coupled_ssfm` the
walk-off term and the β₃ term already followed `exp(−iβz)` and the β₂ term beside them did not, so
a WDM simulation slid its channels one way and dispersed each of them the other. Measured directly,
over 20 km at D = +17 ps/nm/km:

```
  a component 200 GHz above the carrier arrived   1089.89 ps LATE
  D · Δλ · L says it should arrive               1089.89 ps EARLY
  walkoff_from_dispersion agrees with physics
```

Correcting it moved three more signs that had been matched to the old one, and all three are
conventions rather than physics: the Kerr rotation in `propagate_coupled_ssfm` — which has to flip
with β₂ or the soliton stops balancing — `GaussianPulse`'s chirp parameter, so that `C > 0` still
means an up-chirp, and the trial phase the blind dispersion search builds in
`dsp.clock_tone_strength`. The flagship coherent link's EVM is unchanged at 7.39 %, which is the
right outcome: dispersion and its compensation both flipped, so nothing about a link's performance
moved. What moved is which way things point.

The test that caught it had been written as a *pin* rather than a claim, precisely so that
reconciling the kernel would fail and name the compensator as one of the things that moved with it.
It did exactly that.

`python examples/fbg_circulator.py` prints all seven tables.

## What a layout tool knows, and what it does not

A `.pdk` here carries a foundry's *fitted numbers*. gdsfactory draws the mask. Joining them is the
last thing the roadmap had open, and doing it started with two measurements — because this project
has twice now found that a stated obstacle was never measured.

**The first measurement was the licence.** gdsfactory itself is MIT. Its dependency tree is not:

```
gdsfactory  →  kfactory  →  klayout          GPL-3.0-or-later
86 packages resolved in total
```

That is the exact ground this project already refuses FFTW (GPL-2.0-or-later) and klujax
(LGPL-2.0-only) on, and `pyproject.toml` says licences are checked before adoption rather than
after. So gdsfactory is **not a dependency, not even an optional one** — nothing here imports it.

**The second measurement changed what the join could be.** A gdsfactory `CrossSection` carries
`sections` (width, offset, layer), `radius`, `radius_min`, `bbox_layers` — and `extra="forbid"`, so
a kit author cannot even attach anything else. There is no effective index, no group index, no
propagation loss and no dispersion anywhere in the package; the only `neff` in it computes grating
tooth pitch. **A layout tool does not know what any of it does optically**, so importing a PDK
*from* gdsfactory cannot produce the numbers a `.pdk` is made of. They were never there.

So the join runs the other way round, and it is the natural division of labour:

```
netlist (gdsfactory)  →  topology + geometry: what is wired, how long the router made it
.pdk       (maiman)   →  which model each cell is, and what this process measures
Circuit.solve()       →  the answer
```

An instance says it is a `straight` drawn in cross-section `strip`; the kit says `straight` is a
`Waveguide` in this process, that its `o1` is that model's `in`, and that `strip` is 2.44 / 4.20 /
2 dB/cm. **That mapping lives in the file and not in this library** — another process draws the
same geometry under another name, and a table of cell names compiled into a simulator would be
wrong for every kit but one.

`python examples/netlist_circuit.py` solves a netlist **gdsfactory actually emitted**, shipped
unmodified in `tests/data/` with its MIT attribution:

```
instances                       cell         length
bend_euler_R10_A90_P0p5_2f1…    bend_euler   16.637 um
straight_L10_N2_CSstrip_5000…   straight     10 um

transmission              -0.0053 dB
26.637 um at 2 dB/cm      -0.0053 dB
```

**The bend's 16.637 µm is its arc length**, and the radius alone does not give it — you need the
Euler parameter too. The kit asks for it with `from_netlist: {length: info.length}`, so every
routed bend and connecting straight carries the length the router gave *that instance* rather than
one nominal value standing in for all of them.

The second circuit in that example is a racetrack ring, hand-written in the same schema because
gdsfactory's shipped samples contain no ring and no coupler, and a circuit without feedback does
not exercise the one thing a scattering-matrix reduction is for. Its free spectral range comes back
at **713.74 GHz against `c/(n_g·L)` = 713.79** for the 100.000 µm its five instances sum to.

**It refuses rather than skips.** A cell the kit does not model, a layout port the kit does not
map, a port wired twice, a circuit with nothing facing outward — each is an error naming the thing.
A reader that quietly dropped what it did not recognise would hand back a circuit that solves,
plots like a spectrum, and is not the circuit on the mask, which is the worst of the three
outcomes.

**YAML is not a dependency either.** The reader works over a parsed mapping and handles JSON with
the standard library; YAML is read when a YAML parser happens to be installed, and the error names
the one-liner that converts a file when it is not. `pyyaml` is in the `dev` extra so CI exercises
that path, and the shipped package still depends on numpy and nothing else.

**What is still missing.** Routing. A schematic's `routes:` section names links between ports, and
the bends and straights that implement them do not exist until the route is built — so a schematic
read at this level is the circuit without its interconnect, and its lengths are missing. Read a
netlist extracted from the built layout, which is what the shipped sample is. And bend loss: a
bend is modelled as a straight waveguide of the same arc length, which a tight bend in a real
process beats by some margin the kit would have to carry.

## The kernels do not know what they are running on

The propagation kernels are the only part of this worth a GPU: a loop over FFTs on a long array,
where everything else is scalar arithmetic or a closed form evaluated a few dozen times.
[`maiman/kernels.py`](src/maiman/kernels.py) was written as array-to-array functions from the
beginning so that this could be added without touching anything above it, and
[`maiman/backend.py`](src/maiman/backend.py) is the whole of the addition.

**The arrays decide, not a setting.** A kernel handed CuPy arrays runs on CuPy and returns CuPy
arrays; handed NumPy arrays it runs on NumPy. There is no global mode and no flag on the context,
which matters because the kernels are pure functions and a hidden mode would be the one piece of
state that could make the same inputs give different answers. Dispatch is on the array's own type —
`type(a).__module__` names the package it came from — so there is no registry to keep in sync.

**CuPy is not exercised here**, and saying otherwise would be the kind of claim this project exists
to avoid: there is no device and no install in CI. What *is* tested is the half that would actually
break a port. A second array library — [`tests/hostile_backend.py`](tests/hostile_backend.py) — sets
`__array_function__` to `None`, which is NumPy's own way for a type to say it is not NumPy's, so
`np.fft.fft` on one of its arrays raises. Universal functions are deliberately left working, because
`np.exp` on a CuPy array dispatches and comes back a CuPy array; refusing them would be testing a
rule that is not true. What breaks a port is anything that *allocates* — `np.fft.fftfreq` and
`np.zeros` build on the host, and a kernel calling one inside its loop pays a transfer every step.

Every kernel is then run on both libraries and the results compared, and the names the second one
was asked for are recorded. That set is the contract, and it is asserted as an equality rather than
a lower bound:

    abs  complex128  conj  exp  float64  max  pi
    fft.fft  fft.fftfreq  fft.ifft  fft.irfft  fft.rfft  fft.rfftfreq

Thirteen names. A change that reaches for something only NumPy has fails in this repository rather
than on somebody's GPU, and one that stops needing something fails too. CuPy provides all thirteen.

Only the propagation path is converted. The four-wave-mixing closed forms and the 2×2 Jones algebra
are scalar work a device would slow down, and there is a test naming which functions are in and
which are out, so the line is a decision rather than an oversight.

## 400G and 800G, and what they cost

Three reference transceivers, all dual-polarization coherent, all derived from one number and
arithmetic. 400G is DP-16QAM at 59.84 GBd — the shape a 400ZR module has. 800G is that payload
doubled, and there are two ways to double it:

| configuration | GBd | line rate | payload | slot | b/s/Hz |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 400G DP-16QAM | 59.84 | 479 Gb/s | 400 G | 75 GHz | 5.33 |
| 800G DP-16QAM | 119.68 | 957 Gb/s | 800 G | 150 GHz | 5.33 |
| 800G DP-64QAM | 79.79 | 957 Gb/s | 800 G | 100 GHz | 8.00 |

Line rate is baud × bits × 2 polarizations; a 400 Gb/s payload inside 479 leaves 16.4 % for forward
error correction and framing. **Nothing here is quoted from a standard** — the 800G symbol rates
follow from the 400G one, and the required OSNR below is *measured*: a noise-loaded link, bisected
until the **counted** bit error rate lands on the threshold, then compared against a relation that
knows nothing about any of it.

| configuration | closed form | ideal DSP | penalty | blind equaliser |
| :--- | ---: | ---: | ---: | ---: |
| 400G DP-16QAM | 19.47 dB | 19.78 dB | +0.31 | 20.24 dB |
| 800G DP-16QAM | 22.48 dB | 22.86 dB | +0.38 | 23.32 dB |
| 800G DP-64QAM | 26.41 dB | 27.22 dB | +0.81 | 28.56 dB |

The closed form assumes a perfect transmitter, perfect DSP and a noiseless receiver, so the fourth
column is the transmitter's implementation penalty — growing with the format order, because a
denser constellation is less forgiving of the modulator's curvature.

**The fifth column is what the DSP costs on top**, and it used to be a finding. Nothing rotates the
polarization on this bench, so a perfect separator would be the identity and the blind butterfly
equaliser should be free. At 64-QAM it cost eleven decibels.

Chasing that down: it is not the step and not the decision — freezing the radius-directed stage
entirely still left 3249 symbol errors, which identified the *first* stage. A polarization rotation
is **memoryless**: a 2×2 complex matrix, four numbers. Fitting it with seven taps per path means
twenty-eight, and the twenty-four that are not needed fill with gradient noise, which the
single-radius cost has no per-sample truth to hold down. At 64-QAM that noise is the whole margin;
a single tap had already been measured to leave a residual of 0.0013 where seven left 0.29.

Adapting only the centre tap while that stage runs takes 64-QAM through a rotated channel from 3568
symbol errors to **zero at every angle** — and breaks the other case, where the eye is closed by
*memory* rather than by rotation and every tap has to move: at 60 ps/nm of residual dispersion it
costs 16-QAM 1955 errors where letting them all run costs 44. Neither wins outright, and which one a
channel needs is not knowable in advance, so the block runs both and keeps whichever lands closer to
the constellation's rings. That is not a new criterion — it is the statistic the second stage already
steers by, read as a score. Over ten cases it picks right in all of them, and 16-QAM at fifteen taps
improved from 544 errors to 53 as well.

Five other approaches were measured and moved nothing: gating the second stage's decision on whether
the ring it picked was credible, normalising the update by the window energy, annealing the second
stage's step, leaking the weights towards zero, and shortening the first stage.

With the DSP out of the way, the 800G choice is two numbers:

    twice the baud, same format    +3.08 dB of OSNR, twice the spectrum
    denser format, less baud       +4.36 dB of OSNR, two thirds of it

The first is the price of bandwidth and is 3 dB in theory: twice the symbol rate collects twice the
noise and nothing else changes. The second buys a third of the spectrum back, and is what a link
with filled fibre and optical SNR to spare pays for it.

`maiman.analysis` carries the bridge these rest on — `snr_from_osnr`, `snr_for_ber` and
`required_osnr` — and the two reference designs ship as
[`examples/zr400.maiman`](examples/zr400.maiman) and
[`examples/zr800.maiman`](examples/zr800.maiman), laid out and openable in the studio.
[`examples/reference_rates.py`](examples/reference_rates.py) builds them and prints the tables.

## Mixing products add in field, not in power

A link is not one fibre. The four-wave mixing product a span generates arrives on top of the one
the span before it generated, and whether those add or cancel is decided by how far the pumps and
the product have drifted apart over the fibre already behind them. That drift is the phase mismatch
integrated over the distance travelled, and it used to be thrown away: every span drew its products
a fresh random phase, so they added in power and a four-span link came out four times one span
instead of sixteen.

Three carriers at −30 dBm, four 80 km spans against one, each span's loss exactly undone:

| span | D = 0 | × one span | D = 17 | × one span |
| ---: | ---: | ---: | ---: | ---: |
| 78 km | −104.06 dBm | 16.00 | −180.98 dBm | 0.03 |
| 80 km | −104.04 dBm | 16.00 | −154.84 dBm | 13.17 |
| 82 km | −104.02 dBm | 16.00 | −166.46 dBm | 0.89 |
| 85 km | −103.99 dBm | 16.00 | −160.89 dBm | 3.12 |

**The sharp form of the claim is that cutting a span in half must change nothing.** A span boundary
is a bookkeeping decision, not a physical one, so 320 km of lossless fibre must give the same
product whether it is run as one block or as eight — and it now does, to under a hundredth of a
decibel, at every dispersion. Adding the pieces in power put eight of them 9 dB below one.

Note what is *not* claimed: that dispersion suppresses the build-up. At the right span length it
does the opposite — the rotation per span comes back round to a multiple of 2π and the spans stack
again, which is the 13.17 in the table. That periodic re-phasing is a real property of a
dispersion-managed link and is why the map is designed rather than chosen; a model that reported
"less" would be reporting something false.

**One number carries all of it.** The mismatch is a difference of four propagation constants at
frequencies satisfying `ω_i + ω_j = ω_k + ω_F`, so the constant and group-delay terms cancel
identically and only the β₂ term survives. The signal therefore carries a single accumulated
`Σ β₂·L` and nothing about any band's absolute phase — which is fortunate, because `β₀·L` is of
order 10¹¹ radians over 80 km and reducing that modulo 2π in double precision would leave about
five digits of the answer. A dispersion-compensating span is a fibre with negative D, so it
subtracts from that sum on its own; nothing special is done for it.

What remains drawn is one phase per *triplet*, standing in for the pump phase combination that
modelling the pumps by their powers has thrown away. It is keyed on the three pump frequencies and
on nothing else — not on the block, not on which span it is — because the same three pumps make the
same product wherever they are, and a phase redrawn per block is precisely what made the spans
average instead of add. What is still missing is the pumps' own nonlinear phase: only the linear
mismatch is tracked between spans.

## The short wavelengths pump the long ones

A photon can scatter off a silica vibration and come out at a lower frequency, and the process is
stimulated — light already there at the lower frequency makes it more likely. So in a comb the
short-wavelength channels pump the long-wavelength ones, and a flat launch does not arrive flat.
Set `raman_gain_slope` on the fibre. One 80 km span of standard fibre, everything launched at
0 dBm:

| comb | total | span | tilt |
| :--- | ---: | ---: | ---: |
| 4 × 100 GHz | 6.0 dBm | 0.30 THz | +0.00 dB |
| 40 × 100 GHz | 16.0 dBm | 3.90 THz | +0.40 dB |
| 80 × 50 GHz | 19.0 dBm | 3.95 THz | +0.81 dB |
| 80 × 100 GHz | 19.0 dBm | 7.90 THz | +1.63 dB |

A filled C band loses most of a decibel across itself every span, and it accumulates. That is a
large fraction of the margin a link is designed with, and it is why a line system is built with a
tilt to undo rather than assumed flat. A four-channel comb — which is what every other WDM number
here is measured on — moves by three thousandths of a decibel, which is why this was never missed.

**Power is moved, not lost.** The tilt is a closed form (Zirngibl, *Electron. Lett.* 34(8), 1998)
and the sum over channels is unchanged to floating point; the quantum defect the lattice keeps is a
part in ten thousand at these separations and is not modelled. That conservation is also what makes
a sign error impossible to hide — the two ends have to move in opposite directions or the sum
cannot come out.

The gain is taken as rising linearly with separation, which it does up to about 13 THz and not past
it, so a comb spanning the C and L bands together has its far pairs over the peak and the transfer
between them over-predicted. The `diagnostics` port reports the tilt in dB, so what happened is a
number rather than an assumption.

## An orthogonal neighbour is not an absent one

The Kerr coupling used to be scalar per polarization: a channel was modulated by its neighbours'
co-polarized power and by nothing else, so a channel polarized across it counted for zero. In an
isotropic medium orthogonal power counts for exactly two thirds — which, since co-polarized
cross-phase modulation already carries its factor of two, makes orthogonal cross-phase modulation
exactly **one third** of co-polarized. Set `cross_polarization` on the fibre:

| neighbour | axes uncoupled | axes coupled |
| :--- | ---: | ---: |
| co-polarized | 2.000 γPL | 2.000 γPL |
| orthogonal | 0.000 γPL | 0.667 γPL |

The same coefficient does two more things, because it is the same coefficient. A channel's own
orthogonal component modulates it at two thirds the rate, so power split evenly between the axes
turns each of them by `(0.5 + ⅔·0.5)·γPL` instead of `0.5·γPL`. And the two axes, no longer
accumulating the same phase, rotate the state of polarization as the power moves — nonlinear
birefringence, which falls by a factor of three, and cross-polarization modulation, which is that
rotation being driven by a *different* channel's power.

The value is the fixed-axis one, from the χ⁽³⁾ tensor rather than from any averaging. A fibre whose
birefringence scrambles faster than the nonlinearity acts is the Manakov regime instead, where the
distinction washes into a single 8/9 on the total power; this block applies PMD as a separate
element rather than interleaving it, so the fixed-axis form is the one consistent with the rest of
it. The coherent `A_x* A_y²` term, which would exchange power between the axes rather than only
dephase them, is left out — it is the part that averages away first — and the tests assert the
axes' powers are unchanged to twelve digits, so that omission is a number rather than a sentence.

Off by default, and with all the light on one axis it changes nothing at all — on the samples,
which is what makes it safe to leave on for a dual-polarization link and pointless for a
single-polarization one.

## D is not one number

Standard fibre gains 0.058 ps/nm/km of dispersion for every nanometre up the
band. Set `dispersion_slope` and the fibre stops pretending otherwise — and the same
coefficient shows up in two places at once, because it is the same coefficient.

**Across a comb**, channels no longer share a dispersion, so one compensator setting cannot serve
all of them. Over 80 km, with the compensator set for 1550 nm:

| channel | D there | D·L there | mismatch | EVM, one setting | EVM, its own |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1550 nm | 17.000 | 1360.0 | +0.0 | 1.68 % | 1.68 % |
| 1554 nm | 17.232 | 1378.6 | +18.6 | 4.07 % | 1.68 % |
| 1558 nm | 17.464 | 1397.1 | +37.1 | 7.76 % | 1.68 % |
| 1566 nm | 17.928 | 1434.2 | +74.2 | 15.23 % | 1.68 % |

That is the same size of penalty as the mis-settings table above it, arriving without anybody
having mis-set anything. It is why a single dispersion-compensating fibre never flattened a whole
C-band, and why coherent receivers carry a per-channel setting.

**Within one channel** the slope is a cubic phase, and cubic phase broadens a pulse
*asymmetrically* where β₂ broadens it evenly. It is honestly small at these baud rates: over
1000 km the cubic phase across a 32 GBd band is about 0.04 radians and costs a tenth of a point of
EVM. `accumulated_slope` on the compensator removes it anyway, and the test that it does is worth
having because a slope compensation with the wrong sign *doubles* the residue rather than removing
it — and nothing about the width of a pulse could ever tell you.

**β₃ is not the slope by another name.** Even at zero slope a fibre has a nonzero β₃, because
holding D flat across wavelength is itself a statement about how β₂ varies:

    beta3 = (lambda^2 / 2*pi*c)^2 * (S + 2*D/lambda)

For standard fibre at 1550 nm, D = 17 and S = 0.058 give β₃ = 0.13 ps³/km, the value the
literature quotes. Feeding it S = 0.09 — the slope at the *zero-dispersion* wavelength, which is
the number datasheets lead with — gives 0.18, and is the easiest way to be forty percent wrong.

The cubic term's sign is derived from the same Taylor expansion of β(ω) that gives the group delay
and the dispersion, not written down, because with this module's transform pair the quadratic and
cubic terms land on *opposite* signs. Broadening is even in β₃, so a sign error produces exactly
the right width and exactly the wrong skew; the tests measure the skew.

Left at zero the slope changes nothing. Every number taken before it existed still comes out, on
the same samples.

## Finding the dispersion without being told it

A dispersion compensator has to be set to within a few ps/nm. Over 80 km at 32 GBd the true value
is 1360, and being 17 out — one kilometre of fibre — takes the EVM from 1.7 % to 3.8 %, while 136
out lands the symbols at chance. It is not a knob to be roughly right about, because the residual
after a mismatch *is* the mismatch.

No deployed receiver is ever told the number. It measures it during acquisition, from the signal,
before the equaliser or the carrier loop have converged. Set `estimate` on the compensator and it
does the same:

| span | accumulated | estimated | error | EVM declared | EVM blind |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 km | 0 ps/nm | −7.4 | −7.4 | 1.68 % | 2.23 % |
| 20 km | 340 ps/nm | 336.4 | −3.6 | 1.67 % | 1.82 % |
| 80 km | 1360 ps/nm | 1355.3 | −4.7 | 1.68 % | 1.92 % |
| 400 km | 6800 ps/nm | 6793.9 | −6.1 | 1.67 % | 2.05 % |
| 1000 km | 17000 ps/nm | 16991.5 | −8.5 | 1.68 % | 2.39 % |

The last two columns are the ones to read together: being right in ps/nm is not the claim, leaving
behind a link as good as one that was handed the answer is.

**Two stages, because no one statistic does both jobs.** The intensity of a modulated signal
repeats at the symbol rate, so `|A|²` carries a line there — and dispersion gives every pair of
frequencies a symbol rate apart a relative phase that grows with frequency, so their contributions
cancel and the line fades. Scanning that line across ±20000 ps/nm acquires the value with an
enormous capture range and an error of tens of ps/nm. Then a second stage minimises how Gaussian
the intensity looks, which has curvature exactly where the tone is flat.

The scan never compensates anything. The line at the symbol rate is a correlation of the spectrum
with itself shifted by that rate, and compensation only multiplies the spectrum by a phase, so
every candidate is one dot product against a product computed once from a single transform. It is
an identity, held to 1e-12 against compensate-then-transform in the tests, and it made 400
candidates over a 65536-sample window twenty times cheaper.

**It needs excess bandwidth**, and says so: at zero roll-off the tone this searches for does not
exist at all. Between there and roll-off 0.1 it exists but acquisition comes in a consistent
950 ps/nm low, and what recovers those runs is refinement walking its window in rather than any
warning from the contrast figure — which scored 29 on a run that was wrong by 950 and 32 on one
that was right. Resolution also scales as 1/R\_s²: the same link at 10 GBd lands 44 ps/nm out where
32 GBd lands 5. All of that is in
[`estimate_dispersion`](src/maiman/dsp.py)'s docstring, measured rather than asserted, and the
third table of [`examples/dispersion_link.py`](examples/dispersion_link.py) prints it.

## What channels do to each other

Until recently bands propagated independently through the fiber, which made this a good model of
one channel and an optimistic model of a comb. They no longer do. The same `|A|²A` term that gives
a channel its own self-phase modulation lets every *other* channel rotate its phase — and the
coefficient is not free. Expanding the term for a sum of carriers, a channel's own power appears
once and a neighbour's appears twice, because there are two ways to choose which un-conjugated
factor belongs to the neighbour and one way when it is the channel itself.

That factor of two is measurable, and it is exact:

| co-propagating channels | mean nonlinear phase | vs one channel | closed form |
| :--- | ---: | ---: | ---: |
| 1 | 0.027518 rad | 1.000× | 1× |
| 2 | 0.082559 rad | 3.000× | 3× |
| 3 | 0.137600 rad | 5.000× | 5× |
| 4 | 0.192640 rad | 7.000× | 7× |

**Dispersion is the cure here, not the disease.** Chromatic dispersion is normally introduced as
something to compensate. Between channels it is the only thing keeping them apart: it makes them
travel at different speeds, so a neighbour's bit pattern *slides past* instead of sitting on top of
the channel it is modulating. Walk-off is therefore not a separate parameter — it is the
group-delay term of the same expansion of β(ω) that produces the dispersion, so setting D to zero
removes both at once. At 17 ps/nm/km a 100 GHz neighbour separates by 13.62 ps/km, and has slid 140
symbols by the end of a 320 km link.

What walk-off removes is not the cross-phase modulation but its *variation*. The mean phase shift
is fixed by the neighbour's average power and no amount of sliding changes it — sliding
redistributes in time, it does not destroy. That split is the whole mechanism, because a constant
phase offset is absorbed by carrier recovery for free and it is the variation that closes an eye.
Measured on channel 1 of a four-channel QPSK comb over 4 × 80 km, as EVM after carrier recovery:

| launch/channel | D = 0 | D = 17 ps/nm/km |
| :--- | ---: | ---: |
| −3 dBm | +0.18 % | +0.00 % |
| 0 dBm | +0.49 % | +0.03 % |
| +3 dBm | **+2.84 %** | **+0.29 %** |

Ten times less penalty for having dispersion in the fiber.

**Four-wave mixing lands on the channels.** Products appear at `f_i + f_j − f_k`, and on an equally
spaced grid those frequencies *are* channel frequencies — so the crosstalk arrives in band, where
no filter downstream can reach it. The model folds such a product into the channel it lands on,
with a phase drawn from the run's generator for the same reason PMD is drawn: it is set by fiber
details nobody measures. Dispersion suppresses mixing too, by dephasing it, and the phase mismatch
grows as the *square* of the channel spacing:

| channel spacing | D = 0 | D = 2 | D = 17 |
| :--- | ---: | ---: | ---: |
| 25 GHz | 0.0 dB | −4.4 dB | −21.2 dB |
| 50 GHz | 0.0 dB | −14.7 dB | −33.1 dB |
| 100 GHz | 0.0 dB | −26.7 dB | −45.4 dB |
| 200 GHz | 0.0 dB | −38.6 dB | −57.4 dB |

Zero-dispersion fiber is perfectly phase matched at every spacing, which is the whole reason
dispersion-shifted fiber was abandoned for WDM.

The two effects are computed differently, and the difference is worth knowing before reading a
number off the block. Cross-phase modulation is solved on the waveform inside the split-step,
because it depends on the neighbour's instantaneous power sliding past; four-wave mixing is solved
in closed form from the band powers and injected as tones, because the products land at frequencies
no band is sampled at. Not putting the channels on one grid is what makes a WDM comb affordable at
all, and that choice has to be paid for somewhere. See
[`examples/wdm_nonlinear.py`](examples/wdm_nonlinear.py).

## A foundry's numbers, not a foundry's code

A process design kit here is **not** a component library. It adds no devices, carries no layout,
and reading one executes nothing — a `.pdk` is JSON, read the way a `.maiman` project is, and for
the same reason: opening a file somebody sent you must not be equivalent to running their code.
`"component": "os.system"` is refused like any other name that is not in the registry.

What it carries is the half of a real design the engine cannot derive. `maiman.photonics` knows
what a directional coupler *is*. It has no way to know that this process makes a nominal 3 dB
coupler measuring 0.48 at 1550 nm, or that the nitride guide beside the silicon one is twenty times
quieter. Those numbers come off a wafer.

```python
from maiman import load_pdk

pdk = load_pdk("examples/silicon_220nm.pdk.json")
coupler = pdk.make("dc_3db", label="split")        # 0.48, not 0.5
splitter = pdk.make("mmi_1x4", wavelength=1565.0)  # the fits, evaluated there
```

**A value in a kit may be a fit rather than a constant** — a polynomial in `λ − λ_ref` in
nanometres, which is the form a foundry quotes one in. It is evaluated when the component is built.
The device models here are frequency-flat by construction, so what a kit buys is the right constant
for the band you are working in, chosen by the file rather than guessed. That "3 dB" coupler is a
1.5 / 5.5 dB split at 1500 nm and 4.9 / 1.7 at 1600 — which is the entire reason the MMI sits
beside it in the kit.

**A fit has a window, and running outside it is refused.** This is the part worth the code. That
first-order C-band fit, extrapolated to 1310 nm, returns **minus 0.46** — a negative power fraction,
out of arithmetic that never complained once. `valid_wavelengths` turns it into a sentence naming
the range and the device. Everything else a kit can get wrong — a component that is not registered,
a parameter the component does not declare, a cross-section that does not exist, a cross-section
attached to a lumped device with no waveguide to apply it to — is refused at load, naming the file
and the entry, because a kit is read once and used a hundred times.

And nominal is not what you build. The same 1 × 4 splitter, twice:

    out1     out2     out3     out4
    textbook         -6.021   -6.021   -6.021   -6.021   total +0.000 dB   spread 0.000 dB
    silicon-220nm    -6.298   -6.414   -6.531   -6.648   total -0.450 dB   spread 0.350 dB

See [`examples/pdk_import.py`](examples/pdk_import.py) and the kit it reads,
[`examples/silicon_220nm.pdk.json`](examples/silicon_220nm.pdk.json) — representative of the open
multi-project-wafer processes and drawn from published literature, not from anyone's confidential
kit. Replace it with yours.

## The session server

    maiman serve

Three routes, no dependencies beyond the ones the engine already has, bound to loopback.

| route | what it does |
| :--- | :--- |
| `GET /api/manifests` | Every registered component: parameters with units and bounds, ports with types |
| `POST /api/run` | A `.maiman` project document in, results out |
| `POST /api/sweep` | A project, an axis and a range in; a curve out, as numbers |
| `GET /api/health` | Whether it is up, and how many components it knows |

**It is an ordinary client of the public Python API.** Every route is a thin wrapper over something
a script can already call — `manifests()`, `graph_from_dict()`, `Graph.run()`. Nothing in the
server knows any physics and nothing in the engine knows the server exists, which is what stops
the two drifting apart: a feature unreachable from Python is unreachable from the interface too.
The `Results.items()` the server needs is public for the same reason.

**Reduction happens in the engine, never in the browser.** A run holds tens of thousands of samples
per port; an eye diagram drawn from them is a 96×96 histogram. So a signal-carrying port encodes to
a *summary* — how many samples, at what rate, how much power — and anything meant to be looked at
arrives already reduced, from a measurement component the graph contains explicitly. A full
coherent run is **52 kB of JSON**. That is not a size optimisation: it is what keeps a second,
untested implementation of the physics from growing in JavaScript.

Every encoded value carries a `kind`, so a client switches on a string rather than guessing from
which fields happen to be present. A result type nothing knows how to draw arrives tagged `opaque`
rather than failing the run or being guessed at — which is the honest answer for a plugin's own
metric, and a test asserts it is never the answer for anything shipped here.

Non-finite numbers become `null`. `json.dumps` emits bare `NaN` and `Infinity`, which are not JSON
and which `JSON.parse` rejects outright, so one infinite Q factor would fail a whole response
rather than one field — and infinities are a normal result here, not an error.

**Errors say whose fault they are.** 400 means the request could not be read, 422 means it was read
and will not run, 413 means it was too big to attempt:

| | |
| :--- | :--- |
| `no component registered as 'FluxCapacitor'. If it comes from a plugin package…` | 400 |
| `a window of 32000000 samples exceeds the server limit of 4194304` | 413 |
| `pm.in is not connected` | 422 |

`POST /api/run` executes the graph it is given, so the socket binds to 127.0.0.1 unless a host is
passed explicitly, and a run is refused *before* it starts if its window is too large. The registry
already does the harder half of this: a project file may only **name** components that are already
registered and can never import a dotted path, so opening someone else's project is not equivalent
to running their code.

## What this is

A block-diagram simulator for optical systems: drop components on a canvas, wire a link, run it,
get an eye diagram and a BER number — with correct optical noise accounting. And the same model,
scriptable from Python.

Think **OptiSystem's workflow, VPIphotonics' ambition, open source, and verifiable.**

```
  PRBS ──▶ NRZ ──▶ CW Laser ──▶ MZM ──▶ Fiber ──▶ PIN ──▶ BER / Eye
                                 ▲
                            (V_π, ER, IL)
```

## Why

There is a real gap in open-source photonics tooling, but it is **not** where it first appears
to be.

The physics kernels already exist. SSFM, coherent DSP, and S-matrix circuit solvers are all
available in open source — as libraries, in fragments, each with its own incompatible data model.
What does not exist is a single coherent tool that puts them behind a usable interface with a
sound signal representation underneath.

So the gap is **integration and user experience**, not numerics. That shapes the whole strategy:
where a mature open-source kernel exists, Maiman Studio wraps or depends on it rather than
rewriting it. The value added here is the data model, the execution engine, the component
library, and the UI.

| Existing tool | What it does | Relationship |
| :--- | :--- | :--- |
| [OptiCommPy](https://github.com/edsonportosilva/OptiCommPy) | Python: SSFM, coherent DSP, BER | Reference & cross-validation target |
| [GNPy](https://github.com/Telecominfraproject/oopt-gnpy) | Optical network planning / OSNR budgets | Complementary — network layer, not waveform layer |
| [QAMPy](https://github.com/ChalmersPhotonicsLab/QAMpy) | Coherent DSP algorithms | Reference for Phase 3 |
| [SAX](https://github.com/gdsfactory/sax) | S-matrix photonic circuit solver | **Cross-validation reference, not a dependency.** The reduction is twelve lines and SAX resolves to 37 packages including an LGPL sparse back-end; the two agree to 7e-15 |
| [gdsfactory](https://github.com/gdsfactory/gdsfactory) | Photonic layout & PDK ecosystem | **Connected, and still not a dependency.** [`maiman.netlist`](src/maiman/netlist.py) reads the YAML netlists it writes and solves them against a `.pdk`. Nothing imports it: measured, it resolves to **86 packages** and requires `klayout`, which is **GPL-3.0-or-later** — the same ground FFTW and klujax are refused on. Reading a document costs none of that |
| [Meep](https://github.com/NanoComp/meep) | FDTD / full-wave EM | Feeds component models *in*; not a competitor |
| [GNU Radio](https://www.gnuradio.org/) | Block-based SDR | Architectural reference for dataflow scheduling |

## Design decisions

These are the choices that matter, stated up front so they can be argued with. Full reasoning is
in the [architecture document](docs/ARCHITECTURE.md).

| Decision | Rationale |
| :--- | :--- |
| **Engine before GUI** | A wrong engine forces a total rewrite; a wrong GUI does not. |
| **Multi-band optical signal** from day one | A single scalar carrier frequency cannot express a 40-channel DWDM system without a physically impossible sample rate. Discovering that in Phase 3 means rewriting the core. |
| **Noise carried in spectral bins**, separate from sampled fields | ASE spans the amplifier bandwidth; the signal does not. Sampling both together makes realistic runs impossible. |
| **Block-mode execution** (whole time window per call) | Every component becomes a pure function of its inputs. Vectorizes naturally; not a streaming scheduler. |
| **Python-first**, behind a narrow kernel boundary | ~90–95% of SSFM runtime is inside FFT — library code in any language. GPU via CuPy is nearly free. Contributors who write fiber models write Python. Native kernels stay an option, not a prerequisite. |
| **Typed ports** (Optical / Electrical / Binary / Symbol / Metric) | An MZM has an electrical input. Invalid wiring is rejected at edit time, not at run time. |
| **Immutable signals** | A WDM link is hundreds of MB. Value-copying between blocks makes the tool unusable regardless of language. |
| **Python-package plugins**, parameters declared once | Requiring contributors to match a C++ ABI suppresses exactly the contributions the plugin system exists to attract. |
| **Every physics block validated against a closed-form result** | Comparison against commercial tools needs a licence and is not reproducible in CI. A simulator nobody can verify has no scientific value. |
| **No FFTW** | FFTW is GPL-2.0-or-later. Linking it would make the project GPL. pocketfft (BSD) is used instead. |

## Architecture

```text
                    ┌──────────────────────────────────────┐
                    │        Visual Designer (Web UI)      │
                    │   graph editor · plots (WebGL)       │
                    └───────────────────┬──────────────────┘
                                        │  project JSON + WebSocket
                                        │  (data reduced engine-side)
                    ┌───────────────────▼──────────────────┐
                    │            Session Server            │
                    └───────────────────┬──────────────────┘
 ┌──────────────────────────────────────▼───────────────────────────────────────┐
 │                             Public Python API                                 │
 │        maiman.Graph · Component · run() · sweep()                              │
 └──────────────┬─────────────────────────────────────┬─────────────────────────┘
 ┌──────────────▼───────────────┐    ┌────────────────▼─────────────────┐
 │      Component Library       │◄──►│         Execution Engine         │
 │  plugins · registry · schema │    │  scheduler · sweeps · run graph  │
 └──────────────┬───────────────┘    └────────────────┬─────────────────┘
 ┌──────────────▼─────────────────────────────────────▼─────────────────────────┐
 │                     Core Data Model + Numerical Kernels                       │
 │       SimulationContext · Signals · FFT / SSFM / filters / noise              │
 │              back-ends: NumPy → CuPy → (optional) native                      │
 └──────────────────────────────────────────────────────────────────────────────┘
```

The GUI is a client of the public Python API with no privileged access. **If a feature is not
reachable from Python, it does not exist.**

## The core data model

The part most worth reviewing — see [`src/maiman/signals.py`](src/maiman/signals.py). An optical
signal is not one array of numbers:

```python
@dataclass(frozen=True)
class Band:
    """One sampled band: complex envelope in two orthogonal polarizations (Jones vector)."""

    Ex: np.ndarray  # complex64, shape (N,), read-only
    Ey: np.ndarray
    f0: float  # band centre frequency [Hz]
    fs: float  # band sample rate [Hz]


@dataclass(frozen=True)
class NoiseBin:
    """Spectrally-resolved noise, carried separately from the sampled bands."""

    f_start: float
    f_end: float
    psd_x: float  # [W/Hz] per polarization
    psd_y: float


@dataclass(frozen=True)
class OpticalSignal:
    bands: tuple[Band, ...]
    noise: tuple[NoiseBin, ...]
    accumulated_gvd: float    # sum(beta2 * L) over the path so far [s^2]
```

Fields are `sqrt(W)`, so instantaneous power is `|Ex|**2 + |Ey|**2`. Arrays are read-only, which
is what lets metadata-only blocks share buffers instead of copying a span at a time.
`accumulated_gvd` is the one piece of *path* state on the signal, and it is there so that mixing
products generated in different spans can be added as fields — see
[above](#mixing-products-add-in-field-not-in-power).

Global run parameters (bit rate, oversampling, sequence length, RNG seed) live in a shared
`SimulationContext`, not in individual signals — so blocks cannot silently disagree about the
time window, and results are reproducible.

## Roadmap

| Phase | Scope | Estimate¹ |
| :--- | :--- | :--- |
| **0 — Foundations** ✅ | Signal model, context, port types, component base, registry, scheduler, `.maiman` project format, sweeps, CI | ~1 month |
| **1 — MVP: linear link** ✅ | ✅ PRBS → NRZ → laser → MZM → fiber (α + CD) → PIN → filter → eye/Q/BER, validated end to end. **Python only, no GUI.** | ~2–3 months |
| **1.5 — Nonlinear & amplified** ✅ | Adaptive-step SSFM, Kerr, EDFA with ASE and Saleh gain compression, erbium gain dynamics on their own time axis, OSNR, PMD, APD, dispersion slope and its third-order term, cross-polarization Kerr coupling, inter-channel stimulated Raman scattering | ~2 months |
| **2 — Coherent transceiver** ✅ | Gray-coded M-QAM to 256, IQ modulator with bias and quadrature error, 90° hybrid, balanced detection, blind carrier frequency and phase recovery, coarse frequency acquisition over the whole sampled band, blind square-law timing recovery, dual polarization with a blind butterfly equaliser, root-raised-cosine shaping and matched filtering, differential quadrant encoding, receiver-side dispersion compensation over spans to 1000 km with blind estimation of the accumulated value, EVM/MER, constellation diagram, validated against closed-form SER | ~3 months |
| **3 — GUI & WDM** ✅ | Wavelength-selective filters, the ITU grids and a multiplexer/demultiplexer pair on them with crosstalk that falls out of the channel spacing, an OSA, coupled-channel propagation (XPM with walk-off, FWM accumulating coherently across spans), the session server, a schematic editor — add, wire, move and delete blocks, edit parameters, run, sweep, open and save — the OSA's trace drawn in the dock, and 400G/800G reference designs validated against the OSNR relations, and a back-end indirection the propagation kernels dispatch through — CuPy runs it where a device exists, `maiman devices` cross-checks it against NumPy, and a CI job does the same on any runner labelled `gpu` | ~6 months |
| **4 — PIC** ✅ | Bidirectional S-matrix circuit solver, waveguide, directional coupler, all-pass and add-drop ring resonators, cross-validated against SAX; N×N MMI couplers on the self-imaging phase relations; a Mach-Zehnder interferometer assembled from them — switch, interleaver, or both; and PDK import, which reads a foundry's fitted numbers out of a JSON kit and refuses to extrapolate them past the window they were fitted in; and birefringence, with each guided polarization carrying its own indices through the same reduction | — |

¹ One developer, part-time. Estimates, not commitments.

Phase 1 is deliberately smaller than a first instinct suggests: SSFM, PMD, Kerr, APD and the GUI
are all pushed out of it. Shipping a *validated* linear link quickly matters more than breadth.

## Validation

Every physics block ships with a test against a closed-form result, run in CI
([`tests/test_physics.py`](tests/test_physics.py)):

| Case | Expected | |
| :--- | :--- | :-- |
| Attenuation | `P_out = P_in · 10^(-αL/10)` | ✅ |
| Cascaded spans | Loss is additive in dB | ✅ |
| Source power | Independent of the simulated time window | ✅ |
| Phase noise | Broadens the line, conserves average power | ✅ |
| Multi-carrier | Channels stay separate bands; spacing does not drive `Fs` | ✅ |
| Gaussian pulse, CD only | `T₁/T₀ = √(1 + (z/L_D)²)`, `L_D = T₀²/\|β₂\|` | ✅ |
| Chirped Gaussian | `T₁/T₀ = √((1 + Cβ₂z/T₀²)² + (β₂z/T₀²)²)` — pins the sign of β₂ | ✅ |
| Dispersion compensation | `+D` then `−D` restores the input sample-for-sample | ✅ |
| Receiver-side CD removal | Compensator is the propagator inverted; round trip exact to 1e-9 | ✅ |
| CD compensation is all-pass | Energy conserved; a wrong sign lands exactly on twice the span | ✅ |
| β₂ ∝ λ² | Compensating 1550 nm as 1310 nm leaves the predicted `1 − λ₁²/λ₂²` residual | ✅ |
| **Span recovery** | 5 km to 1000 km return to back-to-back EVM; uncompensated 80 km is at chance | ✅ |
| GVD | Energy conserved (Parseval); β₂ = −Dλ²/2πc per band | ✅ |
| PRBS | Period `2ⁿ−1`; `2ⁿ⁻¹` marks; every n-bit window appears once | ✅ |
| Ideal push-pull MZM | `P_out/P_in = cos²(πV / 2V_π)`; null depth equals the declared ER | ✅ |
| PIN detector | `I = R·P`; shot `σ² = 2qIB`; thermal `σ² = 4kTB/R_L` | ✅ |
| Receiver filter | 3 dB at `B`; noise bandwidth `B·√(π/4ln2)`; zero group delay | ✅ |
| **BER** | `½·erfc(Q/√2)` matched against **directly counted errors**, 10⁻⁴–10⁻¹ | ✅ |
| **SER, every order** | Counted errors against `ser_qam` within 12 %, at 1–8 bits/symbol and 4–22 dB | ✅ |
| Rectangular SER ≡ square SER | The generalised expression reproduces the textbook square formula to 2e-16 at every even order, 0–39 dB | ✅ |
| Odd orders are Gray coded | Every nearest-neighbour pair differs in exactly one bit — which a cross constellation provably cannot manage | ✅ |
| Rotational symmetry is measured | 4 for a square, 2 for a rectangle, read off the point set; narrowing the phase search back to π/2 breaks 32-QAM at a 0.6π offset | ✅ |
| Link consistency | `L` km of span ≡ launching `α·L` dB lower, end to end | ✅ |
| Lossless SSFM | Energy conserved with nonlinearity; γ=0 reproduces the exact linear solution | ✅ |
| Self-phase modulation | `\|A(T)\|` exactly unchanged; spectrum broadens | ✅ |
| **Fundamental soliton (N=1)** | Envelope invariant over 4 soliton periods — **the only test that pins the sign of γ against β₂** | ✅ |
| Higher-order soliton (N=2) | Compresses at half a period, recovers at a full one | ✅ |
| EDFA | `P_ASE = 2·n_sp·hν·(G−1)·B_o`; `n_sp = NF·G/2(G−1)` | ✅ |
| OSNR | `58 + P_launch − NF − 10·log10(spans)`, over 16 spans | ✅ |
| **Signal-ASE beat** | Q on an amplified link tracks `2√(B_ref/B_e)·OSNR/(1+√(1+4·OSNR))` to 15% | ✅ |
| ASE beat, coherent | Electrical SNR converges on `2·OSNR·B_ref/R_s` as ASE dominates — 0.23 dB | ✅ |
| Beat is polarization-selective | Co-polarized ASE beats; orthogonal ASE does not, on both detectors | ✅ |
| Filter noise bandwidth | `B_n = B·Γ(1+1/2n)/ln2^(1/2n)`, against numerical integration; order 1 is the Gaussian | ✅ |
| Filtered ASE power | Exactly density × `B_n`; a demux passes its own equivalent noise bandwidth | ✅ |
| Wavelength selectivity | A filter between two channels attenuates both; rejection stops at `extinction` | ✅ |
| OSA normalisation | Trace integrates back to an independent power meter; ASE reads density × RBW | ✅ |
| Per-channel OSNR | Survives a demultiplexer that suppresses three channels of four | ✅ |
| Matched filtering | Costs `10·log10(f_s/R_s)` to omit — the receiver integrates noise it cannot use | ✅ |
| **PMD** | DGD Maxwellian: `⟨τ²⟩/⟨τ⟩² = 3π/8`, mean `∝√L`, spread `0.42·mean` | ✅ |
| APD | `F(M) = kM + (2−1/M)(1−k)`; an **interior optimum gain** exists | ✅ |
| **Cross-phase modulation** | `n` equal channels give `(2n−1)×` one channel's nonlinear phase — exact to 1e-3 | ✅ |
| XPM swing | Peak-to-peak `2·γ·P·L_eff` on a probe beside an on/off pump, with no walk-off | ✅ |
| Walk-off | `D·Δλ` per unit length, derived from β₂ and not declared beside it | ✅ |
| Walk-off conserves the mean | Mean XPM phase fixed at `2·γ·⟨P⟩·L_eff` across a 16× change in slip, while its spread falls 5.7× | ✅ |
| FWM efficiency | `η → 1` phase matched; `→ sinc²(Δβ·L/2)` lossless; even in Δβ | ✅ |
| FWM phase mismatch | `Δβ = −β₂(ω_i−ω_k)(ω_j−ω_k)` — quadratic in spacing, zero at zero dispersion | ✅ |
| FWM product power | Component reproduces `d²γ²P_iP_jP_k·L_eff²·η·e^{−αL}` to 1e-7; cubic in power; `d = 2−δ_ij` gives non-degenerate products exactly 6.02 dB | ✅ |
| **Circuit reduction** | Eliminating internal ports agrees with summing round trips lap by lap to 1e-13 — a different algorithm, sharing no code | ✅ |
| Reduction vs SAX | Same models, same wiring: 7.2e-15 over 4001 frequencies. Not in CI — it costs 37 packages and a licence review | — |
| Non-reciprocal and reflecting devices | An isolator stays one-way and a mirror returns `r·e^{−2iβL}`; the reduction assumes neither | ✅ |
| Dangling ports | An unwired port is `a = 0`, not a mirror — a 3 dB coupler with one port open passes exactly half | ✅ |
| **Ring resonator** | Assembled from a coupler and two arcs, matches Yariv's all-pass and add-drop transfer functions to 1e-13 | ✅ |
| Free spectral range | `c / (n_g L)` to 1e-4, measured between resonances; `n_eff` moves them and not their spacing | ✅ |
| Critical coupling | Extinction below 1e-12 at `κ = 1 − a²`; under- and over-coupled partners notch identically | ✅ |
| Coupler unitarity | `SᴴS = I` at every split ratio — which is what the cross path's factor of j is for | ✅ |
| Resonance linewidth | Lorentzian `FSR(1−r)/π√r` within 3 % of a measured width from critical coupling to κ = 0.5 | ✅ |
| Waveguide group delay | `n_g L / c` read off the transfer function's phase slope, to 1e-9 | ✅ |
| **Timing estimate** | Tracks a known delay one for one over ±0.45 symbol, to 2e-3 | ✅ |
| Fractional delay is exact | Forward then back returns the input to 1e-12 — a phase ramp, not an interpolation | ✅ |
| Timing recovery earns its place | 500 µm of waveguide costs 675 symbol errors in 1920; with the stage, none, and it moves by the 7.00 ps the guide actually holds | ✅ |
| A whole symbol is invisible | Delay by one symbol period and the estimate does not move — the limit is `\|A\|²`, not the code | ✅ |
| Every shipped project still opens | All six `.maiman` files load, run, and carry their canvas layout — the first thing a new user opens, and nothing checked them before | ✅ |
| **A pilot is an erasure, not an error** | LLR set to zero where the coder's bits were overwritten: 4.9e-4 out of a 9.9e-3 channel, against 2.8e-2 if they are believed | ✅ |
| **A netlist a layout tool wrote solves** | gdsfactory's own shipped sample, unmodified: -0.0053 dB against the 26.637 µm at 2 dB/cm anyone can work out by hand | ✅ |
| Lengths come from the netlist, not the kit | The built waveguide is 10 µm where the kit's nominal is 1000 — asserted on the device, so a lost override says *why* | ✅ |
| A routed circuit resonates where its loop says | FSR 713.74 GHz against `c/(n_g·L)` for the 100.000 µm the instances sum to | ✅ |
| An unmapped cell or port is refused by name | Skipping either returns a circuit that solves and is not the one drawn | ✅ |
| A window narrower than one FSR finds nothing | Kept as a test because a flat 0.995 reads as a ring that does not resonate rather than a scan that is too narrow | ✅ |
| **A ring resonates at two sets of wavelengths** | Two combs, each matching `c/(n_g·L)` for its own group index to a part in a thousand, and their ratio the ratio of the indices | ✅ |
| The combs coincide when the indices do | The control for the above — an off-by-one in the stacking would separate them where there is nothing to separate them | ✅ |
| Stacking reduces exactly to one polarization | Same indices twice reproduces the single-polarization matrix element for element, and the block's output to `rel=1e-12` | ✅ |
| The modes do not leak into each other | Exactly zero, checked at the resonance where circulation would amplify any leak into something visible | ✅ |
| A cross term lands where the solver finds it | Nothing produces one; the matrix can still hold it, so block-diagonal stays a model's choice | ✅ |
| **Gain dynamics settle onto the static solve** | The reservoir ODE integrated to rest lands on `EDFA.effective_gain` to 1e-9, at seven input powers — two code paths sharing no arithmetic | ✅ |
| The effective time constant is `τ/(1+P_out/P_sat)` | Measured from a step response against the closed form, to a part in a thousand, and monotone in drive | ✅ |
| A transient is far longer than a window | 25,000 windows at the most saturated point — the measurement the decision to keep it out of the component rests on | ✅ |
| A coarse output grid still gets the right curve | Identical to 1e-6 dB across a 200× range of grid spacing; the integrator takes its own steps and reports how many | ✅ |
| Relaxation is monotone | A first-order system cannot ring, so an overshoot is an integrator bug rather than physics | ✅ |
| **The analyser finds a carrier it was not aimed at** | 1310, 1480, 1550, 1560 and 1625 nm, each located to 0.05 nm with all of its power in the trace — where a fixed window saw nothing outside 1550 | ✅ |
| Full span covers every band, not the loudest | Two carriers 40 nm apart and 20 dB different are both inside the window | ✅ |
| A fixed window still means what it did | `auto_span` off and the 1560 nm carrier is outside the declared span again, exactly as before | ✅ |
| **Acquisition reaches where the fine stage folds** | Band located to ±200 MHz from 0 to ±200 GHz, both signs — 50× past the M-th power's unambiguous range | ✅ |
| The two failure modes are opposites | The fine stage wrong by a whole `symbol_rate/M` at unchanged confidence; the coarse stage's Nyquist wrap correct to derotate by | ✅ |
| No bandwidth is told to it | Roll-off 0 through 1 located identically — a window formulation given 1.6·R_s for a 1.2·R_s band lands 3.9 GHz out | ✅ |
| A flat noise floor does not pull it | Centre unmoved under noise at the signal's own power; only the concentration falls, which is what that number is for | ✅ |
| Acquisition returns a link the fine stage cannot | 4.1, −4.1 and 20 GHz: ~1787 symbol errors without it, zero with it, at the undetuned link's own EVM | ✅ |
| **Pilots resolve every quarter turn** | All four rotations recovered identically and exactly — resolving three of four would make the link work three times in a row and then not | ✅ |
| Pilots are legal symbols, and vary | Drawn from the alphabet's outermost ring, so a pilot is not itself an error, and never constant, so it is not a spectral line | ✅ |
| The estimate reads only the pilots | Every other reference symbol corrupted, and the answer unchanged to 1e-12 — otherwise it is a data-aided estimator in disguise | ✅ |
| Soft information survives the flagship | Real distance to the nearest point and real spread in the LLRs, where the differential decoder gave zero and 1e29 | ✅ |
| **Soft FEC clears what hard FEC cannot** | −25 dBm, 7.4e-3 on the line: the staircase delivers an exact payload where RS(255,239) returns its input | ✅ |
| A staircase stripe row is a codeword | Structural, block by block — the braiding asserted rather than inferred from a curve | ✅ |
| Chase beats its component code | Four errors recovered on a `t=2` code, because they were the least reliable bits | ✅ |
| No competitor means *more* sure, not less | The reliability floor, and the bug it fixes: a negative extrinsic destroys bits the decoder never touched | ✅ |
| A clean block survives the decoder | With varying reliabilities, which is where the broken version failed and a uniform input did not | ✅ |
| The final block is provisional | One error in it is not corrected where the same error anywhere else is — the arrangement, not the code | ✅ |
| Soft cannot be wired into hard | The sixth port type refuses it, because that mistake runs rather than crashes | ✅ |
| **RS(255,239) corrects exactly eight** | Any eight symbol errors anywhere, exactly recovered; nine reported as a failure rather than silently mangled | ✅ |
| The generator's roots are its definition | All sixteen consecutive roots annihilate a derived `g(x)` — the claim, not a transcribed coefficient table | ✅ |
| A burst inside a byte is one error | 64 bad bits in eight bytes correct; 9 bad bits in nine do not — the whole argument for a symbol code | ✅ |
| Closed form matches a counted decode | Bounded-distance expression against Monte Carlo at 3e-3 and 2e-3, within 25 % | ✅ |
| A coded link runs error-free on a broken line | −19 dBm: 7.7e-4 on the line, zero errors in the payload, through real optics and a blind slicer | ✅ |
| Below threshold it degrades *badly* | Post-FEC rate higher than pre-FEC, because miscorrection is what a bounded-distance decoder does past its limit | ✅ |
| **Gain compresses 3 dB at the declared point** | The datasheet definition, checked as a definition, at 10, 20 and 30 dB of gain — the 10 dB row is what catches a dropped `(G₀−2)/G₀` | ✅ |
| The solved gain solves the equation | Newton's answer put back into the implicit Saleh relation, residual under 1e-12 of the small-signal gain | ✅ |
| Compression is smooth, not a ceiling | Gain falls at every step from −40 dBm up and output never stops rising — the two things a clamp gets qualitatively wrong | ✅ |
| Channels share one inversion | 1, 2, 4, 8 channels through a real combiner: 13.61, 12.74, 11.56, 10.12 dBm each, and 8× the input for 5.5 dB more output | ✅ |
| A retired clamp still opens | A project carrying `max_output_power` loads with it dropped rather than refusing | ✅ |
| **RIN-limited SNR** | `1/(RIN·B_n)` to 0.014 dB at −155, −145 and −135 dB/Hz, against the Gaussian filter's closed-form noise bandwidth | ✅ |
| The RIN floor ignores power | Ten times the launch power returns a bit-identical SNR — the one property that makes intensity noise worth modelling | ✅ |
| Intensity noise conserves average power | Moves the variance to `RIN·fs/2` and leaves the mean at the declared dBm, the counterpart of the linewidth invariant | ✅ |
| The two laser noises are independent | Sweeping the linewidth leaves the intensity samples bit-identical, so neither is measuring the other | ✅ |
| **Carrier offset estimate** | Lands on a known offset to 0.5 MHz at both signs across the whole ±3.9 GHz range, on a 32 GBd link | ✅ |
| Frequency recovery earns its place | A 10 MHz LO detuning — 0.03 % of the symbol rate — costs 1677 symbol errors in 1920; with the stage, none | ✅ |
| The stripping power is the geometry's | 8-QAM is refused a quarter turn and stripped at a half, so its range is ±8 GHz and not ±4 | ✅ |
| Past the range it aliases, confidently | Beyond `Rs/2M` the estimate is wrong by exactly `Rs/M` with an unchanged confidence — asserted, not left to be discovered | ✅ |
| No line, no confidence | Circular noise returns an argmax like anything else, and a confidence a fifth of a real one | ✅ |
| The reference design needs both | Detuned 200 MHz, the shipped link runs 3334 errors in 3968; frequency recovery alone leaves 13.0 % EVM, both stages return it to the co-tuned 7.34 % | ✅ |
| **Kernels never touch NumPy** | Every kernel run against an array library that refuses NumPy's *allocating* API and returns identical answers | ✅ |
| A second library gets the same field | `check_device()` propagates an N=1 soliton on each back-end and compares — the one answer that is known without a second run | ✅ |
| The GPU job cannot vanish quietly | A test reads `ci.yml` and holds it to the runner label, the CuPy install and the cross-check | ✅ |
| **Parallel sweeps are invisible** | Bit-identical at 1, 2, 3, 4 and one-per-core workers, in sweep order; derived seeds unchanged | ✅ |
| A failing point does not hang | More points than lanes, so a borrowed graph must come back — verified by deleting the `finally` and watching it deadlock | ✅ |
| **MMI phase relations** | `SᴴS = I` at N = 1, 2, 3, 4, 5, 8 — even amplitudes with invented phases pass every other check and fail this one | ✅ |
| 2×2 MMI ≡ 3 dB coupler | The same matrix to 1e-15, factor of j included; self-imaging at N = 2 *is* the quadrature relation | ✅ |
| MMI split and imbalance | `1/N` per path; a tilted MMI is lossy and its matrix says so rather than claiming unitarity | ✅ |
| **MZI as a switch** | `sin²(φ/2)` / `cos²(φ/2)` against the assembled circuit, and the sum is 1 across a full turn to 1e-12 | ✅ |
| MZI as an interleaver | `FSR = c / (n_g ΔL)` measured between peaks; `n_eff` would be out by 1.7× | ✅ |
| Balanced MZI has no period | Response flat to 1e-9 across 2 THz — "balanced" means it, not "a period too long to notice" | ✅ |
| **PDK fits** | A polynomial in `λ − λ_ref` [nm], evaluated where you ask; a bare number is a constant | ✅ |
| PDK refuses extrapolation | A C-band coupler fit run to 1310 nm returns **−0.46** — a negative power fraction. `valid_wavelengths` turns that into a sentence | ✅ |
| PDK refuses nonsense | Unregistered component, undeclared parameter, missing cross-section, a cross-section on a lumped device — all at load, all naming the entry | ✅ |
| A kit executes nothing | JSON in, registry lookup out; `"component": "os.system"` is refused like any other unknown name | ✅ |
| Every shipped device builds | At the bottom, middle and top of the kit's own window | ✅ |

Component models are derived from published literature and standards (Agrawal, *Nonlinear Fiber
Optics*; ITU-T G.652 / G.694.1; relevant IEEE 802.3 clauses), cited in each component's
docstring — never from inspection of commercial tools.

## Contributing

The core is small enough that changing it is still cheap, which makes right now the most useful
time to push back on it. Most valuable first:

* **Review the signal model and scheduler** — [`src/maiman/signals.py`](src/maiman/signals.py),
  [`src/maiman/graph.py`](src/maiman/graph.py), and §3–§4 of the
  [architecture document](docs/ARCHITECTURE.md). If something there is wrong, it is far cheaper
  to fix now than after fifty components depend on it.
* **Tell us if this duplicates existing work.** If a project already does this well, that is worth
  knowing before several months go into it.
* **Describe your use case.** Which components, which measurements, what you currently use and
  what frustrates you about it.
* **Add a component.** A component is a Python class with declared parameters and typed ports —
  see [`src/maiman/components/`](src/maiman/components/) for the pattern. Every physics block needs
  a test against a closed-form result; a component without one will not be merged.

Open an issue for any of the above.

```bash
pip install -e ".[dev]" && ruff check . && ruff format --check . && mypy && pytest
```

[`CONTRIBUTING.md`](CONTRIBUTING.md) has the rest: how a component is put together, what this
project refuses and why, and what each of the drift guards means when it fires at you.

The issue templates offer **"a model is wrong"** first, which is deliberate — a physics block that
disagrees with the literature is the most valuable thing anyone can report here, and the form asks
for the reference because that is what makes it actionable. Security reports go privately instead:
[`SECURITY.md`](SECURITY.md) says what does and does not count, and is specific about the one real
attack surface — `maiman serve` executes the graphs it is posted, which is why it binds to loopback
and warns if you tell it not to.

## Installing

```bash
pip install maiman
```

Python 3.11 or newer. The only runtime dependency is NumPy, deliberately — the interface ships
inside the package, so `maiman serve` works from a plain install with nothing else to fetch.

**If that line assumes more than you have**, [Installing and running
it](#installing-and-running-it) is the same thing starting from no Python and no terminal, with
every message you might hit on the way and what it means.

For a checkout, `pip install -e ".[dev]"`. Releases are published with PyPI trusted publishing,
so no API token exists anywhere in this repository; [`RELEASING.md`](RELEASING.md) has the
procedure.

## Citing this

[`CITATION.cff`](CITATION.cff), which GitHub turns into a "Cite this repository" button. It has no
release date because there has been no release — cite the commit you ran. And please cite the
primary sources for whichever physics you leaned on: every component's docstring names the paper
or the standard its model comes from, and those authors did the work this only implements.

## License

[Apache-2.0](LICENSE) — permissive enough for industrial adoption, with an explicit patent grant.

Dependency licences are checked before adoption, not after. The concrete case already identified:
FFTW is GPL-2.0-or-later, so it (and `pyFFTW`) cannot be linked without making the whole project
GPL — pocketfft/`scipy.fft` (BSD) is used instead.

---

**[→ Full architecture & roadmap](docs/ARCHITECTURE.md)**
