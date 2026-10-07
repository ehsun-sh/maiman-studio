# Getting started

[← README](../README.md) · [Getting started](getting-started.md) · [The studio interface](interface.md) · [Models and results](physics.md) · [Design and roadmap](design.md) · [Validation](validation.md)

---

## Installing and running it

**This section assumes you have never used Python or a terminal.** If you have, the whole thing is
`pip install maiman` then `maiman serve` — skip to [The interface](interface.md#the-interface).

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
Successfully installed maiman-0.17.0 numpy-2.3.5
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
  77 components
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
4. **The Examples menu opens complete projects**, one submenu per phase of
   [the examples roadmap](EXAMPLES_ROADMAP.md) and numbered as there: first light (a laser on
   a meter, a loss budget, the Mach–Zehnder's cosine, a pulse spreading, compressing and travelling
   as a soliton on the oscilloscope, splitters and couplers), then direct-detection links (an eye
   diagram, PIN against APD sensitivity, a DML's chirp limiting its reach, mode partition noise, a
   PAM4 lane with an equaliser, CWDM4 in the O-band and a 1:32 GPON downstream), amplified and WDM
   links (one EDFA on a spectrum, the OSNR of a chain of spans, a WDM spectrum, an eight-channel
   DWDM link, the optimum launch power, four-wave mixing on G.653 fibre and a ROADM add/drop
   node), and coherent links with the full DSP chain and with soft-decision FEC. Each is a `.maiman` file in `examples/maiman/`, written by a script in
   `examples/python/`.
5. **Read the notes.** Every project the Examples menu opens comes with a lesson in a panel on the
   left of the canvas: what the link is for, the formulas behind each stage, and what to try next. A
   block named in the lesson is a link: click it and the block is selected on the canvas. The
   toolbar's note, frame and arrow tools put your own explanations on the canvas, and **Notes →
   Edit** writes a lesson of your own; both are saved in the `.maiman` file.
6. **File → Save** writes a `.maiman` file — a plain text description of your link that you can
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

### The desktop application

The same studio also comes as a desktop application for Windows, macOS and Linux: one window, no
terminal, nothing to install first. It starts its own copy of the session server on a free port
when it opens and stops it when it closes, so everything above about `maiman serve` happens out of
sight. Installers are built by the **Desktop** workflow and attached to each GitHub release. They
are not code-signed yet; [desktop/README.md](../desktop/README.md) says how to open one anyway, and
how to build it yourself.

From a checkout with Node.js installed, double-click `start-studio.bat` (Windows),
`start-studio.command` (macOS) or `start-studio.sh` (Linux) in the repository root to open it.

Without Node.js, run `python tools/make_shortcut.py` once. It puts a **Maiman App** shortcut with
the Maiman icon in the repository root; double-clicking it opens the studio in your browser using
that same Python, and the small window it shows stops Maiman when you close it.

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

The [`examples/python/`](../examples/python/) folder has two dozen more, each one runnable with
`python examples/python/<name>.py`; the projects the studio opens are beside it in
[`examples/maiman/`](../examples/maiman/). [The examples roadmap](EXAMPLES_ROADMAP.md) lists the
ones still to build, from a laser on a power meter to a submarine cable, in the order we mean to
build them.

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
