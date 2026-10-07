---
title: "Quickstart"
description: "Three short scripts: a link budget, two channels sharing a fibre, and a sweep."
---

# Quickstart

Three short scripts, each one a step further: a link budget, two channels sharing a fibre, and a
sweep. Every output shown here is what the current release prints.

You need `pip install maiman` and nothing else. See [Installing](install.md) if that line assumes
more than you have.

## A first link

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

Run it with `python first.py`:

```
received power: -16.00 dBm
```

0 dBm launched, 80 km at 0.2 dB/km, so −16.00 dBm out. Nothing in that number is a lookup: the
laser makes a field, the fibre attenuates it, the meter integrates it.

Four things happened, and they are the whole shape of the API:

- A `SimulationContext` fixed the parameters every block must agree on: the bit rate, the
  oversampling, the length of the time window, the seed. They belong to the run, not to any one
  signal.
- A `Graph` collected the components. `add` returns the component, so it can be assigned in one
  line.
- `chain` wired single-input, single-output blocks in order. Anything more involved uses
  `connect`.
- `run()` executed every block once, in dependency order, and returned the results, addressed by
  the component that produced them.

## Two carriers on one fibre

A combiner has two inputs, so it is wired port by port; `mux["in0"]` is a reference to one named
port:

```python
from maiman.components import Combiner

g = Graph(ctx)
ch1 = g.add(CWLaser(wavelength=1550.0, label="ch1"))
ch2 = g.add(CWLaser(wavelength=1551.0, label="ch2"))
mux = g.add(Combiner(2))
span = g.add(Fiber(length=80.0, attenuation=0.2))
pm = g.add(PowerMeter())

g.connect(ch1, mux["in0"])
g.connect(ch2, mux["in1"])
g.chain(mux, span, pm)

print(g.run()[pm])
```

```
PowerReading(-12.990 dBm; 1551.00nm=-16.000dBm, 1550.00nm=-16.000dBm)
```

Two carriers stay two independently sampled bands, each with its own centre frequency, so the
channel spacing never enters the sample rate. Put the lasers 6 THz apart instead of 125 GHz and
nothing about the run changes, which is exactly what a single-carrier signal model cannot do.
[The signal model](signal-model.md) explains why.

## A sweep

A single run answers *what does this link do*; a sweep answers *how far can it go*. Back on the
link from `first.py`, axes are keyed by `(component, parameter)`:

```python
from maiman import sweep

result = sweep(link, {(fiber, "length"): [40.0, 80.0, 120.0]})

print(result.axis(fiber, "length"))
print(result.metric(meter, lambda r: r.power_dbm))
```

```
[ 40.  80. 120.]
[[ -8.00000043]
 [-16.00000179]
 [-24.00000162]]
```

`metric` is shaped `(points, runs)`. Pass `runs=5` to repeat each point with independently derived
seeds, and `workers=4` to run points in parallel; the answer is bit-identical however many workers
ran it. [Sweeps](sweeps.md) has the rest.

## Next

- Open the same engine in a browser: `maiman serve`, and [the studio guide](studio.md).
- Read how ports, labels, overrides and errors work in [Graphs, ports and runs](graphs.md).
- Find a block and its parameters in [the component reference](components/index.md).
- Run a real link: the scripts in
  [`examples/python/`](https://github.com/ehsun-sh/maiman-studio/tree/main/examples/python), such
  as `ook_link.py` and `coherent_link.py`.
