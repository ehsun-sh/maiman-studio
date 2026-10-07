---
title: "Graphs, ports and runs"
description: "Components, typed ports, wiring, runs, results, overrides, seeds and loops."
---

# Graphs, ports and runs

A simulation is a directed graph of components. Each component is a pure function of its inputs
and its parameters; the graph says which output feeds which input; a run executes every component
once, in dependency order, over the whole time window.

## Components and labels

Construct a component with its parameters as keyword arguments, in the units the
[reference](components/index.md) lists for them, and add it to a graph. `Graph.add` returns what it
was given:

```python
fiber = link.add(Fiber(length=80.0, attenuation=0.2))
```

Parameters are declared once, in the component's class, with their unit and their valid range, and
they are checked at construction, with messages that say what went wrong:

```
Fiber(length=-1)   →  ValueError: length=-1.0 km is below the minimum 0.0
Fiber(lenght=1)    →  TypeError: Fiber has no parameter(s) ['lenght']; declared: ['attenuation', …]
```

Every component gets a label, which is how results, overrides and project files name it. Unless you
pass `label=`, it is the class name and a counter: `CWLaser1`, `Fiber1`, `PowerMeter1`.

## Typed ports

A link carries more than optical fields, so every port has a type, and a wire between two
different types is refused before anything runs:

| Type | Carries | For example |
| :--- | :--- | :--- |
| optical | Sampled bands plus noise bins; see [the signal model](signal-model.md) | Laser → MZM → fibre |
| electrical | A real waveform | Driver → MZM's RF port; photodiode → filter |
| binary | Bits | PRBS → NRZ driver |
| symbol | Complex symbols | QAM mapper → IQ driver |
| soft | Log-likelihood ratios | Soft demapper → soft FEC decoder |
| metric | A measurement | BER analyser, power meter, diagnostics |

An MZM therefore has two inputs, one optical and one electrical. And `soft` exists as its own type
because wiring LLRs into a block that expects bits would not crash: it would run and quietly give
back the decibels soft decoding exists to recover.

```
GraphError: cannot connect PRBSGenerator1.out (binary) to Fiber1.in (optical): port types differ
```

## Wiring

`chain(a, b, c)` connects a sequence of single-input, single-output components in order. For
anything with more than one port, `connect(src, dst)` takes components or named ports;
`block["name"]` is a reference to one port:

```python
g.connect(ch1, mux["in0"])
g.connect(ch2, mux["in1"])
g.chain(mux, span, meter)
```

Port names are on each component's reference page, with their types. An input takes exactly one
connection. An output may feed as many inputs as you like, and each of them receives the *same*
signal, because signals are immutable and shared, not divided. To split optical power, put a
`Splitter` in the path.

## Running

Execution is **block-mode**: each component is invoked once per run and processes the entire time
window in one call. That keeps every component a pure function and makes vectorised NumPy or CuPy
code straightforward.

- The graph is validated first: port types, connectivity, and cycles. `PowerMeter1.in is not
  connected` is reported before anything runs.
- Components execute in dependency order.
- Intermediate signals are released as soon as their last consumer has run, so peak memory is the
  width of the graph cut, not the whole graph.
- Metric outputs, the outputs of sink components, and anything named in `keep=` are retained and
  returned.

## Results

`run()` returns a `Results` object. Index it by a component for that component's sole output, or
ask for a named port:

```python
res = link.run()
res[meter]                        # PowerReading(-16.000 dBm; 1550.00nm=-16.000dBm)
res.port(meter, "out")            # the same reading, by name
[key for key, _ in res.items()]   # [('Fiber1', 'diagnostics'), ('PowerMeter1', 'out')]
```

The fibre's `diagnostics` port is there because it is a metric: blocks that make choices report
them on a diagnostics port rather than hiding them.

## Overrides and seeds

`overrides` sets parameters for one run only, keyed by `(component or label, parameter)`, and
leaves the graph unchanged:

```python
link.run(overrides={(fiber, "length"): 100.0})[meter].power_dbm   # -20.0
```

`seed` replaces the context's seed for one run, which is how repeated runs draw independent noise
from the same graph. Every stochastic block draws from its own generator, derived deterministically
from the seed and the block's identity, never from a global one.

## Loops

A cycle in a graph is an error, because a cycle cannot be ordered, and for almost everything here
that is the truth: a link goes one way, and a cycle is a wiring mistake. `CycleError` says so.

Real loops exist (a cavity, return loss next to a grating, a recirculating loop), and they are run
by putting a [`Feedback`](components/Feedback.md) block on one wire of the loop. It declares how
many passes to run, carries what reaches it on one pass into the next, and only the last pass's
results are returned. Add a [`DelayLine`](components/DelayLine.md) and each pass becomes a lap one
loop-time after the last. [Models and results](models.md#a-loop-the-engine-will-run) has the
physics.

## Progress

`run(progress=callback)` calls the callback as the run advances, between components and from inside
any component that reports its own progress. It is called from the running thread, so it must be
cheap and must not raise: an exception there aborts a run that was otherwise fine.
