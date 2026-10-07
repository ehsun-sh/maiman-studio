---
title: "Sweeps"
description: "Run one graph over a grid of parameter values, with repeats and parallel workers."
---

# Sweeps

BER curves are the primary deliverable of a tool like this, so sweeps were first-class from the
first commit. A sweep runs one graph once for every combination of the parameter values it is
given.

```python
from maiman import sweep

result = sweep(link, {(fiber, "length"): [40.0, 80.0, 120.0]})
result.metric(meter, lambda r: r.power_dbm)   # shape (3, 1): -8, -16, -24 dBm
```

## Axes

`axes` maps `(component, parameter)`, or `(label, parameter)`, to the values it should take. With
more than one axis, the sweep covers their cartesian product, varying the last axis fastest. Any
parameter of any component can be an axis, including flags.

`fixed` applies the same override at every point, which saves editing the graph just to hold
something constant for one study.

## Repeats

`runs=` repeats each point with independently derived seeds. That is what turns a noisy measurement
into a distribution: a single BER estimate at a marginal operating point is one sample, not an
answer. The studio draws the spread at each point for the same reason.

## Reading the result

A `SweepResult` holds every point in sweep order. Two accessors cover almost every use:

- `result.axis(component, parameter)`: the values one axis took, in sweep order.
- `result.metric(component, extract)`: one number out of every run, shaped `(points, runs)`.
  `extract` receives that component's result, so `lambda r: r.power_dbm` for a power meter, or a Q
  or a BER for an analyser.

Each `SweepPoint` also carries its `values` and the full `Results` of every run, for anything the
accessors do not reach.

## Parallel points, and why the answer does not know

`workers=4` runs four points at a time, each on its own deep copy of the graph; `workers=None` picks
one per core, capped at the number of points. Measured, that is 3.2× on an eight-point nonlinear
span sweep, and **bit-identical** to the sequential result, in sweep order, however many workers
ran it.

That last part is the whole claim. A sweep whose numbers moved with the thread count would be
worthless as a measurement, and the failure would not look like a crash: it would look like a curve
with the right shape and the wrong values. Every point is a function of its own overrides and its
own derived seed, and there is a test on exactly that at 1, 2, 3, 4 and one-per-core workers.

**Threads, not processes.** Measured, the two are the same speed on this work (3.25× against 3.14×
on twelve workers), because the time goes into NumPy's FFTs, which release the GIL, and a split
step is memory-bound long before it is core-bound. Threads then win on everything else: nothing to
pickle, no interpreter to start, and no objection to a component defined in a notebook. The copies
are what make it safe: a run writes a point's overrides onto the components and rolls them back
after, so two threads sharing one graph would produce numbers belonging to neither point.

**The default is one worker**, because a sweep of four cheap points is slower to parallelise than
to run. The graph is left exactly as it was found, including if a run raises; above one worker it
is never run at all, only copied. The session server uses four.

## In the studio

Pick a block and a parameter, give it a range, and the curve appears beside the form that made it.
The server sends numbers, not pictures: a curve is made of scalars, and shipping a histogram at
every point would be megabytes to draw a line. Seven points of a coherent link is 6 kB. The plot
opens on the metric that moved most, which is usually the one the sweep was run to watch. See
[The studio](studio.md#sweeps).
