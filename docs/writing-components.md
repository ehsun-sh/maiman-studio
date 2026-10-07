---
title: "Writing a component"
description: "A component is a Python class with declared parameters and typed ports. Here is a whole one."
---

# Writing a component

A component is a Python class with declared parameters and typed ports. Define one and it is
registered; import its module and it is in the palette, in project files and in sweeps, with no
manifest to write, because the manifest is generated from the class.

## A whole component

A fixed tap coupler: it drops a fraction of the power and passes the rest. This runs as written:

```python
from maiman import Component, Param, PortType
from maiman.signals import OpticalSignal

class Tap(Component):
    """A fixed tap coupler: drops a fraction of the power, passes the rest."""

    display_name = "Tap"
    category = "Passive"

    ratio = Param(0.01, min=0.0, max=1.0, doc="Fraction of the power dropped")

    inputs = {"in": PortType.OPTICAL}
    outputs = {"out": PortType.OPTICAL}

    def run(self, ctx, inputs):
        signal: OpticalSignal = inputs["in"]
        keep = 1.0 - self.ratio
        return {
            "out": OpticalSignal(
                bands=tuple(b.scale_amplitude(keep**0.5) for b in signal.bands),
                noise=tuple(n.scale_power(keep) for n in signal.noise),
                accumulated_gvd=signal.accumulated_gvd,
            )
        }
```

Put it in a link like any shipped block:

```python
t = Graph(ctx)
laser = t.add(CWLaser(power=0.0))
tap = t.add(Tap(ratio=0.1))
meter = t.add(PowerMeter())
t.chain(laser, tap, meter)

print(f"{t.run()[meter].power_dbm:.3f} dBm")   # -0.458 dBm, which is 10·log10(0.9)
```

And `maiman.manifests()["Tap"]` now describes it (name, category, the `ratio` parameter with its
default, bounds and doc, and one optical port each way), which is everything the studio's palette
and inspector need.

## The steps

1. **Subclass `Component`** and set `display_name`, `category`, `inputs` and `outputs`. Read a few
   shipped ones first: `CWLaser` for a source, `Fiber` for a channel, `TimingRecovery` for a DSP
   block with a diagnostics port.
2. **Declare parameters** with `Param` or `BoolParam`, each with its real unit and a `doc` that
   says what it does. Use `choices` for a parameter whose legal values are a set, not an interval,
   and `applies_when` for one that only takes effect under a flag.
3. **Implement `run(ctx, inputs)`**, returning one signal per output port. Never mutate an input;
   return new signals, sharing buffers where only metadata changes.
4. **Export it** from `src/maiman/components/__init__.py`, if it belongs in the library.
5. **Write the closed-form test.** Without it the component will not be merged.
6. **Regenerate the studio's baked data** with `python examples/python/export_ui_data.py`, and
   copy `examples/python/ui_data.json` into the `<script id="maiman-data">` block of
   `src/maiman/studio/index.html`.
7. **Regenerate the reference** with `python tools/gen_reference_docs.py`, which writes its page
   under `docs/components/`. The drift guards tell you if you forget either.

If the component emits a new kind of measurement, it also needs an encoder in
`src/maiman/encoding.py` and a case in the studio's log, so the log prints a fact rather than the
name of a type.

## Units

A unit is part of a parameter's declaration, not a comment, and it is checked against a closed set:
`Param(1.0, unit="%")` is refused at import with the list of units that exist. Read a value in its
declared unit as an attribute (`self.attenuation`) or converted to SI with
`self.si("attenuation")`, which is where every conversion happens, once. Unit confusion is the most
common source of wrong results in this field, so it is handled in one place.

```
ValueError: unknown unit '%'; known units: ['', '1/K', '1/W/km', 'A', 'GHz', 'Hz', 'K', 'MHz', …]
```

## Registration

Defining the class registers it under its type name. Re-registering the identical class is
harmless, so re-importing a module is fine; two *different* classes claiming one name is an error,
because it would make a project file ambiguous and the same file would simulate differently
depending on import order. Set a distinct `registry_name` if you must shadow a name.

A third-party package of components needs nothing more: once it is imported, its blocks are
available to graphs, projects and sweeps. A project that names them opens after the package is
imported, and fails with an error saying so before.

## Before you open the pull request

- The docstring cites the paper, equation and edition the model comes from.
- There is a test against a closed-form result, and it would fail if the model were wrong.
- Every parameter moves something. If one only matters under a flag, `applies_when` says so.
- Idealisations are flags with defaults, and the defaults keep every existing result where it was.
- The one command in [CONTRIBUTING.md](../CONTRIBUTING.md) passes, whole.
