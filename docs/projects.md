---
title: "Project files"
description: "The .maiman format: a versioned, human-readable, git-diffable JSON document."
---

# Project files

A `.maiman` file is the link itself: a versioned, human-readable, git-diffable JSON document. The
studio's canvas draws it, the session server runs it, and `maiman.load` turns it back into a graph:
the same object in all three places.

## Saving and loading from Python

```python
from maiman import load, save

path = save(link, "budget.maiman")
again = load(path)          # a runnable Graph
```

That writes the link from the [quickstart](quickstart.md) as:

```json
{
  "schema_version": 1,
  "maiman_version": "0.17.0",
  "context": {
    "bit_rate": 10000000000.0,
    "samples_per_symbol": 16,
    "sequence_length": 64,
    "seed": 0,
    "precision": "single"
  },
  "nodes": [
    { "id": "CWLaser1", "type": "CWLaser", "params": { "power": 0.0, "wavelength": 1550.0 } },
    { "id": "Fiber1", "type": "Fiber", "params": { "length": 80.0, "attenuation": 0.2 } },
    { "id": "PowerMeter1", "type": "PowerMeter", "params": {} }
  ],
  "edges": [
    { "from": ["CWLaser1", "out"], "to": ["Fiber1", "in"] },
    { "from": ["Fiber1", "out"], "to": ["PowerMeter1", "in"] }
  ]
}
```

Shown compacted; `save` writes one key per line, which is what makes a diff of two versions of a
link readable.

## The format

| Key | What it holds |
| :--- | :--- |
| `schema_version` | The version of the format, present since the first commit so it can change without guessing |
| `maiman_version` | The engine that wrote the file |
| `context` | The [SimulationContext](signal-model.md#the-run-owns-the-time-window): bit rate, oversampling, sequence length, seed, precision |
| `nodes` | One entry per component: its label as `id`, its registered `type`, and the parameters that were set on it. Anything never set is left out and takes its default. A node's `ui` holds its position on the canvas, when the studio saved it |
| `edges` | Wires, from `[label, output port]` to `[label, input port]` |
| `notes` | Optional: the lesson the studio opens beside the canvas, `{"title", "body"}`, the body Markdown with TeX between dollar signs |
| `annotations` | Optional: the sticky notes, frames and arrows drawn over the schematic |

Only explicitly set parameters are stored, so a file records the choices its author made, not the
defaults they accepted. The trade-off is real: if a model's default changes in a later release, a
file that never overrode it will simulate slightly differently, and `maiman_version` is recorded
so that is diagnosable. The engine never reads `notes` or `annotations`, so a project with them
runs exactly as it would without.

A project must be runnable headless with no interface installed. Every shipped project is tested
for exactly that: it loads, it runs, and it carries its canvas layout.

## Opening a file never runs code

A project file only ever **names** components. Resolving those names by importing a dotted path out
of the file would make opening someone else's project equivalent to running their code, because
importing a module executes it. So a name is only looked up among components that are already
registered, and a name that is not registered is an error that says what to do:

```
UnknownComponentError: no component registered as 'FluxCapacitor'. If it comes from a plugin
package, import that package before loading the project.
```

Components from a third-party package register themselves when the package is imported, so import
it first and the project opens.

## Retired parameters

When a model stops using a parameter, the parameter is retired rather than left as a control that
moves nothing. Older projects that still carry it keep opening, with the value dropped, and the
component's code says whether dropping it changes the link the file describes. Usually it does
not. Once (the EDFA's old `max_output_power` clamp) it did, and that is stated rather than glossed.

## In the studio

**File → Save** and **File → Open** go through the browser: the file is chosen in your operating
system's own picker and read locally, and nothing is posted. A server that opened or wrote a path
it was handed would be a worse program and would gain nothing: the file belongs to the person at
the keyboard.

The Examples menu opens the projects in
[`examples/maiman/`](https://github.com/ehsun-sh/maiman-studio/tree/main/examples/maiman), each
written by a script in `examples/python/`.
