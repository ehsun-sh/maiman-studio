---
title: "The session server"
description: "maiman serve: the small HTTP server that hands the studio its page and runs the graphs it posts."
---

# The session server

`maiman serve` starts a small HTTP server that hands the studio its page and runs the graphs it
posts. No dependencies beyond the engine's own, bound to loopback.

## Routes

| Route | What it does |
| :--- | :--- |
| `GET /api/manifests` | Every registered component: parameters with units and bounds, ports with types |
| `POST /api/run` | A `.maiman` [project document](projects.md) in, results out |
| `POST /api/sweep` | A project, an axis and a range in; a curve out, as numbers |
| `POST /api/spectrum` | One block's S-matrix across a frequency window, for the studio's S-matrix tab |
| `GET /api/health` | Whether it is up, and how many components it knows |

## An ordinary client of the Python API

Every route is a thin wrapper over something a script can already call: `manifests()`,
`graph_from_dict()`, `Graph.run()`, a block's `spectrum()`. Nothing in the server understands the
physics and nothing in the engine knows the server exists, which is what stops the two drifting
apart: a feature unreachable from Python is unreachable from the interface too.

## Reduction happens in the engine

A run holds tens of thousands of samples per port, and an eye diagram drawn from them is a 96×96
histogram. So a signal-carrying port encodes to a *summary* (how many samples, at what rate, how
much power) and anything meant to be looked at arrives already reduced, from a measurement
component the graph contains explicitly. A full coherent run is **52 kB of JSON**.

That is not a size optimisation: it is what keeps a second, untested implementation of the physics
from growing in JavaScript.

- Every encoded value carries a `kind`, so a client switches on a string rather than guessing from
  which fields happen to be present. A result type nothing knows how to draw, such as a plugin's
  own metric, arrives tagged `opaque` rather than failing the run; a test asserts that is never the
  answer for anything shipped here.
- Non-finite numbers become `null`. `json.dumps` emits bare `NaN` and `Infinity`, which are not
  JSON and which `JSON.parse` rejects, so one infinite Q factor would fail a whole response.
- Runs stream progress, which is what draws the studio's progress bar from real engine work.

## Errors say whose fault they are

| Status | Meaning | For example |
| :--- | :--- | :--- |
| `400` | The request could not be read | `no component registered as 'FluxCapacitor'. If it comes from a plugin…` |
| `422` | It was read, and will not run | `pm.in is not connected` |
| `413` | Too big to attempt | `a window of 32000000 samples exceeds the server limit of 4194304` |

## Loopback, and why

`POST /api/run` executes the graph it is given, so the socket binds to `127.0.0.1` unless a host is
passed explicitly, and warns when one is. A run is refused *before* it starts if its window is too
large. The registry does the harder half: a project may only **name** components that are already
registered and can never import a dotted path, so posting someone else's project is not equivalent
to running their code. [SECURITY.md](../SECURITY.md) has the details.

Use `--port` when 8765 is taken: `maiman serve --port 8766`.
