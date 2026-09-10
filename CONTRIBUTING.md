# Contributing

The core is small enough that changing it is still cheap, which makes right now the most useful
time to push back on it. The [README's Contributing section](README.md#contributing) says where
help is most valuable; this file says how the project actually works, and what it refuses.

## The one command that decides

```bash
pip install -e ".[dev]" && ruff check . && ruff format --check . && mypy && pytest
```

Run **that**, not a narrower version of it. Two red builds in this repository's short history came
from running a subset: `mypy src tests examples` instead of bare `mypy`, and `ruff format tests`
instead of `ruff format .`. Both passed locally and both failed in CI, because the command that
decides is the whole one. `mypy` takes no arguments here on purpose — its configuration already
names what to check.

## What this project refuses

These are not style preferences. They are the reason the simulator is worth trusting, and a change
that breaks one will not be merged however good the code is.

**Every number shown must come from the engine.** Not from a plausible-looking constant, not from
a curve fitted to a screenshot, not from a figure remembered from a datasheet. If the interface
displays it, a component computed it.

**Every physics block ships with a test against a closed-form result.** Not "the code runs" — a
comparison against an analytical answer the block cannot have been written to agree with by
accident. See [`tests/test_physics.py`](tests/test_physics.py). A component without one will not be
merged, and this is the rule most often argued with.

**Models come from published literature and standards, cited in the docstring — never from
inspection of commercial tools.** Comparing against a licensed tool is not reproducible in CI and
proves nothing to anyone who cannot afford the licence. Name the paper, the equation number, and
the edition.

**Derive rather than transcribe, where you can.** A generator polynomial's *roots* are the claim; a
table of its coefficients is a copy that no test can check. Where a table genuinely has to be
present — the primitive polynomials in [`src/maiman/softfec.py`](src/maiman/softfec.py) — the code
verifies it at construction, so the table is an assertion rather than an assumption.

**Never ship a claim you cannot back.** If a model does not work, say so in the docstring, write a
test that pins the failure, and do not tune it until a curve looks plausible. There is precedent in
this repository for exactly that, including the commit that had to correct its own diagnosis.

**The interface never shows a control it cannot back.** A parameter that moves nothing is worse
than a missing feature, because the user cannot tell. If a model stops using a parameter, retire it
— see `retired_parameters` in [`src/maiman/component.py`](src/maiman/component.py) — and say in the
comment whether dropping it changes the link a saved project describes. Usually it does not. Once,
it did, and the comment says so.

**An idealisation is declared, not delivered by accident.** A laser with no intensity noise, an
amplifier that never compresses — these are fine and useful, and they are flags with defaults, not
silence. What is not fine is a clamp that calls itself a saturation model.

## Adding a component

A component is a Python class with declared parameters and typed ports. Read a few in
[`src/maiman/components/`](src/maiman/components/) first — `CWLaser` for a source, `Fiber` for a
channel, `TimingRecovery` for a DSP block with a diagnostics port.

1. Subclass `Component`. Set `display_name`, `category`, `inputs` and `outputs`.
2. Declare parameters with `Param` / `BoolParam`, each with its real unit and a `doc` that says
   what it does. Units convert once, at the boundary, via `si()`; see
   [`src/maiman/units.py`](src/maiman/units.py).
3. Implement `run(ctx, inputs) -> dict[str, Signal]`.
4. Export it from [`src/maiman/components/__init__.py`](src/maiman/components/__init__.py).
5. Write the closed-form test.
6. Regenerate the interface's baked data and splice it back into the page:

```bash
python examples/export_ui_data.py
```

Then copy `examples/ui_data.json` verbatim into the `<script id="maiman-data">` block of
`src/maiman/studio/index.html`. The tests below will tell you if you forget.

If your component emits a new kind of measurement, it also needs an encoder in
[`src/maiman/encoding.py`](src/maiman/encoding.py) and a `case` in the studio page's `describe()`
so the log prints a fact rather than the name of a type.

## The guards, and what they mean when they fire

Several tests exist only to catch drift between parts that have no other way to disagree. They are
not obstacles; each one has caught a real mistake, most of them more than once.

| Test | Fires when | Do this |
| :--- | :--- | :--- |
| `test_the_palette_is_the_whole_library` | A component is registered but not in the baked export | Re-run the export and splice |
| `test_the_baked_manifests_are_the_live_ones` | A *parameter* changed since the last export | Same — this one catches what the palette test cannot |
| `test_every_kind_the_engine_can_send_has_a_line_to_print` | A new encoded kind has no `case` in `describe()` | Give the log the fact the payload carries |
| `test_no_metric_port_in_the_library_encodes_as_opaque` | A new metric port is never exercised | Add a graph in `tests/test_server.py` that runs it |
| `test_every_block_in_the_project_has_somewhere_to_be_drawn` | A block in the shipped project has no canvas position | Add it to `LAYOUT` in the studio page |
| `test_the_schematic_on_the_canvas_is_a_graph_that_runs` | The shipped project does not execute | The page is showing a link that cannot exist — fix the project |
| The test-count floor in `test_packaging.py` | The suite shrank below what the docs claim | Either you deleted tests, or a doc needs updating |

## Documentation

`README.md` is the argument for the project and carries its measured results. `PRODUCT.md` is who
it is for. `DESIGN.md` is the interface's reasoning, including a "Not done" section that is kept
honest. `docs/ARCHITECTURE.md` is the engine's.

Numbers in those files are **measured**, and several of them are pinned by tests. If you change a
model, re-measure rather than adjusting the prose to match your intuition.

## Commits

Explain *why*, and what you learned. A commit here is expected to say what was measured, what was
surprising, and what was wrong before — including when the thing that was wrong was the previous
commit's own explanation. That is not ceremony; it is the only durable record of why a model looks
the way it does.

Open an issue before a large change, so nobody spends a weekend on something that was already
decided against.
