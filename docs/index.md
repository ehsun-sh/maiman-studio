---
title: "Maiman Studio documentation"
description: "An open-source, modular simulator for optical communication links and photonic systems."
---

# Maiman Studio documentation

Build a link as a block diagram, in the studio or in Python, run it, and read back eye diagrams,
bit error rates, OSNR and constellations, with correct optical noise accounting.

- **One document.** The canvas, the inspector and the engine all work from the same `.maiman`
  file, so the picture and the physics cannot disagree.
- **Validated physics.** Every physics block ships with a test against a closed-form result. See
  [Validation](validation.md).
- **A signal model that scales.** Each carrier is its own sampled band and ASE is carried in
  spectral bins, so a WDM comb never needs a sample rate no machine can afford. See
  [The signal model](signal-model.md).
- **Typed ports.** Optical, electrical, binary, symbol, soft and metric: an invalid wire is
  refused while it is being drawn.
- **One dependency.** NumPy is the only runtime requirement; CuPy is optional for long split-step
  runs.

## Where to start

| Using Maiman Studio | Developing | API |
| :--- | :--- | :--- |
| [Install it](install.md), then the [quickstart](quickstart.md): a power budget in five lines and a two-channel link in ten. Then [the studio](studio.md). | [Architecture](ARCHITECTURE.md), [design decisions](design.md), [contributing](../CONTRIBUTING.md) and [writing a component](writing-components.md). | [The `maiman` package](api.md) and [every component](components/index.md), generated from the code. |

## Citing and support

The source is at [github.com/ehsun-sh/maiman-studio](https://github.com/ehsun-sh/maiman-studio),
under Apache-2.0. A physics block that disagrees with the literature is the most valuable report
anyone can make; [open an issue](https://github.com/ehsun-sh/maiman-studio/issues) with the reference.
