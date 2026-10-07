---
title: "Photonic circuits"
description: "S-matrix circuit solves, devices, two modes per waveguide, PDKs and gdsfactory netlists."
---

# Photonic circuits

A fibre link is a chain: each block takes a waveform and returns one. A photonic integrated circuit
is not. Light in a ring goes round, comes back to the coupler it entered by, and interferes with
itself; there is no order to run the blocks in, because every port's answer depends on every other
port's at once.

## A different kind of solve

So circuits are solved differently. Every device is a **scattering matrix**: for unit amplitude
into port *j*, `s[i, j]` is what leaves port *i*. A circuit is which ports are wired to which. The
ports split into external and internal, and the internal ones are eliminated in one linear solve,
bidirectional, so reflections and counter-propagating light are handled without special cases.

The reduction was cross-validated against [SAX](https://github.com/gdsfactory/sax) (same models,
same wiring) to about **5e-15** over 4001 frequencies, and in CI against a different algorithm that
sums round trips lap by lap, sharing no code, to 1e-13. The closed forms from Yariv and Bogaerts
live in the tests, on the other side of the comparison, where they can disagree. The details are
in [Models and results](models.md#a-circuit-is-not-a-chain).

## Devices

- [Waveguide](components/Waveguide.md): delay, phase and loss, with group delay `n_g L / c` read
  off the phase slope to 1e-9.
- [Directional coupler](components/DirectionalCoupler.md): unitary at every split ratio, which is
  what the cross path's factor of *j* is for.
- [Ring resonator](components/RingResonator.md): all-pass and add-drop, assembled from a coupler
  and two arcs rather than written down.
- [MMI](components/MMI.md): N×N on the self-imaging phase relations; a 2×2 MMI *is* a 3 dB
  coupler, factor of *j* included.
- [Mach-Zehnder interferometer](components/MachZehnderInterferometer.md): two couplers and two
  paths, a switch, an interleaver, or both.
- [Edge coupler](components/EdgeCoupler.md) and [grating coupler](components/GratingCoupler.md):
  getting light onto the chip, often the largest line in a photonic link budget.

**A solved circuit is a transfer function**, and a transfer function is something the link engine
has known what to do with since its first optical filter. So a ring drops into a fibre link as an
ordinary block, and passes both the signal and the ASE that arrives with it, integrated across its
linewidth, not read at one frequency.

## Two modes in one waveguide

A 220 nm silicon strip confines TE strongly and TM weakly, so the two modes have different
effective and group indices: 2.44/4.20 against 1.78/3.80 at 1550 nm. Each guided polarization
carries its own indices through the same reduction, and a ring then resonates at **two** sets of
wavelengths on two free spectral ranges. The solver needed no changes for it: ports are identified
by name, so two modes are simply twice as many ports.

## A foundry's numbers: PDKs

The engine knows what a directional coupler *is*. It cannot know that a given process makes a
nominal 3 dB coupler that measures 0.48 at 1550 nm and drifts across the C band. Those numbers come
off a wafer, and a `.pdk` kit carries them: a JSON file of fitted polynomials in `λ − λ_ref`, each
with the window it was fitted in.

```python
from maiman import load_pdk

kit = load_pdk("silicon_220nm.pdk.json")   # JSON in, validated PDK out, nothing executed
```

- **It refuses to extrapolate.** A C-band coupler fit run to 1310 nm returns −0.46, a negative
  power fraction. `valid_wavelengths` turns that into a sentence instead of a number.
- **It refuses nonsense at load:** an unregistered component, an undeclared parameter, a missing
  cross-section; each error names the entry.
- **It executes nothing.** JSON in, a registry lookup out; `"component": "os.system"` is refused
  like any other unknown name.

## What a layout tool drew: netlists

A layout tool knows the topology and the geometry, and nothing about what any of it does optically:
gdsfactory's cross-sections carry width, layer and bend radius and forbid anything else. So the
join runs the other way: the netlist says what is wired to what and how long the router made each
piece, the kit says what each cell is and what the process measures, and the circuit solve does the
rest.

```python
import numpy as np
from maiman import circuit_from_netlist, load_netlist, load_pdk

kit = load_pdk("silicon_220nm.pdk.json")
netlist = load_netlist("ring_racetrack.netlist.yml")

frequencies = np.linspace(192.6e12, 194.2e12, 200001)
solved = circuit_from_netlist(netlist, kit, frequencies).solve()
through = solved.power("through", "in")
```

`circuit_from_netlist` returns an unsolved circuit, so you can add to it (a source, a facet, a
device the layout does not carry) before solving. It reads gdsfactory's YAML when PyYAML is
installed, and JSON always; nothing imports gdsfactory. Measured, importing it would resolve to 86
packages and require klayout, which is GPL-3.0-or-later.

The file gdsfactory itself ships as a sample, solved unmodified: **−0.0053 dB**, against the
26.637 µm at 2 dB/cm anyone can work out by hand. A cell or port the kit does not map is refused by
name, because skipping either would return a circuit that solves and is not the one drawn.

The full walk-throughs are `netlist_circuit.py`, `pdk_import.py`, `microring_filter.py` and
`birefringent_ring.py` in
[`examples/python/`](https://github.com/ehsun-sh/maiman-studio/tree/main/examples/python).
