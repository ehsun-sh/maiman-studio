# Design and roadmap

[← README](../README.md) · [Getting started](getting-started.md) · [The studio interface](interface.md) · [Models and results](physics.md) · [Design and roadmap](design.md) · [Validation](validation.md)

---

## What this is

A block-diagram simulator for optical systems: drop components on a canvas, wire a link, run it,
get an eye diagram and a BER number — with correct optical noise accounting. And the same model,
scriptable from Python.

Think **OptiSystem's workflow, VPIphotonics' ambition, open source, and verifiable.**

```
  PRBS ──▶ NRZ ──▶ CW Laser ──▶ MZM ──▶ Fiber ──▶ PIN ──▶ BER / Eye
                                 ▲
                            (V_π, ER, IL)
```

## Why

There is a real gap in open-source photonics tooling, but it is **not** where it first appears
to be.

The physics kernels already exist. SSFM, coherent DSP, and S-matrix circuit solvers are all
available in open source — as libraries, in fragments, each with its own incompatible data model.
What does not exist is a single coherent tool that puts them behind a usable interface with a
sound signal representation underneath.

So the gap is **integration and user experience**, not numerics. That shapes the whole strategy:
where a mature open-source kernel exists, Maiman Studio wraps or depends on it rather than
rewriting it. The value added here is the data model, the execution engine, the component
library, and the UI.

| Existing tool | What it does | Relationship |
| :--- | :--- | :--- |
| [OptiCommPy](https://github.com/edsonportosilva/OptiCommPy) | Python: SSFM, coherent DSP, BER | Reference & cross-validation target |
| [GNPy](https://github.com/Telecominfraproject/oopt-gnpy) | Optical network planning / OSNR budgets | Complementary — network layer, not waveform layer |
| [QAMPy](https://github.com/ChalmersPhotonicsLab/QAMpy) | Coherent DSP algorithms | Reference for Phase 3 |
| [SAX](https://github.com/gdsfactory/sax) | S-matrix photonic circuit solver | **Cross-validation reference, not a dependency.** The reduction is twelve lines and SAX resolves to 37 packages including an LGPL sparse back-end; the two agree to 5e-15 |
| [gdsfactory](https://github.com/gdsfactory/gdsfactory) | Photonic layout & PDK ecosystem | **Connected, and still not a dependency.** [`maiman.netlist`](../src/maiman/netlist.py) reads the YAML netlists it writes and solves them against a `.pdk`. Nothing imports it: measured, it resolves to **86 packages** and requires `klayout`, which is **GPL-3.0-or-later** — the same ground FFTW and klujax are refused on. Reading a document costs none of that |
| [Meep](https://github.com/NanoComp/meep) | FDTD / full-wave EM | Feeds component models *in*; not a competitor |
| [GNU Radio](https://www.gnuradio.org/) | Block-based SDR | Architectural reference for dataflow scheduling |

## Design decisions

These are the choices that matter, stated up front so they can be argued with. Full reasoning is
in the [architecture document](ARCHITECTURE.md).

| Decision | Rationale |
| :--- | :--- |
| **Engine before GUI** | A wrong engine forces a total rewrite; a wrong GUI does not. |
| **Multi-band optical signal** from day one | A single scalar carrier frequency cannot express a 40-channel DWDM system without a physically impossible sample rate. Discovering that in Phase 3 means rewriting the core. |
| **Noise carried in spectral bins**, separate from sampled fields | ASE spans the amplifier bandwidth; the signal does not. Sampling both together makes realistic runs impossible. |
| **Block-mode execution** (whole time window per call) | Every component becomes a pure function of its inputs. Vectorizes naturally; not a streaming scheduler. |
| **Python-first**, behind a narrow kernel boundary | ~90–95% of SSFM runtime is inside FFT — library code in any language. GPU via CuPy is nearly free. Contributors who write fiber models write Python. Native kernels stay an option, not a prerequisite. |
| **Typed ports** (Optical / Electrical / Binary / Symbol / Metric) | An MZM has an electrical input. Invalid wiring is rejected at edit time, not at run time. |
| **Immutable signals** | A WDM link is hundreds of MB. Value-copying between blocks makes the tool unusable regardless of language. |
| **Python-package plugins**, parameters declared once | Requiring contributors to match a C++ ABI suppresses exactly the contributions the plugin system exists to attract. |
| **Every physics block validated against a closed-form result** | Comparison against commercial tools needs a licence and is not reproducible in CI. A simulator nobody can verify has no scientific value. |
| **No FFTW** | FFTW is GPL-2.0-or-later. Linking it would make the project GPL. pocketfft (BSD) is used instead. |

## Architecture

```text
                    ┌──────────────────────────────────────┐
                    │        Visual Designer (Web UI)      │
                    │   graph editor · plots (WebGL)       │
                    └───────────────────┬──────────────────┘
                                        │  project JSON + WebSocket
                                        │  (data reduced engine-side)
                    ┌───────────────────▼──────────────────┐
                    │            Session Server            │
                    └───────────────────┬──────────────────┘
 ┌──────────────────────────────────────▼───────────────────────────────────────┐
 │                             Public Python API                                 │
 │        maiman.Graph · Component · run() · sweep()                              │
 └──────────────┬─────────────────────────────────────┬─────────────────────────┘
 ┌──────────────▼───────────────┐    ┌────────────────▼─────────────────┐
 │      Component Library       │◄──►│         Execution Engine         │
 │  plugins · registry · schema │    │  scheduler · sweeps · run graph  │
 └──────────────┬───────────────┘    └────────────────┬─────────────────┘
 ┌──────────────▼─────────────────────────────────────▼─────────────────────────┐
 │                     Core Data Model + Numerical Kernels                       │
 │       SimulationContext · Signals · FFT / SSFM / filters / noise              │
 │              back-ends: NumPy → CuPy → (optional) native                      │
 └──────────────────────────────────────────────────────────────────────────────┘
```

The GUI is a client of the public Python API with no privileged access. **If a feature is not
reachable from Python, it does not exist.**

## The core data model

The part most worth reviewing — see [`src/maiman/signals.py`](../src/maiman/signals.py). An optical
signal is not one array of numbers:

```python
@dataclass(frozen=True)
class Band:
    """One sampled band: complex envelope in two orthogonal polarizations (Jones vector)."""

    Ex: np.ndarray  # complex64, shape (N,), read-only
    Ey: np.ndarray
    f0: float  # band centre frequency [Hz]
    fs: float  # band sample rate [Hz]


@dataclass(frozen=True)
class NoiseBin:
    """Spectrally-resolved noise, carried separately from the sampled bands."""

    f_start: float
    f_end: float
    psd_x: float  # [W/Hz] per polarization
    psd_y: float


@dataclass(frozen=True)
class OpticalSignal:
    bands: tuple[Band, ...]
    noise: tuple[NoiseBin, ...]
    accumulated_gvd: float    # sum(beta2 * L) over the path so far [s^2]
```

Fields are `sqrt(W)`, so instantaneous power is `|Ex|**2 + |Ey|**2`. Arrays are read-only, which
is what lets metadata-only blocks share buffers instead of copying a span at a time.
`accumulated_gvd` is the one piece of *path* state on the signal, and it is there so that mixing
products generated in different spans can be added as fields — see
[above](physics.md#mixing-products-add-in-field-not-in-power).

Global run parameters (bit rate, oversampling, sequence length, RNG seed) live in a shared
`SimulationContext`, not in individual signals — so blocks cannot silently disagree about the
time window, and results are reproducible.

## Roadmap

| Phase | Scope | Estimate¹ |
| :--- | :--- | :--- |
| **0 — Foundations** ✅ | Signal model, context, port types, component base, registry, scheduler, `.maiman` project format, sweeps, CI | ~1 month |
| **1 — MVP: linear link** ✅ | ✅ PRBS → NRZ → laser → MZM → fiber (α + CD) → PIN → filter → eye/Q/BER, validated end to end. **Python only, no GUI.** | ~2–3 months |
| **1.5 — Nonlinear & amplified** ✅ | Adaptive-step SSFM, Kerr, EDFA with ASE and Saleh gain compression, erbium gain dynamics on their own time axis, OSNR, PMD, APD, dispersion slope and its third-order term, cross-polarization Kerr coupling, inter-channel stimulated Raman scattering | ~2 months |
| **2 — Coherent transceiver** ✅ | Gray-coded M-QAM to 256, IQ modulator with bias and quadrature error, 90° hybrid, balanced detection, blind carrier frequency and phase recovery, coarse frequency acquisition over the whole sampled band, blind square-law timing recovery, dual polarization with a blind butterfly equaliser, root-raised-cosine shaping and matched filtering, differential quadrant encoding, receiver-side dispersion compensation over spans to 1000 km with blind estimation of the accumulated value, EVM/MER, constellation diagram, validated against closed-form SER | ~3 months |
| **3 — GUI & WDM** ✅ | Wavelength-selective filters, the ITU grids and a multiplexer/demultiplexer pair on them with crosstalk that falls out of the channel spacing, an OSA, coupled-channel propagation (XPM with walk-off, FWM accumulating coherently across spans), the session server, a schematic editor — add, wire, move and delete blocks, edit parameters, run, sweep, open and save — the OSA's trace drawn in the dock, and 400G/800G reference designs validated against the OSNR relations, and a back-end indirection the propagation kernels dispatch through — CuPy runs it where a device exists, `maiman devices` cross-checks it against NumPy, and a CI job does the same on any runner labelled `gpu` | ~6 months |
| **4 — PIC** ✅ | Bidirectional S-matrix circuit solver, waveguide, directional coupler, all-pass and add-drop ring resonators, cross-validated against SAX; N×N MMI couplers on the self-imaging phase relations; a Mach-Zehnder interferometer assembled from them — switch, interleaver, or both; and PDK import, which reads a foundry's fitted numbers out of a JSON kit and refuses to extrapolate them past the window they were fitted in; and birefringence, with each guided polarization carrying its own indices through the same reduction | — |
| **5 — Gratings, sensing, loops and coupling** ✅ | Fibre Bragg gratings assembled from transfer matrices — apodized, chirped, sampled and phase-shifted — a circulator with isolation and return loss, and a grating read as a strain gauge and a thermometer, by a swept laser or by broadband light on an analyser; noise that carries the spectral shape of what it passed through; loop control that runs a cavity to its fixed point, and a delay line that turns the same loop into a recirculating one lap by lap; pump depletion for four-wave mixing and Raman's measured gain shape past its peak; an erbium transient spread across the spectrum, each channel moving by the fibre's own tilt, drained per channel by its own cross section, and answered by a pump control loop with a bandwidth and a ceiling; an amplifier's own ASE, both ways, depleting its inversion; PMD applied along the span between Kerr steps, and the coherent polarization term that moves power between the axes; a scalar mode solver for core and cladding modes, and the vector HE, EH, TE and TM modes the glass-air boundary splits them into, checked against the exact characteristic equation; glass that disperses as Sellmeier's silica and germania, which makes the default fibre a G.652 one; a long-period grating built on either, alone or as a pair recoupling its cladding light, and a tilted grating whose comb of cladding resonances reads what the fibre is dipped in, split by polarization when its vector modes are solved; edge and grating couplers that put a signal into the chip's TE and TM, the edge coupler's two facets a cavity summed bounce by bounce, and the grating coupler's passband, its teeth's own reflection, its bottom mirror and its apodization computed from its geometry; a PAM4 driver with its modulator's linearity corrected, and a feed-forward and decision-feedback equaliser, fractionally spaced or blind; a laser's linewidth and intensity noise from its own Langevin forces, and the partition noise between a Fabry-Perot laser's modes, carried down a link as each mode's own arrival time and spent at the detector; a directly modulated laser whose chirp comes out of its own rate equations, with its junction's heat moving the line; and templates in the studio's Examples menu, including an eight-channel DWDM link | — |
| **Open** | Nothing on the W-Port: all eight of its interfaces, DO and DPO, run end to end and back. What is left are the approximations each block states in its own docstring and [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) collects | — |

¹ One developer, part-time. Estimates, not commitments.

Phase 1 is deliberately smaller than a first instinct suggests: SSFM, PMD, Kerr, APD and the GUI
are all pushed out of it. Shipping a *validated* linear link quickly matters more than breadth.
