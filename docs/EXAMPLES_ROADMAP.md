# Example projects: a roadmap from first light to real systems

The examples are where Maiman teaches. Each one is a link someone can open, run, read about and
change, and each one is also a test of the simulator: building it is how we find out which block
is missing or which number is wrong. This document is the list, in the order we intend to build
it, and the log of what building it turned up.

The list is in **phases** because the order matters. An early example often reveals a component
the later ones need (a time-domain scope, a wavelength-selective switch), and finding that out on
a simple link is cheaper than finding it on a submarine cable. When an example uncovers a missing
block or a bug, it goes in [the log at the bottom](#found-along-the-way), and the phase it blocks
waits for it.

## How an example is built

Every example has the same four parts, so each one teaches the same way:

1. **A script** in [`examples/python/`](../examples/python/) that builds the graph, runs it and
   prints what it measured. The script is the source of truth; nothing is drawn by hand.
2. **A project** in [`examples/maiman/`](../examples/maiman/), written by the script, with a
   layout, so it opens in the studio laid out.
3. **A lesson** in [`examples/python/lessons.py`](../examples/python/lessons.py): what the link is
   for, the formulas behind each stage, and a short *Try this* list. It opens in the notes panel
   beside the canvas, and `[[label]]` in it links to a block.
4. **Marks on the canvas** from the same module: frames around the stages, and a note with an
   arrow where something is worth looking at.

A test then holds the example to the number its lesson quotes, so the prose cannot drift away
from the physics.

## Legend

| Mark | Meaning |
| :--- | :--- |
| ✅ | Exists today: a script, a studio project, or both |
| 🟢 | Should be buildable from the blocks the palette already has (not yet tried) |
| 🟡 | Needs a new component, named in the row |
| 🔴 | Needs a new kind of model, not just a block |

---

## Phase 1: First light

Single ideas, one or two blocks each. The point is to see one relation work before it is buried
in a link.

All seven are written by [`first_light.py`](../examples/python/first_light.py), and
[`tests/test_first_light.py`](../tests/test_first_light.py) holds each to the numbers its lesson
quotes.

| # | Example | What it teaches | Status |
| :-- | :--- | :--- | :--- |
| 1.1 | Laser on a power meter and an OSA | dBm and mW, linewidth, what an OSA's resolution does | ✅ `laser_meters.maiman`, with lesson |
| 1.2 | Loss budget | $P(L) = P(0)\,10^{-\alpha L/10}$, swept over length | ✅ `loss_budget.maiman`, with lesson |
| 1.3 | MZM transfer curve | $\cos^2(\pi V/2V_\pi)$, bias points, extinction ratio | ✅ `mzm_curve.maiman`, with lesson |
| 1.4 | A pulse spreading in fibre | $T_1/T_0 = \sqrt{1 + (z/L_D)^2}$ | ✅ `pulse_spreading.maiman`, with lesson, on the new **Optical Oscilloscope** |
| 1.5 | Chirped pulse compression | Sign of $\beta_2$ against sign of chirp | ✅ `chirp_compression.maiman`, with lesson |
| 1.6 | The fundamental soliton | $N = 1$: dispersion and Kerr cancel | ✅ `soliton.maiman`, with lesson |
| 1.7 | Splitters and couplers | Power conservation, 3 dB per split | ✅ `splitters.maiman`, with lesson |

## Phase 2: Direct-detection links

The links most fibre in the world still carries. Eyes, Q and BER.

| # | Example | What it teaches | Status |
| :-- | :--- | :--- | :--- |
| 2.1 | 10 Gb/s OOK, back to back and over 20 km | Eye, $Q$, $\tfrac{1}{2}\mathrm{erfc}(Q/\sqrt2)$ | ✅ `ook_eye.maiman`, with lesson |
| 2.2 | Receiver sensitivity: PIN against APD | BER against received power, APD gain and excess noise | 🟢 Attenuator, PINPhotodiode, APDPhotodiode, sweep |
| 2.3 | Dispersion-limited reach of a DML | Chirp from the rate equations, reach at 10 and 25 Gb/s | ✅ `dml_reach.py`; studio project to add |
| 2.4 | Laser noise floors | RIN, linewidth, mode partition noise | ✅ `laser_noise.py`; studio project to add |
| 2.5 | 50/100G PAM4 lane with FFE/DFE | Four levels, equalisation, why a DFE cannot reach a precursor | ✅ `pam4_lane.py`; studio project to add |
| 2.6 | Four-lane CWDM data-centre link (O-band) | Coarse grid, 1310 nm, no amplifier | 🟢 to verify the Multiplexer on a 20 nm grid |
| 2.7 | Passive optical network (GPON, 1:32) | Split loss, power budget, two wavelengths on one fibre | 🟢 downstream; 🟡 upstream needs a **burst-mode receiver** |

## Phase 3: Amplified and WDM links

Many channels, amplifiers and the noise they add, and the Kerr effect once the power is high.

| # | Example | What it teaches | Status |
| :-- | :--- | :--- | :--- |
| 3.1 | One EDFA, before and after | Gain, ASE, noise figure on an OSA | 🟢 EDFA, OpticalSpectrumAnalyzer |
| 3.2 | Chain of amplified spans | OSNR falls $10\log_{10}N$ | ✅ `amplified_link.py`; studio project to add |
| 3.3 | Four channels on the 100 GHz grid | Grid, ASE floor, OSNR and resolution bandwidth | ✅ `wdm_osa.maiman`, with lesson |
| 3.4 | Eight-channel DWDM with DCF | Dispersion map, XPM, dropping one channel | ✅ `dwdm_link.maiman`, with lesson |
| 3.5 | Optimum launch power | Noise against nonlinearity: the BER bathtub | 🟢 sweep on the booster of 3.4 |
| 3.6 | FWM on a zero-dispersion fibre | Why G.653 fibre failed for WDM | ✅ `wdm_nonlinear.py`; studio project to add |
| 3.7 | EDFA gain tilt and pump control | Spectral gain, a control loop with a bandwidth | ✅ `edfa_gain_tilt.py`, `edfa_pump_control.py` |
| 3.8 | Channel drop transient | Surviving channels jump when others leave | ✅ `edfa_transient.py` |
| 3.9 | A ROADM add/drop node | Express, add and drop on a wavelength-selective switch | 🟡 needs a **WSS** block (a mux/demux pair stands in today) |
| 3.10 | Distributed Raman amplification | Gain inside the span, lower noise figure | 🟡 needs a **Raman pump / amplifier** block (inter-channel SRS exists) |

## Phase 4: Coherent transmission

Phase and amplitude, and the DSP that recovers them.

| # | Example | What it teaches | Status |
| :-- | :--- | :--- | :--- |
| 4.1 | QPSK back to back, ideal carrier | Constellation, EVM, $\mathrm{EVM} \approx 1/\sqrt{\mathrm{SNR}}$ | 🟢 |
| 4.2 | 16-QAM over 80 km with the full DSP chain | CD compensation, timing, frequency and phase recovery | ✅ the studio's own project, with lesson; in the Examples menu |
| 4.3 | Soft-decision FEC | LLRs, the post-FEC cliff | ✅ `coherent_sdfec.maiman`, with lesson |
| 4.4 | Dual-polarisation 256 Gb/s | Butterfly equaliser, PMD | ✅ `dualpol_link.py`; studio project to add |
| 4.5 | Blind acquisition of a large offset | Why the 4th-power estimator folds at $\pm R_s/8$ | ✅ `acquisition_link.py` |
| 4.6 | 400ZR and 800ZR reference designs | Required OSNR, line rates | ✅ `zr400.maiman`, `zr800.maiman`; lessons to add |
| 4.7 | 1000 km on a recirculating loop | Lap-by-lap accumulation, as a lab measures it | ✅ `recirculating_loop.py` |
| 4.8 | Probabilistic shaping (FlexO DPO) | Shaped 16-QAM, 3.28 bit/symbol | 🟡 the shaper exists in `maiman.pcs`; needs a **PCS mapper** block to be drawn |

## Phase 5: Photonic integrated circuits

The same light, on a chip, solved as S-matrices.

| # | Example | What it teaches | Status |
| :-- | :--- | :--- | :--- |
| 5.1 | Directional coupler: split ratio against length | Supermodes, coupling length | 🟢 DirectionalCoupler, sweep |
| 5.2 | MZI interleaver | Path difference to FSR | ✅ `mzi_interleaver.py` |
| 5.3 | Ring resonator filter | FSR, Q, extinction, critical coupling | ✅ `microring_filter.py` |
| 5.4 | A birefringent ring | Two combs on two FSRs | ✅ `birefringent_ring.py` |
| 5.5 | Getting light on and off a chip | Edge and grating couplers, their passbands | 🟢 EdgeCoupler, GratingCoupler |
| 5.6 | A circuit from a foundry PDK and a layout netlist | Reading a kit, refusing to extrapolate | ✅ `pdk_import.py`, `netlist_circuit.py` |

## Phase 6: Sensing

Gratings and cladding modes as instruments.

| # | Example | What it teaches | Status |
| :-- | :--- | :--- | :--- |
| 6.1 | FBG as a strain gauge and thermometer | $\lambda_B = 2 n_\text{eff}\Lambda$ and how it moves | ✅ `fbg_sensor.py` |
| 6.2 | FBG with a circulator | Reflection as a drop port | ✅ `fbg_circulator.py` |
| 6.3 | Tilted grating refractometer | Cladding resonances read the outside index | ✅ `tilted_grating_refractometer.py` |
| 6.4 | Long-period grating pair | Recoupling cladding light | 🟢 LongPeriodGrating |
| 6.5 | OTDR | Rayleigh backscatter, locating a break | 🔴 needs a **pulsed source and a backscatter model** |

## Phase 7: Real-world systems

Case studies that put the earlier phases together the way an operator would meet them. These
are where we expect to find the most missing pieces.

| # | Example | What it teaches | Status |
| :-- | :--- | :--- | :--- |
| 7.1 | Metro ring with ROADMs | Filter narrowing through cascaded nodes | 🟡 needs the WSS from 3.9 |
| 7.2 | Submarine segment, 6000 km | Gain equalisation, GSNR budget, SDM trade-offs | 🟢 in loop form (4.7); 🟡 needs a **gain-flattening filter** |
| 7.3 | Data-centre interconnect with 400ZR over a DWDM line | Pluggables on an open line system | 🟢 from 3.4 and 4.6 |
| 7.4 | 5G fronthaul, analogue radio over fibre | RF on light, intermodulation, EVM of the radio signal | 🟡 needs an **RF tone source** and an **electrical spectrum analyser** |
| 7.5 | Free-space optical link | Beam divergence, turbulence fading | 🔴 needs a **free-space channel** |
| 7.6 | FMCW LiDAR | Chirped laser, beat frequency to range | 🟡 needs a **swept-frequency source** and a beat-note analyser |
| 7.7 | Quantum key distribution (BB84, decoy states) | Single photons, QBER, secure key rate | 🔴 needs a **single-photon detector** and a photon-counting model |

---

## New components this list asks for

Collected from the 🟡 and 🔴 rows, in the order the phases need them:

| Component | First needed by | Notes |
| :--- | :--- | :--- |
| Time-domain scope (power or field against time) | 1.4 | ✅ **Optical Oscilloscope**: power and instantaneous frequency against time, in the dock's Scope tab |
| Burst-mode receiver | 2.7 | Fast settling threshold for PON upstream |
| Wavelength-selective switch | 3.9 | Per-channel pass, block and attenuate; filter shape per port |
| Raman amplifier | 3.10 | Counter-pumped, using the measured gain shape the SRS model already has |
| PCS mapper | 4.8 | A block over `maiman.pcs` |
| Gain-flattening filter | 7.2 | An `OpticalFilter` with an arbitrary measured shape may be enough |
| RF tone source, electrical spectrum analyser | 7.4 | Also useful for 1.3 (harmonics of the MZM) |
| Swept-frequency source | 7.6 | |
| Pulsed source with backscatter, single-photon detector, free-space channel | 6.5, 7.7, 7.5 | Each is a model, not just a block |

## Found along the way

What building the examples turned up: missing blocks, bugs, numbers that disagreed. Newest
first. Each entry says which example found it and what was done.

| Date | Example | Finding | Outcome |
| :--- | :--- | :--- | :--- |
| 2026-10-06 | 1.2, 1.3 | A project cannot carry a sweep, so the two sweep lessons tell the reader which parameter and range to set by hand | Open: a `.maiman` file could save its sweep beside its notes |
| 2026-10-06 | 1.4 | After a run the dock stayed on whichever tab was open, so a pulse example showed an empty constellation | A run whose results the open tab cannot draw now moves to the first tab that can |
| 2026-10-06 | 1.1 | The toolbar's run settings (32 GBd, 4096 sym, seed 2026) and the status bar's 65 536 samples were written into the page once, for the flagship, and shown for every project; they were wrong for the flagship too, which runs 8192 symbols | Read from the open project's context, and redrawn whenever it changes |
| 2026-10-06 | phase 1 | Eleven templates no longer fit as rows of the File menu | An **Examples** menu, grouped by phase and numbered as here; it also reopens the flagship (4.2) |
| 2026-10-06 | 1.4–1.6 | No block could show a single pulse: the eye diagram folds it onto two symbols | **Optical Oscilloscope** added, with peak, FWHM, RMS width, centroid, energy and chirp measured on every sample; held to the closed-form Gaussian broadening |
| 2026-10-06 | all templates | The studio had nowhere to explain a link | Lesson panel, notes, frames and arrows added; lessons written for the five projects the studio opens |
| 2026-10-06 | 2.1 | `ook_link.maiman` duplicates `ook_eye.maiman` with other labels and no lesson | Open: decide whether to keep both |
