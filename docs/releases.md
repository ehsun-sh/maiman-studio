---
title: "Release notes"
description: "What changed in each release of Maiman Studio."
---

# Release notes

Every release, newest first, as published on [GitHub](https://github.com/ehsun-sh/maiman-studio/releases). Install the latest with `pip install --upgrade maiman`.

## 0.17.0 (2026-10-07)

- A Maiman App shortcut with the app icon that opens the studio with Python alone (run `python tools/make_shortcut.py` once).
- Examples in seven phases, from first light to real-world systems, now with distributed Raman amplification. The Examples menu is grouped by phase.
- Lessons, notes, frames and arrows on the canvas.
- Opening or starting a project now clears every instrument.

## 0.16.0 (2026-10-02)

`pip install maiman==0.16.0`

Four-wave mixing is solved exactly now, where it used to be a truncated series.

### Constant tones: every order at once

`tone_solver` integrates the coupled equations of the launched tones and every product above
`mixing_floor` through the span (`kernels.fwm_tone_solve`, a Dormand–Prince pair). Cross-phase,
depletion and every order of mixing are in them. It lands on the split-step to five digits where
the series was 25 % off at 30 mW over 20 km, or 2.3 times out over four spans. It carries:

- two polarizations, coupled by silica's isotropic Kerr tensor;
- PMD, applied at each tone's own frequency, with the waveplates at the pieces' midpoints;
- stimulated Raman scattering, as a power coupling that leaves the mixing phases alone.

### Modulated channels: one wide grid

`composite_fwm` split-steps every band on one composite grid, so modulated channels mix
exactly. It carries two polarizations and PMD (measured from the first band), lets outer slots keep
a wide product's skirt, and adds Raman inside the step. The step is Yoshida's fourth-order
composition, so a step several times longer gives the same answer; products agree to about 2e-4.

### Phases that are carried, not drawn

A mixing product's carrier phase is carried from span to span, expanded about the first band with
the β2 the mixing uses, and a coherent receiver beats its local oscillator by the phase both paths
carry. With `carry_phase` and `cascaded_fwm`, second-order products add across spans as the
split-step's do.

The second-order series stays as the cheap path. A new test pins why it stops there: split by
order against the exact tone equations, one 10 km span at 10 mW is 0.82 of the exact power at
γ², 1.15 with γ³ and 1.03 with γ⁴. The model's folded pump phase (0.89) already beats the next
truncation.

### Around it

- **Pump loop:** proportional and derivative terms, a loop filter, a delay, a noisy detector, a
  dither, and back-calculation anti-windup (Åström & Hägglund).
- **EDFA:** ASE shaped by a measured erbium spectrum, held to Giles and Desurvire's closed forms.
- **Gratings:** solved in TM as well as TE, with Li's inverse rule for the field across the
  walls. A finite grating coupler is solved with its guided light carried off, and converges with
  a stronger absorber and room for the beam.
- **W-Port receivers:** real log-likelihood ratios, with the shaper's prior on the amplitude bit;
  802.3 error blocks mark what a failed CRC32 covered.
- **Lasers:** junction heating through a Foster network of package stages. Fibres take a
  dispersive surrounding medium from Sellmeier terms.

---

65 components, 2274 tests, every physics block checked against a closed-form or independently
computed result in CI.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

## 0.15.0 (2026-09-23)

`pip install maiman==0.15.0`

The shaped path can be read back, which closes the W-Port: all eight of its interfaces, DO and DPO,
now run TP0 to TP7 and back, against the specification's own test vectors.

### Through the encoder's door

Clause 9.2.4 permutes the last 35 bits of every codeword — eighteen information bits and all
seventeen parity ones — after the parity is computed. There are four orderings and a codeword takes
the one its index selects, `r % 4`. It is applied to the encoder's **matrix**, not to its output,
and that distinction is the whole of it: a row twenty back is read as this one's front half, so by
the time the parity that follows is computed it is parity over permuted bits, and no reordering of
the output can put that right.

Figure 27's table did not survive the document's conversion into a page of run-together digits, so
the four orderings were read off the specification's own TP4 and TP5 — for every mode and every
encoder — by matching each tail bit's values across the blocks where no front half has been read
yet and nothing but a permutation can have happened.

With it, `dpo_transmit` goes **TP0 to TP7 bit for bit** for FlexO-6(e), FlexO-8e and FlexO-8, reusing
the interleavers, the symbol mapper and the DSP frame the DO modes use, unchanged.

### And back: two views of one matrix

The same feedback that makes the permute work makes it interesting to undo. Un-permuting the whole
stream restores every back half and breaks every front half; leaving it alone breaks the backs.
Neither single view decodes.

So the decoder holds both. A codeword's front half is what the line carried, because that is what
its parity was taken over; its back half is that same line untangled, because the permute happened
after. `ofec_decode_stream` takes a `tail` and does exactly that — the gather untangles, the scatter
re-tangles, the information comes out of the untangled view — and nothing else about the iteration
changes.

The measurement that settles it, and the tests hold all three:

```
   a clean shaped stream, decoded ...        bits "corrected"
   as the line carried it                    some hundreds
   untangled wholesale first                 some hundreds, a different set
   with both views                           none
```

`dpo_receive` runs the whole shaped path backwards — deframe, demap, deinterleave, decode with the
tail, unshape, descramble, de-adapt — and hands back the FlexO information with its CRC32 flags. It
round-trips every mode exactly, and repairs four hundred flipped symbols with the CRC32s agreeing
afterwards.

### What is left

On the W-Port, nothing. The shaping tables are still normative data loaded from
`MAIMAN_WPORT_TABLES` rather than shipped, error marking stops at a flag per CRC32 because this
project has no client layer to hand it to, and the decoder is this project's own throughout — the
specification leaves decoding open. Each of those is stated where it applies.

---

65 components, 1910 tests, every physics block checked against a closed-form or independently
computed result in CI.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

## 0.14.0 (2026-09-23)

`pip install maiman==0.14.0`

The W-Port's DPO path, which the last release wrote off as unreachable and which turned out not to be.

### Probabilistic constellation shaping

`maiman.pcs` is clause 9: the transcoder that makes FlexO-6(e), FlexO-8e and FlexO-8 *DPO*
different from their DO namesakes. It does not change the constellation — the line is DP-16QAM
either way — it changes which of its points come up. Each 16QAM amplitude is one bit of a pair,
and shaping makes that bit the *inner* amplitude more often than the outer one:

```
   mode          b     shaping rate    P(inner)   net bits/symbol
   FlexO-6(e)    72       33.7 %        0.8265         2.594
   FlexO-8e     106       11.0 %        0.681          3.125
   FlexO-8      116        5.71 %       0.623          3.281
```

**Those probabilities are written nowhere in the code.** The two lookup tables are bijections on
their words ordered by falling weight, so a field of nine bits reaches only the heavy half of a
ten-bit table — fewer indices, all of them low-energy words — and the measured probability falls
out of how far in a field can reach. The tests read it back off the shaper's output and compare
with the specification's Table 4.

The chain: the scrambled stream is split a bit at a time between the four encoders, each share is
cut into 84 coder blocks, and each block alternates chunks of shaping bits with runs of sign bits.
Every chunk feeds four lookup arrays round robin; each array cuts its `b` bits into twelve fields
that index the tables **least significant bit first**; each 128-bit result is rewired — a different
permutation per polarization, quadrature and encoder — and the four are bit-interleaved into a
512-bit column. Test points TP3 and TP4 come out bit for bit, for all three modes, and the shaping
inverts exactly.

The adaptation and the scrambler needed only a mode table: the DPO modes read the FlexO structure
in 2,056-bit columns instead of 10,280-bit ones, and twenty narrow rows are four wide ones, so the
CRC32 covers the same 41,120 bits either way. 433 rows and 22 CRC32s for FlexO-6(e), 522 and 27 for
FlexO-8e, 548 and 28 for FlexO-8. Verified against TP0, TP1 and TP2 for each.

### Why it was reachable after all

0.13.0 said the shaping lookup tables were published only inside the document and could not be had.
That was true of the PDF. The MSA wiki also carries the specification as a `.docx`, and the `.docx`
still has the OLE attachments the PDF dropped — `SCS_LUT10`, `SCS_LUT11` and the rewiring table. The
same wiki's test vector archive has `b72`, `b106` and `b116` members beside the `qpsk` and `16qam`
ones this project already used, so every DPO test point is checkable.

**The tables are not in this repository.** They are normative data, and their within-weight ordering
is not an enumeration anything here could reproduce — colex, lexicographic and numeric were all
tried and none of them is it. So they are handled the way the OFEC vectors are: point
`MAIMAN_WPORT_TABLES` at a directory holding them and the shaper runs; without it, it raises and
says what is missing. What does ship is the block order — 3,552 numbers saying where each bit of an
encoder's block comes from — because that is a fact about the block's own bit classes rather than a
table quoted from the document, and it is the same permutation for every mode and every encoder.

### Where the DPO path stops

At the encoder's door. That path's encoder output ordering is not the DO path's, and clause 9.2.4
permutes the last 35 bits of each codeword back through the code, so TP4 to TP5 is not modelled and
a DPO transmitter cannot yet be run to TP7. That is the one thing left on the Open list, and it is
smaller than what it replaced.

---

65 components, 1891 tests, every physics block checked against a closed-form or independently
computed result in CI.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

## 0.13.0 (2026-09-23)

`pip install maiman==0.13.0`

The last two items on the Open list, and one of them had been there since the list was written.

### Mode partition noise, made by the link

It used to be computed *at* the laser, by `dispersed_power`, and the reason was written down as the
open item: every band here is an envelope in its own retarded frame, so the constant group delay a
span gives it has been divided out, and a detector summing several bands adds them as though they
had all arrived at once — and reports the laser's quiet total, which is the one thing that is not
true a span later.

So the delay is **carried** rather than applied. `WalkoffHistory` rides on the signal beside the
dispersion and Kerr histories, one number per carrier; a span with `carry_walkoff` adds each band's
`D L dlambda` to it; and the detector spends it once, delaying each band's power before the sum.
`FabryPerotLaser` is the source that needs it — seven longitudinal modes on one reservoir, each a
band, from the multimode rate equations that were already here.

```
   7 modes, 1.1 nm apart, 17 ps/nm/km, a 12.8 ns window

   span     modes arrive over   received noise (relative variance)
   0 km          0 ps           1.58e-2   the partition cancelled in the sum
   5 km        561 ps           1.29e-2   still largely correlated
   20 km      2244 ps           3.33e-2
   50 km      5610 ps           6.47e-2   against a ceiling of 8.58e-2
```

The ceiling is the sum of the modes' own variances — what a receiver sees once nothing cancels at
all — and it is approached from below. The dip at five kilometres is not a glitch: a delay shorter
than the fluctuations' own correlation time leaves them partly aligned.

Held against `dispersed_power`, which does the whole thing at the laser and shares no code with
this, to a thousandth of the mean. `carry_walkoff` starts false, so a link that does not ask sees
exactly what it saw before.

### The W-Port framing around OFEC, TP0 to TP7

`maiman.ofec` ended where the code ends and said so. `maiman.wport` is everything either side of
it — clause 8 before, clauses 11 and 12 after — written with the same document open and held to the
same standard, which is the specification's own test vectors. The compliance point is TP7 from TP0
and every point between is an implementation choice; this reproduces all of them, for DP-QPSK and
DP-16QAM alike.

```
   TP0  58 x 10,280 bits of FlexO-4(e)          116 rows for FlexO-8(e)
        a CRC32 every four rows, then a CRC32 and zero pad
   TP1  596,736 bits                            1,193,472
        scrambled, x^16 + x^12 + x^3 + x + 1, reset to 0xFFFF each block
   TP2  the codec payload            maiman.ofec takes it from here
   TP6  688,128 line bits                       1,376,256
        four bits to a DP-QPSK symbol, eight to a DP-16QAM one
        FAW, training, reserved and pilots inserted
   TP7  175,104 symbols in each polarization
```

A DSP frame is 24 subframes of 7,296 symbols, and its budget closes exactly: 2,736 pilots, 240
training symbols that are not pilots, 22 + 74 of alignment and reserve, and 172,032 payload — which
is precisely what the symbol mapper made.

Two readings the text leaves open, which only one answer to reproduces the vectors. The reserved
symbols **straddle** the pilot at slot 64 rather than displacing it, because the overhead goes in
first and the pilot grid second. And the pilot PRBS10's seeds appear at its output least significant
bit first, which makes its recurrence the stated polynomial's reciprocal — the same register read
from the other end, and the only reading that gives Table 15.

The CRC32 has one more hold on it than the vectors: `0xFC891918`, the check value catalogued for
CRC-32/BZIP2, which is what this clause's procedure is over a byte string's bits.

Not covered, and said so: the DPO path, FlexO-6(e)/8(e) with probabilistic constellation shaping,
whose shaping lookup table the published vectors do not exercise. Writing one from memory is the
thing `maiman.ofec` was careful not to do — and it is now the only thing left on the Open list.

---

65 components, 1872 tests, every physics block checked against a closed-form or independently
computed result in CI.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

## 0.12.0 (2026-09-22)

`pip install maiman==0.12.0`

Two more items off the Open list. Both are behind flags that start false, so no result from an earlier release moves.

### A grating coupler, from its own geometry

Four things that were numbers or were missing.

**What its teeth send back.** The same corrugation that radiates on its first order couples the forward mode to the backward one on its second — a Bragg grating written in the chip. Set `from_teeth` and it is solved rather than quoted:

| fibre angle | period that radiates there | reflection |
| ---: | ---: | ---: |
| 0° | 572.0 nm | −1.6 dB |
| 5° | 591.0 nm | −19.4 dB |
| 10° | 611.1 nm | −21.5 dB |
| 15° | 632.3 nm | −26.2 dB |

- The angle is the whole defence: a vertical coupler reflects its own Bragg line, and the detuning that keeps a tilted one quiet is the same `2π n sin θ / λ` that sets the emission angle.
- A duty cycle of exactly one half has no second harmonic, and reflects nothing.
- Checked against this library's own transfer-matrix grating, which shares no algebra with the closed form, to 1e-9.

**The height the fibre sits at.** `fibre_height` widens the beam where it lands — 150 µm up costs more than a decibel — curves it, which the overlap integral carries as a phase, and makes the gap an etalon ringing at `λ²/(2nh)`, measured against that to 5 %. The tilt walks each round trip `2h tan θ` sideways, so the ripple falls from **0.81 dB at normal incidence to 0.06 dB at 12°**.

**A bottom mirror.** `substrate_extinction` makes the substrate absorbing — aluminium is `1.44 + 16j` — and the share leaving upward rises from **0.59 to 0.99**. An index with the other sign is a gain medium under this convention and is now refused rather than quietly flipped.

**Apodization.** `grating_strength_end` tapers the radiation along the grating; the emitted profile follows from energy conservation alone:

```
   uniform, 0.14 /um                 -3.19 dB
   tapered, 0.05 to 0.4 /um          -2.63 dB
   with a mirror as well             -0.48 dB
```

### A laser's junction warms, and its line moves

Most of a laser's drive becomes heat: `I(V_j + I R_s)` in, and only the light carries any away. Set `thermal` and the junction warms by `R_th` times what is left, moving the line by 0.09 nm/K. Two halves, on two time axes:

**The bias sets where the line sits.**

| bias | junction | wavelength | against cold |
| ---: | ---: | ---: | ---: |
| 20 mA | 0.90 K | 1310.0810 nm | −14.1 GHz |
| 35 mA | 1.60 K | 1310.1437 nm | −25.1 GHz |
| 50 mA | 2.36 K | 1310.2128 nm | −37.2 GHz |
| 80 mA | 4.20 K | 1310.3782 nm | −66.1 GHz |

That is 4 to 6 pm per milliamp, what a DFB is specified at, and it belongs to the band's own centre frequency: a window of a few hundred nanoseconds cannot hold 25 GHz as a phase ramp.

**The pattern moves it, slowly.** The junction follows its dissipation through one pole of about a microsecond, so a long mark warms it and the line drifts red — thermal chirp, the low-frequency tail under the carriers' adiabatic chirp, and the reason a line code's run length matters here:

```
   mark, 30 to 50 mA      junction    line drifts
     100 ns               +0.095 K      -1.34 GHz
     500 ns               +0.394 K      -6.03 GHz
    2000 ns               +0.865 K     -13.45 GHz
```

The temperature is a third state beside the carriers and the photons, started settled as they are. One test runs it to rest against `R_th P_diss` — two code paths, one closed form — and another measures the pole's time constant back out of a step, to 5 %.

---

64 components, 1815 tests, every physics block checked against a closed-form or independently computed result in CI.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

## 0.11.0 (2026-09-20)

`pip install maiman==0.11.0`

Three more items off the Open list. Each is behind a flag that starts false, so no result from an earlier release moves.

### Four-wave mixing feels the pumps' own phase

Self- and cross-phase modulation turn a mixing product's drive, `A_i A_j A_k*`, and the carrier the product lands on by different amounts — the textbook `−γ(P_i + P_j − P_k − P_F)`, since the cross-phase of everything else turns every carrier alike and cancels. Set `pump_phase` on the fibre and that enters in three places:

1. **Inside a span**, as a rate the mixing integral carries down the span as the pumps decay, `L_eff(z)` rather than `z`. Summed as a series in `r/α`, or by quadrature past it; the two agree to 1e-12.
2. **Between spans**, through a Kerr history the signal now carries — each carrier's angle, and the angle a weak carrier on the same path would have. It is the nonlinear counterpart of the accumulated dispersion, and every optical block carries it through.
3. **In the product's frame.** The split-step keeps turning the band a product was added to, so a new span's share has to arrive turned the same way.

What checks it is the split-step solution of the same fibre with every tone in one band, which has no mixing model at all. At 20 mW, after four amplified 80 km spans:

| D [ps/(nm·km)] | linear mismatch ÷ split-step | with `pump_phase` ÷ split-step |
| ---: | ---: | ---: |
| +4 | 23.9 | 1.04 |
| −4 | 0.044 | 1.01 |
| +17 | 0.021 | 0.99 |

A join of two paths that went through different Kerr fibre is recorded rather than refused; only a span that asks for `pump_phase` refuses it.

### A tilted grating's comb, polarization by polarization

Set `vector` on a `TiltedFiberBraggGrating` and it solves the cladding's true HE, EH, TE and TM modes and couples each input polarization on its own. With the fringes tilted in `x–z`, each mode of order `ν` has two orientations of one effective index, and one sign separates the two polarizations:

```
π ∫_core [ J_{ν−1}(K_t r) S  ∓  J_{ν+1}(K_t r) D ] r dr
S = e_r e_r' + e_φ e_φ'     D = e_r e_r' − e_φ e_φ'     minus for p, plus for s
```

- p reaches TM0m and never TE0m; s the reverse, exactly.
- Untilted it is the long-period grating's vector coupling to 1e-12, and the core's own reflection matches the scalar core's to 0.2 %.
- In weak guidance each LP family's vector modes together carry what the scalar LP mode does, to 2 %.

Near the cladding's cutoff the split is largest: five nanometres below the Bragg line, a 30 mm grating's two polarizations differ by up to **14 dB**. The block sends the field's `x` through p's comb and `y` through s's. The vector modes cost about ten times the scalar ones, and the two polarizations share one mode solve.

### Two long-period gratings, and the cladding light comes back

A single grating's cladding light is lost under the coating. Set `pair` and the grating is written twice, `separation` apart over bare fibre: the first splits the core's light, the two run at different speeds, the second recombines them. That is a Mach-Zehnder interferometer inside one fibre, and the notch fills with fringes.

- **No gap** is one grating twice as long, to 1e-12.
- **At a 3 dB split** the fringes swing between `(1 + g)²/4` and `(1 − g)²/4`, `g` the cladding amplitude surviving the gap — a perfect null with no loss, matched to 1e-4 at 0, 3 and 10 dB.
- **The fringe spacing** is `λ²/(Δn_g d)` with `Δn_g` from the mode solver: within 5 % at a 20 cm gap, 2 % at 80 cm.
- **At a full crossover** the pair passes back everything one grating cuts.

---

64 components, 1782 tests, every physics block checked against a closed-form or independently computed result in CI.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

## 0.10.0 (2026-09-19)

`pip install maiman==0.10.0`

This release removes two items from the Open list. Both are opt-in, so no result from an earlier release moves.

### An amplifier's own noise empties it

Saleh's equation is the reservoir's energy balance: `P_sat ln(G₀/G)` is everything the inversion hands out, over every field in the fibre. The amplifier's own spontaneous emission is one of those fields. It has no input, and it leaves by both ends, each carrying `NF·G·hν·B` — exactly what the block emits into the link. Set `EDFA(saturate=True, self_saturation=True)` and that load is counted:

```
ln(G₀/G) = [(G − 1)·P_in + 2·NF·G·hν·B] / P_sat
```

With no input the gain is `W(βG₀)/β` in Lambert's W. That is the gain a coil can hold before its own noise empties it. The test solves W by Halley's iteration, which shares nothing with the component's own solver, and the two agree to 1e-12. With the defaults:

| small-signal | gain in the dark | forward ASE | cost at −40 dBm in |
|---|---|---|---|
| 20 dB | 19.98 dB | −7.9 dBm | 0.02 dB |
| 30 dB | 29.81 dB | +1.9 dBm | 0.19 dB |
| 40 dB | 38.59 dB | +10.7 dBm | 1.38 dB |

At 0 dBm input, every one of them is within 0.02 dB of the model without it. The transient integrators carry the same term:

- an amplifier whose channels are all dropped comes to rest on Lambert's gain, not on `G₀`;
- it relaxes at `τ/(1 + (P_out + P_ASE)/P_sat)`, which a test measures to a part in a thousand.

### The glass disperses

`StepIndexFibre`, `LongPeriodGrating` and `TiltedFiberBraggGrating` take `material_dispersion`:

- the cladding follows Malitson's fused silica;
- the core is germania-doped silica, interpolated in mole fraction toward Fleming's GeO₂, at the fraction that gives the quoted index step (3.5 mol % for 0.0052).

The quoted indices hold at 1550 nm. Every mode solver, scalar and vector, takes the fibre at the wavelength it is solving.

With nothing fitted, the default fibre becomes a G.652 fibre:

| | constant indices | dispersing glass | G.652 |
|---|---|---|---|
| D at 1550 nm | −4.8 ps/(nm·km) | **16.7 ps/(nm·km)** | ≤ 18 |
| zero dispersion | none | **1308.3 nm** | 1300 – 1324 nm |

Silica's own dispersion zero comes out at 1272.7 nm.

A long-period grating's notches are set by the difference of two indices that disperse differently, so they move by nanometres: the first moves 4.2 nm shortward and the fourth 1.4 nm longward. Each notch still phase-matches on the dispersive indices to 1e-9. A tilted grating's comb is read against indices that hold at 1550 nm, where the comb is, so it moves by picometres.

---

64 components, 1737 tests, every physics block checked against a closed-form or independently computed result in CI.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

## 0.9.0 (2026-09-19)

`pip install maiman==0.9.0`

Two things 0.8.0 did not have: the FEC that coherent pluggables actually run, and a way to look inside a single photonic block. Both are opt-in, so no result from an earlier release moves.

### OFEC, bit for bit

`maiman.ofec` implements the Open ROADM MSA 6.0 W-Port encoder from the normative text. It has four encoders, each a semi-infinite staircase of 16 × 16 blocks, in which every bit is the front of one extended BCH(256, 239) codeword and the back of another twenty block-rows later. After them come the intra-block permutation of Table 8 and the inter-block interleaver.

It is checked against the specification's own test vectors, and **no bit differs**:

- the encoders turn TP2 into TP5;
- the interleavers turn TP5 into TP6.

That covers 2,752,512 bits for DP-QPSK and 5,505,024 for DP-16QAM. The text leaves two things open: which encoder each bit of a codec block feeds, and whether TP6 is merged from the two interleavers. Only one reading of each reproduces the vectors, and that is the one implemented.

The vectors come with no licence to redistribute, so they are not in the repository:

- setting `MAIMAN_OFEC_VECTORS` to where they unzip runs them;
- without them, a fingerprint of the output, taken while it matched, keeps it from drifting.

The specification leaves decoding open, so the decoder is this project's own. It is soft-in soft-out Chase on each bit's two codewords:

- at 1 % raw BER the output is clean;
- at the specification's 2.0e-2 threshold, three passes of Chase-4 leave one bit in eighty wrong;
- six passes of Chase-6 leave 17 in 1e5.

The gap to the threshold is the decoder's, not the code's, and the documentation says so.

This release also corrects what the project said about OFEC: it belongs to Open ROADM, G.709.6, OpenZR+ and 800ZR, not to OIF 400ZR, which uses another code.

### A block's own spectrum in the studio

Select a grating, a ring, a coupler or any other photonic block. Its properties now offer **Show its S-matrix spectrum**. The new S-matrix tab plots every entry of the block's scattering matrix across a wavelength window, as power, phase or group delay, with no link run through it.

- Each device opens on the window worth looking at, such as a grating's line or four free spectral ranges of a ring, and the window can be moved.
- `POST /api/spectrum` builds the block from the same node a run would, then calls `ScatteringDevice.spectrum()`. A script can make the same call.
- Group delay is a local derivative, not a differenced unwrap of the sampled phase. The unwrap turned a straight waveguide's 14 ps into −1 ps once the grid was coarser than the delay.
- Where an entry is 40 dB below its own peak, its phase means nothing. The delay is left undefined there, and the plot shows a gap instead of a spike.

A 10 mm Bragg grating reads **48 ps** away from its line, which is one transit, and **72 ps** at its band edge.

---

64 components, 1714 tests, every physics block checked against a closed-form or independently computed result in CI.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

## 0.8.0 (2026-09-18)

`pip install maiman==0.8.0`

Four pieces of physics that were compact numbers or missing in 0.7.0. Every one is opt-in, so no result from an earlier release moves.

### An edge coupler's facets are a cavity

`EdgeCoupler(etalon=True)` sums the reflections between the fibre's facet and the chip's bounce by bounce, each beam diffracting over its own path with its own Gouy phase. Where nothing diffracts it is the Airy function of a thin film (5e-5); at zero gap it is one interface (1e-12); against a brute-force angular-spectrum propagation of the same cavity it agrees to 2e-4, and to 7e-3 at six degrees of fibre tilt.

Fifty microns of air ripples by **0.49 dB**, fringes `c / 2g` apart. Tilting the fibre walks each bounce off the chip mode, and eight degrees takes the ripple to **0.04 dB** — which is why fibre arrays are polished at an angle.

### A grating coupler's passband from its stack

`GratingCoupler(from_stack=True)` computes what the PDK's two numbers stood for. The share of light going up rather than into the substrate comes from the silicon, the buried oxide and the substrate as a thin-film stack, checked against every layer's boundary conditions solved at once to 1e-12. The grating's exponential beam meets the fibre's Gaussian at best **80.1 %**, the literature's number; and as the emission angle turns with wavelength while the fibre's does not, the overlap falls — which is the passband.

220 nm of silicon on 2 µm of oxide comes to **3.19 dB and 35.7 nm** from its geometry alone. A 2.2 µm oxide buys back 1.1 dB.

### A laser's own noise

`DirectlyModulatedLaser(noise=True)` and `maiman.laser` carry the textbook Langevin forces and nothing else. What the textbooks derive from them comes out:

| | Schawlow–Townes | Henry | measured |
|---|---|---|---|
| α = 0 | 3.23 MHz | 3.23 MHz | 3.08 MHz |
| α = 4 | | 54.9 MHz | 56.2 MHz |

Henry's factor of seventeen is written nowhere in the integration. The intensity noise peaks at 5.07 GHz, where the laser rings (5.09).

`integrate_multimode` gives a Fabry–Perot laser its longitudinal modes on one carrier reservoir. The modes trade power, so the total is quiet — a twentieth of the sum of their noise for seven modes — and `dispersed_power` shows the cancellation undoing itself after a span: seven times the laser's own noise at 60 km. Mode partition is computed at the laser rather than in a link, because bands here are carried in their own retarded frames; the README says so where it applies.

### An equaliser for a receiver in the field

`FFEDFEEqualizer` takes two samples a symbol with `fractional`: sampled half a symbol late, a symbol-spaced FFE is left with six percent of its symbols wrong; a T/2-spaced one decides every symbol at any phase. With `blind` it trains on its own decisions — landing exactly on the reference-trained taps up to a postcursor of 0.4, and failing at 0.5, which the tests keep in view.

---

64 components, 1686 tests, every physics block checked against a closed-form or independently computed result in CI.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

## 0.7.0 (2026-09-18)

`pip install maiman==0.7.0`

Phase 5 is finished. Five pieces of physics the previous release did not have, and one bug it did.

### A laser that chirps because it is being modulated

`DirectlyModulatedLaser` drives the laser itself rather than a modulator after it — one chip instead
of two. Single-mode rate equations, with the transient and adiabatic chirp *measured* from the
field's own phase rather than declared, along with the threshold current, slope efficiency and
relaxation frequency they imply. `examples/dml_reach.py`.

### An amplifier that answers back

`PumpControl` closes the loop a deployed EDFA actually runs: a bandwidth, a setpoint, and a pump
ceiling it can run out of — so a channel drop's excursion is the loop's failure to keep up rather
than the erbium's full swing. 3.21 dB uncontrolled becomes 0.15 dB at 10 kHz. And a watt is not a
watt: each channel now drains the reservoir by its own cross section and photon energy, so dropping
the short-wavelength half of a comb is a larger disturbance than dropping the long-wavelength half
of the same power. `examples/edfa_pump_control.py`.

### Polarization, done where it happens

The coherent term `(1/3) A_y² A_x*` moves power between the axes instead of only dephasing them, and
is stepped exactly in the circular basis — matching a Runge-Kutta integration of the x/y equations to
1e-12. A circular state takes two thirds of the nonlinear phase a linear one does, where the
phase-only form gives five sixths, and an ellipse's axes turn at `(2/3) γ S3`. PMD waveplates are now
applied *along* the span between Kerr steps rather than afterwards. Both opt-in, so no earlier result
moves; both reach the Manakov 8/9 under fast scrambling, which the tests measure rather than assume.

### The fibre's own modes

`maiman.vector_modes` solves the true modes — HE, EH, TE and TM — by matching all four tangential
components across every interface. The LP approximation assumes every index step is small; the core's
is, and the cladding-air step is a third of an index, which is where each LP mode turns out to stand
for a family. Checked against the exact characteristic equation of Snyder and Love (1e-10), against
two layers where the core step vanishes (1e-15), and against the scalar solver where guidance is weak
(2.5e-7). Set `vector` on a `LongPeriodGrating` and its LP04 notch at 1584.1 nm becomes an HE14 notch
at 1583.0 — a nanometre, on a device whose entire output is where its notch went.

### A grating that reads a liquid

`TiltedFiberBraggGrating`. Tilting the fringes a few degrees opens every azimuthal order to the core
mode, which then reflects into a comb of cladding modes below the Bragg line it weakens. The comb
feels what the fibre is dipped in and the Bragg line cannot, while temperature moves both together —
a refractometer carrying its own temperature reference. In water, notches twelve nanometres down move
75 pm while those beside the Bragg line move 0.2. `examples/tilted_grating_refractometer.py`.

### Fixed

Above azimuthal order one the eigenvalue equation is of order `u^l` near `u = 0`, small enough that
quadrature noise changed its sign: both mode solvers reported a mode at `n_eff = n1` that the fibre
does not guide. A fibre of V = 6 had an LP31 and an LP41 that are not there.

---

64 components, 1629 tests, every physics block checked against a closed-form or independently
computed result in CI.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

## 0.6.0 (2026-09-16)

A minor, and nothing in it moves a number an earlier release produced. Two new blocks, an analysis
that spreads across the spectrum, and a roadmap that finally says what is missing. Every figure below
comes from the engine and is asserted in CI.

### A PAM4 lane, and the equaliser that makes it work

`PAM4Driver` Gray-codes pairs of bits onto four levels, and with `predistort` picks the voltages that
space a Mach-Zehnder's output *powers* evenly — a `cos²` curve does not do that for evenly spaced
volts, and the outer eyes are what pay. `FFEDFEEqualizer` takes one sample a symbol and equalises it
with feed-forward taps and decision feedback.

```
   26.5625 GBd PAM4 into a 7 GHz receiver, 4096 symbols
   equaliser        SNR       symbol errors
   none              9.70 dB  642
   FFE, 5 taps      31.51 dB    0
   FFE, 9 taps      38.64 dB    0
```

It is **trained on the reference, as a compliance receiver is, and then measured with its taps frozen
and its feedback fed its own decisions**, so a DFE's error propagation is in the count. The FFE
converges to the least-squares solution for the same samples; a feedback tap lands on the postcursor
it cancels to 1e-3; and `ser_pam` is one axis of `ser_qam`, held to it exactly and to counted errors
in Gaussian noise within 5 %.

### An erbium transient, across the band

One reservoir is one average inversion, and an inversion that moves takes every wavelength's gain with
it — by a different number of decibels at each. `ErbiumSpectrum` holds a coil's Giles parameters, or
derives emission from absorption by McCumber's relation; `spectral_gain_transient` integrates the same
reservoir `gain_transient` does and spreads it across the spectrum.

```
   8 channels at -6 dBm, 7 dropped        (an illustrative spectrum, not a fibre)
   channel   tilt   excursion
   1530 nm   1.92    +6.18 dB
   1550 nm   1.00    +3.21 dB   the single-reservoir answer, to 1e-10 dB
   1560 nm   0.84    +2.70 dB
```

A link designed around the centre wavelength's 3.2 dB under-protects its short-wavelength receivers by
three decibels.

### The roadmap says what is absent

The table stopped at Phase 4 while three releases of work went in. It now carries them, and an **Open**
list collects the approximations the code already states where each is made: the pump control loop, one
rate of drain for every wavelength, PMD applied after the Kerr effect, scalar cladding modes, a grating
coupler's passband from a PDK rather than its stack, a directly modulated laser, bit-exact oFEC.

---

**62 components.** `pip install --upgrade maiman`, then `maiman serve`.

## 0.5.0 (2026-09-15)

A minor. Four new blocks, a mode solver, templates in the studio, and one change that **moves numbers**
at high power. Every figure below comes from the engine and is asserted in CI.

### What changes in your results

**Four-wave mixing no longer creates energy.** Its products used to be made out of nothing: the pumps
left a span exactly as bright as if they had made none, so a lossless span came out with more power
than it was given. Each product now takes its photons from the two pumps that made it and gives one
to the idler. A lossless span conserves energy to 1e-14, and `diagnostics.fwm_depletion` reports the
fraction moved. At link powers — four channels at 0 dBm through 80 km — that is parts per million
and nothing you measured moves. Against the full nonlinear Schrödinger equation, pump loss agrees to
0.2 % at γPL = 0.065 and drifts to 8 % at 0.39, where the undepleted product formula stops holding.
A span where a product would take more than its pump holds is now **refused**.

**Raman past its peak.** A comb wider than 13.2 THz is integrated with silica's measured gain shape
and photons conserved: S+C+L tilts 6.39 dB, not the 11.19 a straight line says. Inside the peak the
closed form is used exactly as before. The notes used to say C and L together cross the peak; they
span 9.7 THz and do not.

### Loops that run in time

`DelayLine` delays the field by a fixed time, carrier phase included. In a `Feedback` loop each pass
becomes a lap one loop-time after the last:

```
   lap   arrives    peak power
     0      0 ps     0.500 mW
     1    800 ps     0.199 mW
     2   1600 ps     0.079 mW
     3   2400 ps     0.031 mW
```

A 10 ps pulse into a 3 dB coupler closed through 800 ps and 1 dB. With the delay at zero, nothing
arrives at 800 ps. `examples/recirculating_loop.py` prints this.

### A mode solver, and a long-period grating

`maiman.modes` solves LP modes of a three-layer step-index fibre, core and cladding, with Bessel
functions computed from their integral representations — the package still depends on NumPy alone.
Bessel values match tables within 4e-12, LP11 appears at the zero of J0, and eight cladding modes
agree with an independent finite-difference solve to 1e-10.

`LongPeriodGrating` couples the core mode to those cladding modes, solved exactly. At a 500 µm
period its notches land at 1354.9, 1388.7, 1455.7 and 1584.1 nm; dip it in water and LP04 moves
2.6 nm shortward, at an index of 1.43 12.1 nm. Scalar modes and no material dispersion, both stated.

### Getting light onto a chip

`EdgeCoupler` is the overlap of two Gaussian beams in closed form — offset, tilt and a diffracting
gap — checked against the integral itself to 2e-7, then Fresnel at both facets. A standard fibre
onto a 3 µm mode loses 5.78 dB. `GratingCoupler` centres by phase matching with the grating's
dispersion: a 611 nm period at 10 degrees centres at 1549.8 nm and tunes 7.0 nm per degree, where a
dispersionless grating would tune 10.5. Both project the signal onto the die's axes, so TE and TM
are now something a link decides.

### Templates

**File** opens four complete links, including a new eight-channel DWDM link on the ITU 100 GHz grid:
multiplexer, 80 km with cross-phase modulation, DCF, demultiplexer and a counted receiver. OSNR
33.04 dB, the neighbour 40.0 dB down, no errors, in about seven seconds.

---

**60 components.** `pip install --upgrade maiman`, then `maiman serve`.

## 0.4.1 (2026-09-13)

A patch. Nothing the engine computes moves; what moves is whether you can see a loop converge, and
whether the repository's own checks pass.

### A loop you can watch converge

0.4.0 shipped `Feedback`, and its `residual` port is how you know a loop has settled. The studio
could not show it. Every scalar was rounded to a fixed number of decimals *before* it was formatted,
so a converged residual of 8.4e-6 was printed as a flat `0` — the one thing that number must never
say. Small scalars now keep their exponent everywhere a result is printed.

A `Feedback` block is also captioned with its residual on the canvas, and the log warns while a loop
is still changing by more than a part in a thousand per pass:

```
   passes   loop caption   log
        1   Δ ∞            loop has not converged: … Raise its passes
        6   Δ 8.4e-6       (nothing — it has settled; drop port -1.751 dBm)
```

Checked in a browser against the circulator-and-grating cavity from the 0.4.0 notes.

### An example of it

`examples/fbg_circulator.py` gains an eighth section: return loss beside a grating is refused as a
cycle, then run by `Feedback`, and the residual ratio is printed next to the loop's round-trip gain —
0.0966, the echo's 0.10 times the grating's |r| of 0.966. A test holds the example to the numbers the
suite asserts.

### CI was red, including on 0.4.0

The type-check step had failed on every commit since the grating-precision work, the 0.4.0 release
commit included. The published package was never affected — the failures were forty-one missing or
unmatchable annotations in two test files — but a release should not go out on a red check, and this
one does not.

---

**56 components.** `pip install --upgrade maiman`, then `maiman serve`.

## 0.4.0 (2026-09-12)

A minor, and one that **changes numbers**. Two corrections move results that 0.3.0 produced; they are
first here rather than last. Every figure below comes from the engine and is asserted in CI.

### What changes in your results

**Dispersion now points the right way.** The GVD term in `propagate_dispersion` was transcribed from a
textbook written in the `exp(−iωt)` convention; `numpy.fft` gives this engine `exp(+iωt)`. A β₂-only
model cannot feel the difference — every pulse width came out right — which is why it lasted. What
could feel it was everything built on the other convention, including the walk-off term in the *same
expression*: over 20 km at D = +17 ps/nm/km a component 200 GHz above the carrier arrived 1089.89 ps
late where `D·Δλ·L` says early. Three conventions move with it:

- the Kerr rotation (the nonlinear phase is now `−γPz`, which is what `exp(−iβz)` gives),
- `GaussianPulse.chirp`, flipped so `C > 0` is still an up-chirp,
- the trial phase of the blind dispersion search.

Link performance does not move — the flagship coherent link is 7.39 % EVM before and after. **Which way
things point does**: phase directions, and a chirped fibre Bragg grating that compensates a span is now
a *negative* chirp, as its own physics always said.

**Noise downstream of a ring, an interferometer or a grating is now read where it is.** A noise bin was
flat, so after anything wavelength-selective it could carry only the average of what survived, and
everything reading at one frequency read that average. A test asserted a ring on resonance improved
OSNR by more than 12 dB; the correct figure is **1.06 dB**, because that ring's 12.44 GHz linewidth is
the 12.5 GHz reference band. A detector behind such a ring was beating against ASE about 16 dB too low.
Noise bins now carry the shape of what they went through; flat bins read bit-for-bit as before.

### Loop control

`CycleError` has always told people to break a loop "with an explicit loop-control component". Now there
is one. `Feedback` sits on any one wire of a cycle and the graph runs a declared number of passes:

```
   passes   drop, 1550 nm    residual
        1      -40.000 dB         inf   nothing has been round yet
        2       -1.701 dB     9.6e-02   the feed-forward sum
        6       -1.751 dB     8.4e-06   the exact cavity, to 4e-7 dB
```

A circulator with 20 dB of return loss beside a grating, against `Circuit.solve`. The residual falls by
**0.0966 per pass — the loop's round-trip gain**, 0.1 × 0.966, asserted in a test. A graph with no
`Feedback` runs once, exactly as before.

### A circulator that leaks, which turned out not to be a loop

0.3.0 shipped the circulator without isolation, on the grounds that the leak made the drop configuration
a cavity. It does not: solved exactly, the drop port equals the feed-forward sum with a difference of
**exactly zero**. What the leak broke was `PortGroup`, which could only partition ports; inputs may now
be shared between groups. At 40 dB of isolation the neighbour two nanometres from a dropped channel is
set by the circulator, not the grating — 37.0 dB of rejection instead of 45.7. Return loss is the part
that really is a cavity, and `Feedback` above runs it.

### Broadband light, and the instrument it enables

Because noise carries its shape, a white-light source and an OSA now read a fibre Bragg grating: the
reflection of flat ASE draws its peak at exactly the input density times `tanh²(κL)`, and a thousand
microstrain moves it 1.209 nm — the same shift the tunable-laser sweep reads.

### A channel plan

`maiman.grid` has the ITU grids, and the thing they exist for is that **G.694.1 is uniform in frequency
and G.694.2 in wavelength**: 100 GHz is 0.781 nm at 1530 and 0.817 at 1565, so a "100 GHz grid" built by
stepping 0.8 nm drifts off ITU by most of a channel across the C band. `Multiplexer` and
`Demultiplexer` sit on those grids as routers rather than splitters, and a demultiplexer port is the
same function as `OpticalFilter`, asserted sample for sample.

### Tests that said more than they checked

`pytest.approx` keeps a 1e-12 absolute floor even when only `rel` is given, and every wavelength shift
in the grating tests is smaller than that. Eighteen assertions checked to a picometre whatever their
docstrings claimed — 1.209 pm per microstrain "to 0.1 %" was accepted anywhere from 0.2 to 2.2. All now
say what they mean, and all still pass.

---

**56 components.** `pip install --upgrade maiman`, then `maiman serve`.

## 0.3.0 (2026-09-12)

A minor rather than a patch, because the scheduler grew a concept. Every number below comes from
the engine and is asserted in CI.

### A mirror in the line, and a graph that goes one way

Every optical block until now is matched at both ends: light enters one port and leaves another,
and a dataflow edge is an arrow that says so. A fibre Bragg grating is the first one that is not.
Its useful output comes back out of the fibre it arrived on, and the only way to that output is a
circulator.

**The solver was already fine.** `circuit.py` has said since it was written that a non-reciprocal
device and a reflecting one both solve correctly — the reduction never transposes anything and
never drops a term. Nothing outside a test had made it prove it.

**The scheduler was not.** The canonical drop —

```
signal → circ.in1 ; circ.out2 → fbg.in ; fbg.reflected → circ.in2 ; circ.out3 → receiver
```

reads as `circ → fbg → circ` and was refused as a feedback loop. It is not one. At the level of
ports, `out3` depends on `in2` and on nothing else, and no light in that picture travels backwards
in time. The cycle was an artefact of scheduling whole components.

So a component may now declare that its ports fall into independent `PortGroup`s, and the run order
is over those. The default is one group holding everything — true of every other block in the
library — and the sort is bit for bit what it was. A real loop is still a real loop, including a
circulator whose three hops are wired into a ring: splitting a component cannot launder a cycle
that runs through all of it. The groups are checked to be a partition before anything runs, because
the cost of the claim being wrong is a port that never executes or one that executes twice, and
both look like a wiring bug somewhere else.

**This is the only behavioural change in the release**, and it changes nothing that existed.

### The grating is assembled, not written down

Two hundred per-section transfer matrices, multiplied. Erdogan's closed form for a uniform grating
appears nowhere in the model — it is in the tests, on the other side of the comparison:

```
        δn    κL    tanh²(κL)      model     max error
     1e-05  0.203    0.039981   0.039981      2.3e-13
     5e-05  1.013    0.588552   0.588552      3.4e-12
     1e-04  2.027    0.932915   0.932915      7.9e-12
     5e-04 10.134    1.000000   1.000000      2.3e-11
```

`R + T = 1` everywhere to 1e-10, and the first-null bandwidth lands on `λ²/(π n_eff L)·√((κL)²+π²)`.
That construction is what buys chirp and apodization, neither of which has a closed form.

### Apodization is mainly about something other than sidelobes

```
       profile     peak R   sidelobe   GD ripple
       uniform     0.9329     -7.6 dB     101 ps
 raised-cosine     0.5886    -31.4 dB     6.5 ps
      gaussian     0.5823    -41.4 dB     8.1 ps
```

The textbook reason is crosstalk. The reason that decides whether a *compensator* is usable is
group-delay ripple, and 101 ps is a whole bit at 10 Gb/s.

### A dispersion compensator that needs no receiver

`DispersionCompensator` is DSP: it inverts the fibre's all-pass filter, so it needs the field, so it
needs coherent detection. A chirped grating is glass and works in front of a photodiode. 10 cm
chirped over 0.71 nm is −1360 ps/nm — 80 km of standard fibre:

```
     span       bare    + grating
      0 km    21.21 ps    44.81 ps
     40 km    29.46 ps    28.52 ps
     80 km    46.06 ps    21.34 ps
    120 km    64.89 ps    30.54 ps
```

Back to the launched width at the span it is matched to.

### The circulator charges twice

Its loss is per hop, so light reaching a reflective device pays going out and coming back:

```
                1550 nm     1552 nm    accounting
       drop     -1.702 dB  -47.359 dB  two hops (-1.4) + R (-0.30)
    express    -12.434 dB   -0.700 dB  one hop (-0.7) + T (-11.73)
```

The neighbour's 45 dB of rejection on the drop port is the **grating's own sidelobe floor** two
nanometres out, not an isolation figure anybody declared.

**Isolation is deliberately not a parameter.** A real circulator leaks 40 to 60 dB backwards, and
carrying that would make every output a function of two inputs instead of one. In this
configuration that leak is a genuine weak cavity, which the engine is right to refuse without an
iteration count.

### The named windows were never the limit

The section loop has always eaten two arrays — a coupling per section and a local Bragg wavelength
per section — and the three named windows and the linear chirp were two ways of filling them.
Handing them over directly costs the loop nothing, and a named window and its array are checked to
be the same device *bit for bit*.

**A sampled grating is one array.** Erase the writing periodically and the single peak becomes a
comb — the tuning element of a sampled-grating DBR laser, a multi-channel compensator, an
interrogator that reads a whole array at once:

```
  7 peaks, spacing            0.41478 nm
  predicted λ²/(2 n_eff Λs)   0.41494 nm
```

The model knows nothing about combs. It has a coupling that is switched on and off.

**A phase-shifted grating is one more.** Break the periodicity once and the stop band acquires a
transmission window: two mirrors facing each other, which is a cavity. At π, dead centre, on 2 cm:

```
  window at 1550.000000 nm, 0.70 pm wide = 87 MHz
  unbroken, the same grating transmits 1.8e-4 there
```

87 MHz is the narrowest feature anything in this library produces — the DFB laser's cavity. Moving
the break off centre makes the halves unequal mirrors and the resonance dies: 1.000, 0.705, 0.099
at positions 0.5, 0.45, 0.35.

The phase rides on the **coupling** and not on the detuning, which keeps each section unimodular
and makes a phase constant along the whole length do nothing at all — the right answer, since where
a grating's fringes start is not observable.

### The same device is a strain gauge and a thermometer

```
  strain        1.209 pm per microstrain     p_e = 0.22
  temperature  11.191 pm per kelvin          α + ξ = 7.22e-6 /K
```

Both a material number times λ_B; nothing is fitted. Warming does two things that are not the same
size — expansion at 0.55e-6/K against the thermo-optic coefficient at 6.67e-6/K — which is why this
is a thermometer made of glass rather than one made of geometry, and why a coating moves the
thermal number a lot and the strain number not at all.

**One grating produces one wavelength and there are two unknowns behind it:**

```
  9.26 microstrain per kelvin

  +10 K, no load        -> 1550.111910 nm
  92.6 ustrain, no heat -> 1550.111953 nm
  difference             43.4 fm
```

**And a second grating at another wavelength does not fix it.** Both sensitivities scale with λ so
their ratio does not, and the 1530/1570 pair has a condition number of 4.6e16. What works is a
grating carrying no load beside the working one — a genuinely independent second equation:

```
   recovered temperature    15.000 K    (error +0.0000)
   recovered strain        400.000 ue   (error +0.0001)
```

Read as though the temperature were known to be zero, which is what an uncompensated gauge assumes,
400 µε at +15 K reads as 538.85 — 9.26 per kelvin, arriving on schedule.

### Two things the engine was right to refuse

**Broadband interrogation is not buildable here.** A white-light source and an OSA is the other
standard interrogator, and broadband light is carried as a `NoiseBin` whose density is flat by
construction, reached by a response as one averaged scalar. The reflection would come back with no
peak in it. Spectrally resolved noise is the change that would fix it, and it reaches the EDFA,
OSNR and everything else that touches ASE.

**Summing an array's reflections onto one fibre is refused by `Combiner`**, correctly: a
single-wavelength probe means every grating reflects a copy of the same band, and co-located
carriers have to be added as fields on a common grid rather than multiplexed.

### A sign that is not the grating's fault

The grating's own physics says a positive chirp gives `D > 0`, the sign standard fibre has — so a
compensator should be a *negative* chirp. Against this engine's `Fiber` it is the positive one,
because `kernels.propagate_dispersion` carries the opposite quadratic sign from
`photonics.propagation_constant`, which that function's docstring has said since before this device
existed. The grating is the first component to put both in one graph where it shows. Not reconciled
here — that is its own piece of work and would move every dispersion number in the project — but
pinned, so the day it is reconciled the failure names the compensator as one of the things that
moved.

---

**53 components, 1382 tests.** `pip install --upgrade maiman`, then `maiman serve`.

`python examples/fbg_circulator.py` and `python examples/fbg_sensor.py` print every table above.

## 0.2.1 (2026-09-12)

A studio release. The physics is unchanged from 0.2.0 — the work since then is in the page that
ships inside the wheel, which is the only way it reaches anyone. One default did change, and it
changes numbers; it is the first item below rather than a footnote.

### The OSA finds a carrier it was not aimed at

An analyser you have to aim is no use for finding something. The window was
`center_wavelength ± span/2` — 1550 nm and 1000 GHz by default — and everything outside it was
dropped on the floor. The wavelength is usually the thing being measured, so that was backwards.

One CW laser at 0 dBm, measured:

| laser | fixed window | automatic |
| :--- | :--- | :--- |
| 1550 nm | 1.0000 mW | 1.0000 mW, peak 1550.00 |
| 1560 nm | 0.0000 mW | 1.0000 mW, peak 1560.00 |
| 1310 nm | 0.0000 mW | 1.0000 mW, peak 1310.00 |

`auto_span` is **on by default** and sweeps what the signal actually occupies — every band's
sampled bandwidth and every noise bin, plus 5 % margin. It is the union rather than a fit to the
strongest, because fitting to the loudest puts a dim neighbour outside the window, which is the
failure being fixed, arriving silently.

**This changes numbers.** An OSA in an existing project sweeps a different window now: the shipped
WDM demo went from a 6.40 nm slice to the whole 35.27 nm comb. Turn `auto_span` off and
`center_wavelength` and `span` mean exactly what they always did. That mode is not vestigial —
`points` is fixed either way, so span buys coverage and costs resolution, and once the channel is
known the narrow window is the one you want.

### Parameters that do nothing are greyed, and say why

Fifteen parameters across eight components turn out to be gated by a flag. The receivers'
`load_resistance` and `temperature` do nothing without `thermal_noise`; the driver's and the
sampler's `roll_off` and `filter_span` do nothing without their shaping flags; the EDFA's
`saturation_power` has done nothing without `saturate` since saturation shipped.

The condition is declared in the model — `Param(..., applies_when="saturate")`, or `"!auto_span"`
for one that applies when the flag is off — and the manifest carries it, so the interface asks and
never decides. A condition naming nothing is refused when the class is defined, not when somebody
opens the panel.

Greyed rather than hidden, and with the reason: a box that disappears takes the knowledge that it
exists with it. The row reads "no effect while auto_span is on" — the switch and the state it is
in, so the sentence names the thing to change.

### A canvas with a viewport

Blocks below the fold were not off-screen, they were gone. The SVG clipped to a viewBox fixed at
`0 0 1000 530`, so a block dragged to y=900 was never drawn — and zooming out, the one gesture that
means *show me more*, scaled the element down and revealed nothing.

Pan and zoom are the viewBox now rather than a CSS transform. The box is the window rather than the
page: a canvas as large as anyone needs, and zooming out that genuinely shows more. A block at
y=900 is invisible at 100 % and on screen at 25 %. The wheel zooms and keeps what is under the
pointer under the pointer — measured rather than derived, so it is exact whatever
`preserveAspectRatio` is doing. The view survives a refresh, per tab, range-checked rather than
trusted.

### Select more than one block, and align them

The selection is a set. Shift or ctrl adds and removes, a drag on the bed draws a band, and a group
delete takes one snapshot so undo reverses "I deleted those four" in one step. Select and Pan are
real tools now, with the cursor naming the tool rather than the surface; space is a momentary pan
whatever the tool is.

Align is eight operations — six edges, two distributes — on the toolbar after the zoom group and in
Edit as "Align selected…". They grey out until there is a selection to act on, which is also how
somebody finds out they need one. Right-drag pans as well as middle-drag, on the canvas only.

### Every analyser, not the first one

A pane showed one analyser when the graph had three: `firstOfKind` returned the first match and
dropped the rest, and which one that was depended on the order the engine happened to return its
results in — so adding an unrelated block could silently change which trace you were looking at.

All of them are kept now. Spectra share one pair of axes and are told apart by colour, the way two
traces on an OSA are; the eye and the constellation stay tiled and captioned, because two 2-D
histograms on one set of axes is mud. The rule is whether the marks can overlap without destroying
each other — lines can, density cannot.

### Also

An install guide for someone who is not a programmer, menus with something behind them, plot panes
that no longer outgrow the dock, and a menubar that sheds its least-missed parts before the two
halves can collide in a narrow window.

---

`pip install --upgrade maiman`, then `maiman serve`.

## 0.2.0 (2026-09-11)

Five pieces of work since 0.1.0. Every number below comes from the engine and is asserted in CI.

### Coarse carrier acquisition

`FrequencyRecovery` is exact below `symbol_rate / (2M)` — 4 GHz for square QAM at 32 GBd — and past
that it does not degrade, it **aliases**: a 4.1 GHz offset reads as −3.9 GHz at a confidence of
23.1, against 24.0 for a correct estimate. The link goes from zero symbol errors to 6002 % EVM.

A sweep over trial offsets cannot fix that, and the reason is not an implementation limit: an alias
of `symbol_rate / M` is a phase advance of exactly one `2π/M` turn per symbol, and `M` is *defined*
as the turn that maps the alphabet onto itself. The aliased constellation is identical to the
correct one, symbol for symbol.

So `CoarseFrequencyRecovery` does not read symbols. It takes the **first circular moment of the
received waveform's power spectrum** — circular because a band straddling the wrap has no
meaningful arithmetic mean, and because a flat noise floor contributes nothing to one, so it never
has to be told the signal's bandwidth.

| | fine (M-th power) | coarse (circular moment) |
| :--- | :--- | :--- |
| reads | symbols | the waveform |
| unambiguous over | ±4 GHz | ±fs/2 — Nyquist, not the method |
| accuracy | sub-MHz | ~±150 MHz |
| past its range | confidently wrong | correct — the alias *is* the sampling |

Measured end to end: +20 GHz recovers to 7.41 % EVM and zero symbol errors against 7.39 % back to
back, and the same holds at +60, +100 and +200 GHz.

### Erbium gain dynamics

An EDFA shares one inversion between everything passing through it, so dropping channels hands the
survivors the gain those channels were using. `maiman.transient` solves the trip between the two
steady states.

It is an **analysis and not a block**, and that is a measurement: the relaxation constant is
`τ / (1 + P_out/P_sat)` — 3.3 to 9.9 ms here — against a simulation window of 128 ns. The fastest
transient is twenty-five thousand windows long, so within one window the gain is a constant, which
is what `EDFA` already assumes. The amplifier therefore gains no parameter it could not honour.

The model is the dynamic form of the Saleh steady state already in the component, so a test
integrates the ODE to rest and requires it to land on that component's own Newton solve — two code
paths sharing no arithmetic, meeting at **1e-9**.

Dropping seven channels of eight from a −6 dBm comb: **+3.21 dB** through one amplifier, **+9.94 dB**
through a chain of eight. Sub-linear, because each amplifier down the chain starts less saturated
than the one before.

### Birefringence

Every photonic block applied one response to both field components — which says the die treats TE
and TM alike, and no strip waveguide ever has. For 500 × 220 nm silicon at 1550 nm:

```
              n_eff    n_group    ring FSR (100 µm)
TE             2.44       4.20         713.8 GHz
TM             1.78       3.80         788.9 GHz
```

So a ring resonates at **two sets of wavelengths on two free spectral ranges**, each matching
`c/(n_g·L)` to a part in a thousand. At 1546.69 nm `Ex` sits in the notch and `Ey` passes; one comb
over, it is the other way round.

**The solver needed no change at all.** `SMatrix` identifies ports by *name*, so a device carrying
two modes is one with twice as many ports, and the reduction already solved those. `birefringent`
is off by default and a test asserts that off, both components take exactly the path they took
before the flag existed — `rel=1e-12`.

### A layout tool's netlist, solved

The roadmap said "PDK import through gdsfactory". Measuring it found there is nothing to import: a
gdsfactory `CrossSection` carries width, offset, layer and bend radius with `extra="forbid"`, and
there is no effective index anywhere in the package.

So the join runs the other way. `maiman.netlist` reads the YAML netlists it writes — as documents,
running nothing — and the `.pdk` supplies what the layout cannot: which model each cell is, which
layout port is which model port, and which parameters the layout decides per instance.

On a file **gdsfactory actually emitted**, shipped unmodified with its MIT attribution:
`−0.0053 dB` against the 26.637 µm at 2 dB/cm anyone can work out by hand. The bend's 16.637 µm of
that is its *arc* length, which the radius alone does not give.

**Nothing imports gdsfactory.** Measured: 86 packages, and it requires `klayout`, which is
GPL-3.0-or-later — the same ground this project already refuses FFTW and klujax on. PyYAML is
dev-only; the reader works over a parsed mapping and handles JSON with the standard library.

### Documentation numbers

Four numbers in the docs were measured rather than trusted, and four were wrong: 48 components
against a registry of 51, 1184 tests against a suite of 1278, a pilot residual of 10.9 mrad over 64
pilots where the shipped link reads 5.4 over 128, and a reduction called "thirty lines of numpy"
that is twelve lines inside a forty-six-line method.

The README's front page is now held by tests: the component count exactly against the registry, and
the test count must be phrased as a floor or the guard fails.

---

**Upgrading.** `0.x` means the API is not stable. `EDFA` and the photonic blocks gained parameters;
none of the defaults move an existing project's numbers, and a test asserts that for the
birefringence flag specifically.

`pip install --upgrade maiman`

## 0.1.0 (2026-09-11)

First release. `pip install maiman`.

An optical link and photonic systems simulator: build a link as a block diagram, run it, and read
it back as an eye diagram, a bit error rate, an OSNR figure or a constellation.

**48 components, 1184 tests, and every physics block checked against a closed-form result in CI.**
That last part is the reason the project exists — a simulator nobody can verify has no scientific
value, and comparing against a commercial tool needs a licence and is not reproducible. Models are
derived from published literature and standards, cited in each component's docstring.

### What runs

**Fiber.** Adaptive-step split-step Fourier with Kerr, dispersion and its slope, PMD,
cross-polarization coupling, and inter-channel stimulated Raman scattering. Coupled-channel WDM
with walk-off and four-wave mixing that accumulates coherently across spans.

**Coherent transceivers.** Gray-coded M-QAM to 256 — odd orders rectangular, and the SER formula
generalises rather than being a second implementation. IQ modulator with bias and quadrature
error, 90° hybrid, balanced detection, dual polarization with a blind butterfly equaliser.

**Receiver DSP, all blind.** Chromatic dispersion compensation with blind estimation of the
accumulated value over spans to 1000 km; square-law timing recovery; M-th power carrier frequency
recovery; blind phase search. The reference design carries all of them, and its LO is detuned
200 MHz because two free-running lasers never are on the same frequency.

**Amplifiers.** EDFA with ASE in noise bins rather than samples, and Saleh gain compression driven
by total power — so WDM channels share one inversion, and eight of them get 5.5 dB more output for
eight times the input.

**Forward error correction.** RS(255,239) per ITU-T G.709, and a soft-decision braided BCH
staircase with Chase-II and Pyndiah decoding. On a 32 GBd 16-QAM link at −25 dBm the line runs
7.4e-3 and the payload comes out exact, where the hard-decision code corrects nothing.

**Photonic circuits.** Bidirectional S-matrix solver cross-validated against SAX to 7e-15;
waveguides, directional couplers, ring resonators, N×N MMIs on the self-imaging phase relations,
Mach-Zehnder interferometers. PDK import reads a foundry's fitted numbers from JSON and refuses to
extrapolate past the window they were fitted in.

**An interface.** `maiman serve` opens a schematic editor — add, wire, move, edit, run, sweep, open
and save `.maiman` projects — that is a client of the same public API and ships inside the package.

### Honest about what is not here

No frequency-offset acquisition sweep beyond ±Rs/2M. No EDFA transient dynamics. One response
applied to both polarizations in a waveguide. The staircase code is the family OIF's oFEC belongs
to and is **not** oFEC — the Implementation Agreement is normative about an interleaver this does
not reproduce. Each of these is stated where it matters rather than left to be discovered.

### Requires

Python 3.11+, NumPy. That is the whole runtime dependency list, deliberately.

---

Apache-2.0. Published with PyPI trusted publishing — no API token exists anywhere in this
repository.
