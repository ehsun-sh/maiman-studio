# Models and results

[← README](../README.md) · [Getting started](getting-started.md) · [The studio interface](interface.md) · [Models and results](physics.md) · [Design and roadmap](design.md) · [Validation](validation.md)

---

## Try it

```bash
pip install maiman
maiman serve
```

Step by step, assuming neither Python nor a terminal:
**[Installing and running it](getting-started.md#installing-and-running-it)**.

Or, to work on it:

```bash
pip install -e ".[dev]" && pytest
```

```python
from maiman import SimulationContext, Graph
from maiman.components import CWLaser, Combiner, Fiber, PowerMeter

ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=16, sequence_length=64)
g = Graph(ctx)

laser = g.add(CWLaser(power=0.0, wavelength=1550.0))  # 0 dBm
fiber = g.add(Fiber(length=80.0, attenuation=0.2))  # 80 km, 0.2 dB/km
meter = g.add(PowerMeter())
g.chain(laser, fiber, meter)

print(g.run()[meter])  # PowerReading(-16.000 dBm; 1550.00nm=-16.000dBm)
```

Two carriers stay two independently sampled bands, which is the point of the signal model:

```python
g = Graph(ctx)
ch1 = g.add(CWLaser(wavelength=1550.0, label="ch1"))
ch2 = g.add(CWLaser(wavelength=1551.0, label="ch2"))
mux = g.add(Combiner(2))
fiber = g.add(Fiber(length=80.0, attenuation=0.2))
meter = g.add(PowerMeter())

g.connect(ch1, mux["in0"])
g.connect(ch2, mux["in1"])
g.chain(mux, fiber, meter)

print(g.run()[meter])
# PowerReading(-12.990 dBm; 1551.00nm=-16.000dBm, 1550.00nm=-16.000dBm)
```

Each band carries its own centre frequency, so channel spacing never enters the sample rate.
Put those two lasers 6 THz apart instead of 125 GHz and nothing about the run changes — which is
exactly what a single-carrier signal model cannot do.

## Results

`python examples/python/ook_link.py` builds a 10 Gb/s OOK link and characterises it. Abridged output:

```
Receiver sensitivity (back to back)        Dispersion-limited reach (0 dBm launch)
  launch      Q    BER (from Q)  counted     distance     Q    BER (from Q)
  -22 dBm   1.59     5.61e-02    460/8184        0 km   94.75    0.00e+00
  -20 dBm   2.51     6.09e-03     49/8184       40 km    9.26    1.03e-20
  -19 dBm   3.15     8.17e-04      7/8184       60 km    6.48    4.63e-11
  -16 dBm   6.25     2.11e-10   none counted    80 km    3.76    8.58e-05
  -14 dBm   9.83     3.99e-23   none counted   120 km    0.63    2.65e-01
```

Two things are worth reading off that. **Sensitivity is −16 dBm** for a Q of 6 — the right figure
for a PIN into a plain 50 Ω load. **Reach is ~62 km**, and it is set by dispersion, not by loss:
with dispersion switched off the same 60 km span gives Q = 15.4 instead of 6.5. That is the
textbook result for uncompensated 10 G NRZ on standard fiber.

The two columns are also a cross-check on each other. 120 km of 0.2 dB/km is 24 dB, and launching
0 dBm through it gives the same Q as launching −24 dBm back to back. Modulator, fiber, detector,
filter and analyzer all have to agree for that to hold; it is
[a test](../tests/test_ber.py), not a coincidence.

Both curves come from `sweep()`, and the same script writes the schematic to
[`examples/maiman/ook_link.maiman`](../examples/maiman/ook_link.maiman) — versioned JSON, diffable, runnable headless.

### Coherent

`python examples/python/coherent_link.py` runs the same treatment on a 32 GBd coherent link —
PRBS → Gray-coded M-QAM → IQ driver → IQ modulator → 90° hybrid with balanced detection —
and finds the received power each format needs for a BER of 1e-3 — a **soft-decision** FEC
threshold, not the hard-decision one; see [What FEC is for](#what-fec-is-for) for what
RS(255, 239) actually carries from there:

```
format     rate      launch for BER 1e-3     SNR there   EVM there
BPSK          32 Gb/s      -38.0 dBm received        6.9 dB     45.2%
QPSK          64 Gb/s      -35.0 dBm received        9.9 dB     32.0%
8-QAM         96 Gb/s      -29.7 dBm received       15.2 dB     17.5%
16-QAM       128 Gb/s      -27.5 dBm received       17.4 dB     13.5%
32-QAM       160 Gb/s      -23.8 dBm received       21.1 dB      8.8%
64-QAM       192 Gb/s      -21.7 dBm received       23.2 dB      6.9%
128-QAM      224 Gb/s      -18.3 dBm received       26.6 dB      4.7%
256-QAM      256 Gb/s      -16.3 dBm received       28.7 dB      3.7%
```

The required-SNR column is the one to check against a textbook: 9.9 / 17.4 / 23.2 / 28.7 dB are
the standard figures for QPSK through 256-QAM at 1e-3. The step from BPSK to QPSK costs exactly
3 dB — the same energy per bit for twice the rate, which is why coherent systems start at QPSK
and never look back.

**The odd orders are rectangular, not cross.** 8-QAM is 4×2, 32-QAM is 8×4, 128-QAM is 16×8: the
extra bit goes on I. A cross — a square with its corners cut off — is the better constellation, 2.30
dB peak-to-average against the rectangle's 3.48 at 32 points, and worth about a decibel of required
SNR. Two things argued against it. A cross **cannot** be exactly Gray coded, which is a proved
result rather than a gap in the construction, so its labelling is a published heuristic that would
have to be transcribed rather than derived — and this project does not transcribe models it cannot
check. The rectangle, being a product of two Gray PAMs, extends everything already written for
square QAM by arithmetic: the levels, the slicer, and an error rate that **is** the square formula,
term for term, with `M_I` and `M_Q` put in separately. If cross QAM arrives it belongs beside this,
chosen by a parameter, not instead of it.

A rectangle is not invariant under a quarter turn, and two things here had assumed every
constellation was. The blind phase search now covers the constellation's **own** rotational
symmetry — `[0, π)` for a rectangle, `[0, π/2)` for a square — because an 8×4 grid turned a quarter
is a 4×8 grid, a different alphabet, and a search that never looked past π/2 could not find an
offset of 0.6π at all. Measured: 0.0022 mean square error with the widened search against 0.1176
with the old one. And differential *quadrant* encoding is refused on an odd order rather than
quietly mis-encoding it — a rectangle's blind ambiguity is a half turn, which quadrant differencing
does not address.

Nothing here is configured to come out right. The shot-noise-limited SNR is asserted against
`R·P/(2qB)`, the counted symbol errors against
[`ser_qam()`](../src/maiman/modulation.py), and the modulator's 3 dB against `10·log10(2)`.

### What FEC is for

A link is not specified at the bit error rate it delivers. It is specified at the rate its
*decoder* delivers, and every sensitivity figure in this project was quoted at a pre-FEC threshold
with nothing behind the word.

The code is **RS(255, 239) over GF(2⁸)**, the generic FEC of ITU-T G.709 — the one in the OTN
frame overhead. It is specified completely enough to build from the standard: the field is
generated by `x⁸+x⁴+x³+x²+1`, sixteen parity symbols correct up to `t = 8` symbol errors anywhere
in the codeword, and the overhead is 6.69 %. Syndromes, Berlekamp-Massey, a Chien search and
Forney's magnitudes; the generator polynomial is *derived* and its sixteen roots asserted rather
than a coefficient table being transcribed.

**A byte is a byte.** Eight bad bits in one symbol cost the same as one — which is why this code
was chosen for a bursty channel, and why every expression below starts from the *symbol* error
rate. A test puts 64 bad bits in eight bytes (correctable) beside 9 bad bits in nine (not).

**On a real link, counted end to end.** 10 Gb/s OOK, a real modulator, a photodiode with its shot
and thermal noise, a filter, and a blind slicer deciding — no formula anywhere in the path:

```
received   pre-FEC BER   post-FEC BER   symbols repaired   codewords failed
-20.0 dBm    5.99e-03      5.73e-03            22              37/40
-19.5 dBm    2.39e-03      1.18e-04           182               1/40
-19.0 dBm    7.72e-04      0.00e+00            63               0/40
-18.5 dBm    1.72e-04      0.00e+00            14               0/40
```

That cliff is about a decibel wide, and the top row is the honest half of it: **below threshold the
decoder makes things worse.** A bounded-distance decoder does not degrade gracefully — past `t` it
can find a wrong error pattern and apply it confidently, so 5.99e-3 in comes out at 5.73e-3 with
miscorrections mixed in. Clipping the output rate at the input rate would look tidier and would
flatter the code. A test asserts the post-FEC rate is *higher* there.

**A correction this work forced.** The sensitivity tables above are quoted at a pre-FEC BER of
1e-3. RS(255, 239) takes 1e-3 to about **1.1e-6** — nowhere near the 1e-15 a transport system is
specified at. To reach 1e-15 it needs roughly 1e-4 at its input. So **1e-3 is a soft-decision
threshold, and this is a hard-decision code.** The word "FEC" was covering both; it no longer does.

The closed-form curve and the decoder are independent and are checked against each other. They
disagreed by a factor of four the first time and **the expression was the one that was wrong**: a
wrong byte is wrong in *one* bit at these error rates, not four. Conditioning on "at least one of
eight flipped" gives `p/p_s`, which is about an eighth at small `p`, not the half it would tend to
at `p = 0.5`. The Monte Carlo caught it.

**Two things the engine did not have, and now does.** A coder makes more bits than it is given, and
the run window here is a fixed number of symbols *on the line* — so the window is the line rate and
coding makes the payload smaller, not the line faster. `FECEncoder` is therefore a source: it makes
its own payload, codes it, fills the window exactly, and emits the uncoded payload on a second port
for the decoder to check against. That is what an OTN framer does.

And there was no digital output from a direct-detection receiver at all. Every bit in this library
used to be decided inside `BERAnalyzer`, which decides *with the transmitted sequence in hand* and
emits a number — the right way to measure a link and no way to build one. `Slicer` is the missing
block: blind in both choices a decision circuit makes, with the sampling instant at maximum eye
variance and the threshold from Lloyd's two-means. It converges on the midpoint of the rails rather
than the optimum, which sits nearer the zero rail because a one carries more noise — a real penalty,
stated rather than smoothed over.

### What soft-decision FEC is for

The hard-decision code above decides each bit, then corrects. This one never decides until the end:
it works on **log-likelihood ratios** all the way through, and how near each symbol sat to a
decision boundary is exactly the information the other one throws away. That is worth two to three
decibels, and here is what it looks like on the same engine.

32 GBd 16-QAM coherent, staircase code at 16.4 % overhead, four blocks, eight iterations — a real
IQ modulator, a real hybrid with balanced detection and its own shot noise, a max-log demapper, and
an iterative decoder working on what that produced:

```
received     pre-FEC BER   post-FEC BER  |  RS(255,239) at the same input
-26.0 dBm      1.49e-02      6.04e-04    |    1.49e-02  — corrects nothing
-25.0 dBm      7.36e-03      0           |    7.21e-03  — corrects nothing
-24.0 dBm      3.20e-03      1.78e-05    |    1.02e-03
-23.0 dBm      1.34e-03      0           |    8.75e-06
-22.0 dBm      4.73e-04      0           |    3.31e-09
```

**At −25 dBm the line is running at 7.4e-3 and the payload comes out exact**, while the classic
G.709 code at that same input returns essentially what it was given — every one of its codewords is
far past `t = 8`. Reading across, the hard code needs about −22 dBm to reach 1e-9 and this one is
already clean at −25: roughly three decibels of sensitivity, which is the number the whole
soft-decision argument is about.

**The code is a braided BCH staircase** — Smith, Farhood, Hunt, Kschischang and Lodge,
*J. Lightwave Technol.* 30(1), 2012 — with Chase-II component decoding (Chase, 1972) and Pyndiah's
soft-output variant (Pyndiah, 1998). It is the family **OFEC belongs to and it is not OFEC**, and a
block that carried the name while being a reconstruction would be the one thing this project
refuses. The parameters are stated as a choice, not as a standard.

### And OFEC itself, bit for bit

`maiman.ofec` is OFEC, written with the normative document open — the Open ROADM MSA 6.0 W-Port
Digital Specification, clause 10 (the same code is in ITU-T G.709.6, OpenZR+ and OIF's 800ZR, and
**not** in 400ZR, which uses another; this README once said otherwise). Four encoders, each a
semi-infinite matrix of 16 × 16 blocks in which every bit is the front of one extended BCH(256, 239)
codeword and the back of another twenty block-rows later, the `^ r` twist the specification credits
with a minimum distance of at least 42, Table 8's intra-block permutation and the four-subset
inter-block read-out. Its component generator is exactly the one `softfec.bch_code(8, 2)` builds.

**Bit-exact is a claim with a test behind it**: against the specification's own test vectors the
encoders reproduce TP5 from TP2 and the interleavers TP6 from TP5 with no bit different —
2,752,512 bits for DP-QPSK, 5,505,024 for DP-16QAM. The two places the text left open were settled
by the vectors, which only one reading reproduces: which encoder each bit of a codec block goes to,
and that TP6 is already merged four (or eight) bits at a time from the two interleavers. The vectors
carry no licence to redistribute, so they are not in this repository; point `MAIMAN_OFEC_VECTORS` at
them and the test runs, and without them a fingerprint of the output taken while it matched holds it
in place.

The decoder is this project's — the specification leaves decoding open — soft-in soft-out Chase on
each bit's two codewords:

```
   raw BER    3 passes, Chase-4    6 passes, Chase-6
   1.0 %      clean                clean
   1.5 %      18 in 1e5            clean
   2.0 %      1 in 80              17 in 1e5
```

The specification quotes 2.0e-2 with three iterations for its decoders; this one reaches it with
six passes and six test bits. The gap is the decoder's, not the code's.

### And the framing around it, symbol for symbol

`maiman.wport` is everything on either side of the code: clause 8 before it and clauses 11 and 12
after. The specification's compliance point is TP7 from TP0 — every test point between is an
implementation choice — and this reproduces all of them, for DP-QPSK and DP-16QAM alike.

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

**A DSP frame is 24 subframes of 7,296 symbols.** A pilot every 64 of them, from a PRBS10 reset at
each subframe's head; 11 training symbols at that head, the first of which *is* that subframe's
first pilot; and, in the first subframe only, a 22-symbol alignment word and 74 reserved symbols.
The budget closes exactly: 2,736 pilots, 240 training symbols that are not pilots, 22 + 74 of
alignment and reserve, and 172,032 payload — which is precisely what the symbol mapper made.

Two readings the text leaves open, and the vectors settle: the reserved symbols **straddle** the
pilot at slot 64 rather than displacing it, because the overhead goes in first and the pilot grid
second; and the pilot PRBS10's seeds appear at its output least significant bit first, which makes
its recurrence the stated polynomial's reciprocal. The CRC32 is checked twice over — against the
vectors, and against the catalogued check value `0xFC891918` that every CRC-32/BZIP2 publishes.

Not covered, and deliberately: the DPO path, FlexO-6(e)/8(e) with probabilistic constellation
shaping, whose shaping lookup table the published vectors here do not exercise — writing one from
memory is the thing `maiman.ofec` was careful not to do.

### And the shaping in front of it, on the DPO modes

`maiman.pcs` is clause 9: the transcoder that makes FlexO-6(e), FlexO-8e and FlexO-8 *DPO*
different from their DO namesakes. It does not change the constellation — the line is DP-16QAM
either way — it changes which of its points come up. Each 16QAM amplitude is one bit of a pair, and
shaping makes that bit the *inner* amplitude more often than the outer one:

```
   mode          b     shaping rate    P(inner)   net bits/symbol
   FlexO-6(e)    72       33.7 %        0.8265         2.594
   FlexO-8e     106       11.0 %        0.681          3.125
   FlexO-8      116        5.71 %       0.623          3.281
```

Those probabilities are not written anywhere in the code. The two lookup tables are bijections on
their words **ordered by falling weight**, so a field of nine bits reaches only the heavy half of a
ten-bit table — fewer indices, all of them low-energy words — and the measured probability falls out
of how far in the field can reach. The tests read it back off the shaper's output and compare with
Table 4.

The chain is the scrambled stream split a bit at a time between the four encoders, each share cut
into 84 coder blocks, and each block alternating chunks of shaping bits with runs of sign bits;
every chunk feeds four lookup arrays round robin; each array cuts its `b` bits into twelve fields
that index the tables **least significant bit first**; each 128-bit result is rewired — a different
permutation per polarization, quadrature and encoder — and the four are bit-interleaved into a
512-bit column. Checked against the specification's TP3 and TP4 for all three modes, bit for bit,
and inverted exactly.

**The tables are not in this repository.** `SCS_LUT10`, `SCS_LUT11` and the rewiring table are
normative data published inside the specification document, and their within-weight ordering is not
an enumeration anything here could reproduce — colex, lexicographic and numeric were all checked and
none of them is it. So they are handled the way the OFEC test vectors are: point
`MAIMAN_WPORT_TABLES` at a directory holding them and the shaper runs; without it, it raises and
says what is missing.

**And through the encoder's door.** Clause 9.2.4 permutes the last 35 bits of every codeword —
eighteen information bits and all seventeen parity ones — after the parity is computed. There are
four orderings and a codeword takes the one its index selects, `r % 4`. It is applied to the
encoder's *matrix*, not to its output, and that distinction is the whole of it: a row twenty back is
read as this one's front half, so by the time the parity that follows is computed it is parity over
permuted bits, and no reordering of the output can put that right. With it, `dpo_transmit` goes
**TP0 to TP7 bit for bit** for all three modes — the specification's own compliance point — reusing
the interleavers, the symbol mapper and the DSP frame the DO modes use, unchanged.

**And back again.** The same feedback that makes the permute work makes it interesting to undo:
un-permuting the whole stream restores every back half and breaks every front half, and leaving it
alone breaks the backs. So the decoder holds **both views of one matrix** — a codeword's front half
as the line carried it, because that is what its parity was taken over, and its back half untangled,
because the permute happened after. `dpo_receive` runs the whole path backwards, `ofec_decode_stream`
takes a `tail` for the part that matters, and the tests measure what each single view costs: on a
stream with nothing wrong with it, either one alone makes the decoder correct hundreds of bits that
were never wrong, and both together make it correct none.

**Both receivers can do better than a hard decision dressed as a soft one.** Left at their
defaults, `wport_receive` and `dpo_receive` slice the received amplitude and hand the decoder a
fixed confidence, sign only — which is what every test above measures, bit-identically to before.
Given `noise_variance`, they call `symbol_llr` instead: DP-QPSK and DP-16QAM are separable, every
one of `[XI, XQ, YI, YQ]` an independent pulse-amplitude channel under clause 11's labelling, so the
per-bit log-likelihood ratio is a one-dimensional Gaussian sum — exact (log-sum-exp) or max-log
(Tosato & Bisaglia, "Simplified soft-output demapper for binary interleaved COFDM with application
to HIPERLAN/2", ICC 2002). On the DPO modes that LLR is by default the *uniform*-prior one: the
shaper biases the amplitude bit toward the inner points (0.623, 0.681, 0.8265 by mode), which the
channel term alone does not use. `dpo_receive(..., shaping_prior=True)` adds it -- `ln (1 - p) / p`
to the magnitude bit, nothing to the sign -- and the ratio is checked against Bayes' rule summed
over the four points, and against the prior itself when the sample says nothing. It takes the axes
as independent, so it is the marginal's prior and not the joint's.

**Three things had to exist before any of this could be wired up**, and each is its own block:

`SoftDemapper` turns symbols into LLRs, max-log, with the noise variance estimated blind because a
receiver has no other option. `PortType.SOFT` is a sixth port type, so a soft output physically
cannot be wired into a block expecting bits — that mistake does not crash, it runs and quietly
gives back the decibels this whole path exists to recover, so the type system is what makes it
unrepresentable. And `SoftFECEncoder` is a *source* for the same reason the hard one is: a coder
makes more bits than it is given, the window is a fixed number of symbols on the line, so coding
shrinks the payload rather than speeding the line.

**The last block of a stream is provisional.** Every bit is checked twice — once by its own
stripe's rows, once transposed as a column by the next stripe — so the final block has half its
protection missing until the next one arrives. With two blocks in a window that is half the payload
and it sets a floor near 1e-4 that has nothing to do with the channel. It is why real decoders run
a sliding window and never emit the newest block, and there is a test on it rather than a footnote.

Getting the iteration right took three attempts, and all three failures are now tests. A
no-competitor reliability *below* the input's own made the extrinsic negative and destroyed bits the
decoder had never touched — 268 errors out of zero corrections, which is the signature that located
it. A loop that let each stripe overwrite the previous one's result discarded half of what the
braiding exists to produce. And a component decoder fed its own previous extrinsic agreed with
itself, so its output collapsed to zero and its corrections quietly unwound two passes later.

### How the flagship resolves its quarter turn

Every blind stage in a coherent receiver gets the phase right *modulo* the
alphabet's own symmetry — rotate a square QAM constellation by 90° and nothing about the received
samples changes, so no amount of data distinguishes the two. Something has to break the tie, and
there are exactly two ways: difference the quadrant out, or know some of the symbols.

The link used to do the first. `DifferentialDecoder` has to **slice** in order to difference, and
that is fatal to anything downstream that wants a measurement rather than a decision — the distance
from its output to the nearest constellation point is *exactly zero*, and a demapper reading it
returns LLRs of order 1e29.

It now does the second. `PilotInserter` overwrites every 64th symbol with one the receiver already
knows; `PilotPhaseRecovery` estimates `angle(Σ r·conj(p))` over those positions and removes one
constant angle from the whole window. It decides nothing.

```
                      differential            pilots
overhead              0 % of the rate         1.6 % of the rate
cost near threshold   2× the errors           none
EVM                   7.34 %                  7.31 %
counted BER           4.20e-10                3.56e-10
soft output           LLR ~1e29 (useless)     LLR ~121, real spread
blocks on the canvas  19                      18
```

**That table is the measurement taken when the change was made, and the canvas has moved since**:
soft-decision FEC arrived afterwards and took it to twenty. The EVM and BER columns are of that
link, not of the one the flagship is now — which reads 7.39 % and 5.27e-10, because it carries a
coder, a demapper and a decoder that the eighteen-block version did not. What the table is
evidence for is the *comparison*, both halves of which were measured on the same canvas on the same
day.

**It was cheaper in blocks, not just in errors.** Two disappeared with the differential
arrangement: a second mapper that existed only to hand the error analyser a non-differential copy
of the same bits, and a second analyser that existed only because an EVM taken after a decision
device reads zero however bad the link is. Neither had a reason to exist any more, so the canvas
lost a block while gaining a capability.

**Pilots replace payload, they do not add to it** — the window is a fixed number of symbols on the
line, so a pilot costs a data symbol. They are drawn from the alphabet's own **outermost** points,
which carry the most energy a phase estimate can be made from and, unlike an off-grid pilot, are
legal symbols: unit-power QPSK pilots inside a 16-QAM link sit exactly between four points, and the
error count then measures the pilots rather than the channel. The sequence is not constant either,
because a repeated symbol puts a line in the spectrum at the pilot rate.

The diagnostics report the **residual** — how far the removed angle sat from an exact multiple of
90°. Everything upstream is supposed to leave only an ambiguity, so a few milliradians says that
held. On the shipped link it is 5.4 mrad over 128 pilots. A large one would mean the constant
removed here was covering for a stage that did not do its job.

### What the flagship's post-FEC number measures

The shipped coherent link now runs end to end on soft-decision FEC: a staircase coder as the
source, pilots for the quadrant, a max-log demapper, and an iterative decoder. Twenty blocks, and
about a second.

**Its post-FEC error rate is dominated by the window's edges, and that is worth knowing before
reading it.** This link filters *circularly* — the root-raised-cosine shaping and the dispersion
compensator both wrap — so the first and last symbols carry a fold from the far end of the
sequence. The analyser is allowed to discard them and does, through `ignore_edges`. **A decoder is
not**: it has to decode every bit it is handed.

Measured on the shipped graph: all seven pre-FEC bit errors sat in the first 0.05 % and the last
0.07 % of the window. The interior was clean. A real stream has no edges, so this is an artefact of
simulating a finite window rather than anything about the link — and it is left visible rather than
buried under a longer window, because a post-FEC rate that quietly measures the simulation boundary
is the kind of number this project exists not to print.

**Pilots are erased, not believed.** A pilot overwrote whatever the coder put in that symbol, so
those bits say nothing about the codeword — and zero is exactly what that means. Not a confident
value: the receiver knows the pilot, but it is not the codeword bit, so anything confident is
confidently false. The difference is not subtle, measured on a 9.9e-3 channel:

```
pilots treated as       post-FEC BER
erasures (LLR = 0)        4.9e-04
ordinary bits             2.8e-02   ← worse than the decoder's own input
```

A code corrects roughly twice as many erasures as errors, and that is the whole of why.

**Carrying a block code quantises the window.** A staircase block is 16384 coded bits, so at four
bits per symbol the link runs in multiples of 4096 symbols and refuses anything else, naming a
length that works. That is a real reduction in what the sequence-length control accepts on the link
people open first — and it is not an implementation wart: a block code quantises a frame in
hardware too, and nobody runs an OTN link at 256 symbols either.

### A coherent link that runs on soft decisions

`examples/maiman/coherent_sdfec.maiman` — open it from the Examples menu. 32 GBd 16-QAM, the staircase code at
16.4 % overhead, decoded on log-likelihood ratios from a max-log demapper rather than on bits:

```
pre-FEC BER   post-FEC BER   blocks
  7.36e-03         0            4
```

At −25 dBm the line is running at a few times 1e-3 — past what RS(255,239) can touch — and the
payload comes out exact.

**The flagship carries this too, now.** This project stays because it measures a different thing:
an ideal laser and no fiber, so what it reports is the decoder. The flagship has a real span and
real carrier recovery, and its post-FEC number is dominated by the simulation window's edges —
see [What the flagship's post-FEC number measures](#what-the-flagships-post-fec-number-measures).

So the coded link declares an **ideal carrier** — both lasers at 1550 nm, no linewidth — which
removes the ambiguity rather than resolving it. A real system pairs soft FEC with pilot symbols,
which this library does not have; that absence is why the idealisation is declared here rather than
worked around quietly. Carrier recovery is demonstrated on the flagship, where it belongs.

**Four blocks, not two.** Every bit is checked twice — by its own stripe's rows and, transposed, by
the next stripe's — so the last block of a stream has half its protection missing until the next
one arrives. At two blocks that is half the payload, and it floors the error rate near 1e-4 for
reasons that have nothing to do with the channel.

### What EDFA saturation is for

Until now the EDFA's docstring said, in its own words, that **saturation was a clamp and not a
model**: `max_output_power` held the output at a ceiling and the gain fell to whatever achieved
it. That has a kink in it. Below the ceiling the amplifier was perfectly ideal; above it the gain
fell as `1/P_in`. Real amplifiers do neither — they compress from the first photon, smoothly, and
never hit a wall.

The replacement is the standard steady-state description (Saleh, Jopson, Evankow and Aspell,
*IEEE Photon. Technol. Lett.* 2(10), 1990), which is implicit in the gain:

```
G = G₀ · exp( −(G − 1) · P_in / P_sat )
```

solved exactly — Newton on `ln G`, to 1e-12 — rather than approximated. The root is bracketed by
`[0, ln G₀]` and the function is increasing and convex, so Newton started at the upper end descends
onto it monotonically: no bisection fallback and no iteration cap that could quietly hand back a
half-converged gain. A test puts the answer back into the equation, because that is the only honest
check on a solver for an implicit relation.

**The parameter is the one a datasheet quotes** — `saturation_power` is the *output* power at 3 dB
of gain compression — and is converted to the model's own `P_sat` by `P_3dB·(G₀−2)/(G₀·ln2)`, which
follows from setting `G = G₀/2` above. For a large gain that is `1.443 · P_3dB`; the `(G₀−2)/G₀`
factor is carried in full because it is 0.98 at 20 dB of gain and 0.80 at 10, and a version that
dropped it as "large gain" passes a 20 dB test and fails a 10 dB one. Both are in the suite.

A 20 dB amplifier with a +17 dBm saturation power:

```
in dBm   out dBm   gain dB   compression
   -40    -20.00     20.00      0.00 dB
   -20     -0.06     19.94      0.06 dB
   -10      9.46     19.46      0.54 dB
     0     16.99     16.99      3.01 dB   ← the declared point, by construction
    10     21.65     11.65      8.35 dB
```

**Saturation is driven by total power, ASE included**, which is what makes an EDFA behave like a
fixed power source rather than a fixed gain. Feed one 20 dB amplifier more channels at −5 dBm each
and watch them share:

```
channels   total out   per channel
    1       13.61 dBm   13.61 dBm
    2       15.75 dBm   12.74 dBm
    4       17.58 dBm   11.56 dBm
    8       19.15 dBm   10.12 dBm
```

Eight times the input for 5.5 dB more output. Every doubling of the channel count costs the
channels already there about a decibel, which is the arithmetic a WDM operator plans around, and it
falls out of the model rather than being put in — the amplifier never sees channels, only the
power they add up to. A version that saturated each band against its own copy of the amplifier
would pass a single-channel test and be wrong here, so the suite checks it through a real graph
with a real combiner and not only through the gain function.

**It is off by default.** An amplifier that never compresses is an idealisation, and it is now
declared as one the way a zero-linewidth laser is: `saturate` is a flag and it starts false. That
keeps every existing result in this repository exactly where it was — the change moved the
manifests and not one number — and makes turning it on a decision someone takes. What it replaces
was worse than an idealisation, because a clamp claims to be a saturation model and is not.

`max_output_power` is retired. Old projects still open, with the parameter dropped — and unlike the
other retirements in this library that **does** change the link they describe, which is stated in
the code rather than glossed. The value it carried was fitted to a fiction.

### What happens between two steady states

The gain above is the one the erbium *settles* into. Getting there takes milliseconds, and what a
line system actually fears is the trip rather than either endpoint: an erbium amplifier shares one
inversion between everything passing through it, so take channels away and the survivors inherit
the gain those channels were using.

**This is an analysis, not a block, and the reason is a measurement.** The relaxation constant is
`τ / (1 + P_out/P_sat)` — for the 20 dB amplifier here, between 3.3 and 9.9 ms. A simulation window
of 4096 symbols at 32 GBd is **128 ns**. The fastest transient is twenty-five thousand windows
long, so within one window the gain is a constant, which is exactly what `EDFA` already assumes and
is right to. A per-sample block could not show any part of this.

So there is no `metastable_lifetime` parameter on the amplifier. **A parameter that cannot change
the result of a run is a control the interface cannot back**, and this library does not ship those.
The lifetime belongs to [`maiman.transient`](../src/maiman/transient.py), which can use it.

**It is the dynamic form of the equation already in the component, not a second model beside it.**
Writing the Saleh reservoir in terms of `g = ln G`:

```
τ · dg/dt  =  ln G₀ − g − (e^g − 1)·P_in/P_sat
```

Set `dg/dt = 0` and it returns `G = G₀·exp(−(G−1)·P_in/P_sat)`, which is what `effective_gain`
solves by Newton. Same `G₀`, same `P_sat`, exactly one quantity added — so a test integrates the
ODE from a long way off and checks it lands on the component's own solver. Two code paths sharing
no arithmetic, meeting to **1e-9**.

Linearising about that rest point gives the effective constant, which is why a saturated amplifier
answers in a fraction of its own lifetime: the stimulated emission draining the reservoir is itself
proportional to how full it is. That closed form is tested against a step response measured from
the integrator, to a part in a thousand.

`python examples/python/edfa_transient.py` drops seven channels of eight from a −6 dBm comb:

```
                      one amplifier      down a chain of 8
gain before                15.63 dB
gain after                 18.84 dB
survivor excursion         +3.21 dB               +9.94 dB
settles in                 47.9 ms
```

**The chain is the number that matters**, and it accumulates *sub*-linearly rather than as eight
times the first: each amplifier down the chain starts less saturated than the one before, so it has
less compression left to give back. Eight times 3.21 dB would be 25.7; the measured figure is 9.94.
Chaining is exact rather than approximate — there is no feedback from a later amplifier to an
earlier one, so a span integrates in order, each taking the previous one's output on the same grid.

**What is absent.** One reservoir means one gain for every wavelength, and a real erbium transient
*tilts* the gain spectrum as the inversion changes, so a survivor's excursion depends on where in
the band it sits. The pump is implicit in `G₀` and does not respond — a deployed amplifier has a
control loop pushing back on exactly this excursion, and what is computed here is the uncontrolled
case, which is the one such a loop is sized against. Both are real and both are missing rather than
approximated.

### A drop tilts the spectrum

One reservoir is one average inversion, and an inversion that moves takes every wavelength's gain with
it — by a different number of decibels at each. For a homogeneously broadened medium the gain in dB is
linear in the inversion, `G(λ, n) = n (A + G*) − A`, where `A` and `G*` are the coil's unpumped loss
and fully inverted gain: the Giles parameters a doped-fibre datasheet publishes. `ErbiumSpectrum`
takes them as given, or derives the emission curve from the absorption curve by McCumber's relation.

`spectral_gain_transient` integrates the same reservoir `gain_transient` does, reads its gain at the
amplifier's centre wavelength as the inversion, and spreads that across the spectrum. At the centre
wavelength it is the single-reservoir answer to 1e-10 dB, and every other channel swings by the tilt
`(A + G*)(λ) / (A + G*)(λ_ref)` times it, at every instant:

```
   8 channels at -6 dBm, 7 dropped          (an illustrative spectrum, not a fibre)
   channel   tilt   excursion
   1530 nm   1.92    +6.18 dB
   1540 nm   1.50    +4.81 dB
   1550 nm   1.00    +3.21 dB    the single-reservoir answer
   1560 nm   0.84    +2.70 dB
```

A link designed around the centre wavelength's 3.2 dB under-protects its short-wavelength receivers by
three decibels. The tilt does not depend on the inversion, which is why it is one curve per amplifier,
measured once. `examples/python/edfa_gain_tilt.py` prints the table.

Still not modelled: channels at different wavelengths drain the inversion at their own cross sections,
where the reservoir here is driven by total power; and the pump control loop that pushes back.

### The loop that answers it

No line system lets its survivors rise by three decibels when a fibre is cut. It measures its own
gain and moves the pump, and what is left of the excursion is the loop failing to keep up.
`controlled_gain_transient` makes the pump a state rather than a constant and integrates an integral
controller beside the inversion — integral, because that is what leaves **no steady-state error**: the
gain returns to the setpoint rather than near it.

```
   7 channels of 8 dropped; uncontrolled the survivors rise 3.21 dB
   loop        excursion   left of it
   10 Hz         +2.15 dB       67 %
   100 Hz        +1.12 dB       35 %
   1 kHz         +0.43 dB       14 %
   10 kHz        +0.15 dB        5 %
```

The erbium answers in milliseconds, so a loop a decade faster than that removes most of the excursion
and one a decade slower watches it happen. A loop can also run out: holding *output power* through a
nine decibel drop needs nine more decibels of gain, and with a 25 dB pump ceiling the output ends
2.3 dB below its setpoint and `pump_limited` says so. A ceiling below the amplifier's own small-signal
gain is refused outright, because a loop that starts against its limit was never a loop.

**And a watt is not a watt.** Stimulated emission empties the reservoir once per photon, so a channel
drains it by its photon flux times its cross section: 1.90× at 1530 nm against 0.85× at 1560, on the
same spectrum the tilt comes from. Pass `channel_powers` and the reservoir is driven by that weighted
sum, so dropping the short-wavelength half of a comb is a larger disturbance than dropping the
long-wavelength half of equal power. A comb sitting at the reference wavelength weighs exactly one,
which is the single total this model was always driven with. `examples/python/edfa_pump_control.py` prints
all three tables.

### Its own noise is a load

Saleh's equation is the reservoir's energy balance: `P_sat ln(G₀/G)` is everything the inversion
hands out, output minus input, over every field in the fibre. The amplifier's own spontaneous
emission is one of those fields — with no input, and leaving by both ends. Set `self_saturation`
(it applies once `saturate` is on) and it is counted. Each end carries exactly what the block emits
into the link, `2 S_ASE B = NF·G·hν·B` over both polarizations, so:

```
ln(G₀/G) = [(G − 1)·P_in + 2·NF·G·hν·B] / P_sat
```

With no input at all that has a closed form, `G = W(βG₀)/β` in Lambert's W, where
`β = 2·NF·hν·B/P_sat`. That is the gain a coil can hold before its own noise empties it. The test
solves W by Halley's iteration, which shares nothing with the component's Newton on `ln G`, and
the two agree to 1e-12. With the 17 dBm, 5 dB, 4 THz defaults:

```
   small-signal   gain in the dark   forward ASE    cost at −40 dBm in
   20 dB             19.98 dB          −7.9 dBm        0.02 dB
   30 dB             29.81 dB          +1.9 dBm        0.19 dB
   35 dB             34.46 dB          +6.6 dBm        0.54 dB
   40 dB             38.59 dB         +10.7 dBm        1.38 dB
```

At 0 dBm input every one of them is within 0.02 dB of the model without it. Its own ASE matters
where the amplifier is lightly loaded and its gain is high, and that is where it was missing.

The transient integrators carry the same term. So an amplifier whose channels are all dropped
comes to rest on Lambert's gain rather than on `G₀`, and it answers faster than `τ` even in the
dark, at `τ/(1 + (P_out + P_ASE)/P_sat)`. A test measures that rate against the closed form to a
part in a thousand. The block emits that ASE flat across its band and, knowing no cross
sections, drains the reservoir with it at the centre wavelength's rate. Given an erbium spectrum,
`spectral_gain_transient` weighs each slice by its own cross section and photon energy instead,
the way it weighs channels. A linear tilt barely moves the result — across a band centred on the
reference it gains on one side what it loses on the other, 3e-4 at 0.2 dB/nm — but an absorption
peak on the short side, which is where erbium's sits, puts the load 11 % above the centre's. The
flag is off by default, because it moves every saturated amplifier's gain.

The `EDFA` block itself still emits that ASE flat, in every run — it is the component, not the
analysis, that knows no cross sections. `transient.spectral_ase_noise_bin` reshapes the same total
power (the noise figure's, unchanged) into the standard steady-state spectrum instead,
`n_sp(λ)·hν·(G(λ)−1)` per slice (Giles and Desurvire, *J. Lightwave Technol.* 9(2), 271, 1991;
Desurvire, *Erbium-Doped Fiber Amplifiers*, Wiley, 1994) — an offline reshaping of the block's own
output, carried as a `NoiseShape` so total power is untouched and only where in the band it sits
moves. A flat spectrum gives a shape of exactly one and the reshaped bin is bit-identical to the
unshaped one. `spectral_gain_transient(..., spectral_ase=True)` makes the reservoir's own drain
consistent with that same shape, through `self_saturation_weight_spectral`, rather than the cross
section alone. Both off by default.

### What RIN is for

Every noise in this project until now got *quieter*, relative to the signal, as the launch power
went up. Shot noise grows as the square root of the current and thermal noise does not grow at all,
which is why every sensitivity curve here keeps improving to the right. **Relative intensity noise
does not work like that.** It is a fluctuation *of the power itself*, so it scales with the signal,
and the ratio it fixes is one that no power budget moves.

`CWLaser` now takes a `rin` parameter in dB/Hz — the single-sided spectral density of the
fractional power fluctuation, `S_δP(f) / P̄²`, which is the quantity a laser datasheet quotes.
A sampled window carries a one-sided bandwidth of `fs/2`, so the fractional fluctuation per sample
is drawn from `N(0, rin·fs/2)` and the field is the square root of what is left (Agrawal,
*Fiber-Optic Communication Systems*, 4th ed., §4.6.2).

**The closed form is exact and the test is an equality.** Detect a CW laser with shot and thermal
noise off, and what is left in the photocurrent is the laser's own intensity noise:

```
RIN         measured SNR   1/(RIN·B_n)   the same, at ten times the power
-155 dB/Hz    56.264 dB      56.278 dB          56.264 dB
-145 dB/Hz    46.265 dB      46.278 dB          46.265 dB
-135 dB/Hz    36.268 dB      36.278 dB          36.268 dB
```

`B_n` is the receiver filter's *noise* bandwidth, which for the Gaussian shape used here is
`1.0645 × B` in closed form rather than approximately — which is why this can be checked as an
equality (0.014 dB) instead of an order of magnitude. The third column is the point: **+10 dBm
gives bit-identical SNR**. A model that got the variance right and this wrong would pass a
variance check and be useless for the one question RIN is asked.

**On a link it shows up as a penalty that grows with power.** 10 Gb/s OOK, PIN with shot and
thermal noise on, Q in dB against an otherwise identical ideal laser:

```
received      ideal Q  |  penalty at -155  |  at -145  |  at -135 dB/Hz
 -20 dBm       8.06 dB |      0.00 dB      |  0.00 dB  |    0.01 dB
 -15 dBm      17.98 dB |      0.00 dB      |  0.01 dB  |    0.13 dB
 -10 dBm      27.75 dB |      0.01 dB      |  0.12 dB  |    1.04 dB
  -5 dBm      36.49 dB |      0.08 dB      |  0.77 dB  |    4.30 dB
   0 dBm      41.54 dB |      0.24 dB      |  1.99 dB  |    7.63 dB
```

At −20 dBm the columns are indistinguishable — thermal noise is a hundred times larger and the
laser's intensity noise is invisible. Every 5 dB of extra power makes it more visible, not less,
until at 0 dBm a −135 dB/Hz laser has given back 7.6 dB of the Q the same link would have had.
The penalty is quoted as a difference at equal power on purpose: it does not depend on whatever
else limits the ideal column at the top of the sweep.

`rin = 0` is a **sentinel for an ideal laser, not 0 dB/Hz of noise** — 0 dB/Hz would be a
fractional intensity variance of one per hertz of bandwidth, which is not a laser. Every real
device sits between about −110 dB/Hz for a cheap Fabry-Perot and −165 dB/Hz for a good DFB, so the
top of the scale is free to carry the sentinel. It is the default, and the shipped links keep it:
turning it on by default would move every number in this file, which is a decision for whoever
wants the noise and not for the parameter's default.

The phase and intensity noises are drawn from **separate streams**, so sweeping the linewidth does
not move the intensity samples. Otherwise a plot of one against the other shows a coupling that is
not there.

### What timing recovery is for

Everything downstream of the receiver used to sample on a grid it was *given*. `IQSampler` takes
one column out of every symbol and the column is a parameter — exact when the transmitter's symbol
grid and the receiver's sample grid are the same one, which in a simulation they are by
construction, and wrong the moment anything in the path holds a delay that is not a whole number of
samples.

**One shipped block is enough to do it.** A silicon waveguide at a group index of 4.2 holds 14 ps
per millimetre. At 32 GBd a symbol is 31.2 ps, so a quarter of a millimetre is a ninth of one:

```
waveguide     delay | EVM off   errors off | EVM on    errors on     moved
      0 um   0.00 ps |   6.44%     0/1920  |   6.44%     0/1920    +0.00 ps
    250 um   3.50 ps |  18.39%    64/1920  |   6.43%     0/1920    -3.52 ps
    500 um   7.00 ps |  37.06%   675/1920  |   6.45%     0/1920    -7.00 ps
   1115 um  15.61 ps | 110.20%  1527/1920  | 2993.36%  1806/1920  +15.62 ps
```

The method is **one FFT bin**. A modulated signal is cyclostationary, so `|A|²` carries a line at
the symbol rate; the phase of that line *is* the timing — Oerder and Meyr, 1988. Nothing is
searched and nothing iterates. The correction is a phase ramp, which is an exact fractional delay
for a periodic band-limited window rather than an interpolation with a passband to argue about,
and the test asserts that forward-then-back returns the input to 1e-12.

**Half a symbol is the cliff, and past it the failure is a different one.** At 1115 µm the delay is
exactly half a symbol: both directions are the same distance, so whichever it takes puts the
instant in the right place and the *sequence* one place out. The constellation is fine and the
errors are total, which is the signature of a slip rather than of bad timing. No estimator reading
`|A|²` can do better — delay by a whole symbol and the intensity waveform is **identical**, so the
line has the same phase. That is framing, not timing, and it is resolved against a known reference
exactly as the quadrant ambiguity is.

The estimate carries the **magnitude** of the line it came from, because the phase of nothing is a
number like any other: a signal shaped to exactly the Nyquist bandwidth carries no symbol-rate line
at all, and a heavily dispersed one nearly none.

### What frequency recovery is for

A transmitter laser and a local oscillator are independent oscillators, and they are not on the
same frequency. This library already modelled that honestly — bands carry their own centre
frequency, so an LO tuned off the transmitter beats against the signal inside `CoherentReceiver`
exactly as two real lasers do. Nothing removed it blind until this block; the analyser removed it
*data-aided*, with the transmitted sequence in hand, which is what a bench instrument does and not
what a receiver can.

**How little it takes.** On the shipped 32 GBd 16-QAM link, 10 MHz — three hundredths of one per
cent of the symbol rate — is enough:

```
  offset | errors, no block | errors, block |  found offset | confidence
       0 |      0/1920      |    0/1920     |    +0.000 MHz |    21.7
  10 MHz |   1677/1920      |    0/1920     |   +10.254 MHz |    22.1
 100 MHz |   1796/1920      |    0/1920     |  +100.098 MHz |    21.6
   1 GHz |   1791/1920      |    0/1920     | +1000.000 MHz |    22.3
 3.9 GHz |   1790/1920      |    0/1920     | +3899.902 MHz |    22.0
```

EVM is left out of that table on purpose: once the constellation is spinning, the analyser's
common-gain fit has nothing to fit and the percentage it reports is not a measurement. The error
count still is.

For scale: a laser on the ITU grid is specified to ±2.5 GHz and a good tunable holds ±100 MHz, so
the offset a receiver actually meets is two to three orders of magnitude past the point where the
link stops working.

**The method is one FFT**, of the symbols raised to their own alphabet's rotational symmetry —
Leven, Kaneda, Koc and Chen, *IEEE Photon. Technol. Lett.* 19(6), 2007. Raise a symbol to that
power and every point that differed only by one of those turns lands in the same place: the
modulation becomes a constant and the offset becomes a tone at M times itself. The power is taken
from the geometry by the same `rotational_symmetry()` that bounds the blind *phase* search, which
is the only way the two agree — 4 for a square constellation, 2 for a rectangular one or for BPSK.

That makes the acquisition range fall out of the format, and a rectangular format wins twice:

```
format   M   unambiguous range   errors at 1 GHz off/on   confidence
QPSK     4        ±4 GHz              1448  ->  0            197
8-QAM    2        ±8 GHz              1671  ->  0             34
16-QAM   4        ±4 GHz              1791  ->  0             22
64-QAM   4        ±4 GHz              1888  ->  80            16
```

64-QAM's 80 is not residual offset — it is that link's own noise floor, which is 74 errors with no
offset at all. The confidence column falls with order because a higher-order alphabet strips less
cleanly, and it is reported rather than hidden: it is the spectral peak over the median, so a value
near 1 says the argmax found no line and the megahertz beside it are an accident with decimal
places.

**This is not the phase problem, and the order is not a preference.** `CarrierRecovery` removes a
phase that *walks*; this removes one that *ramps*. A phase search covers a quarter turn and averages
over a window, so a ramp steep enough to cross that quarter turn inside the window makes it slip
rather than track — and a slip looks like noise. Every deployed coherent receiver puts frequency
before phase.

**Past half the stripped bandwidth it aliases rather than degrades**, which is the failure worth
knowing because it does not look like one. Beyond `symbol_rate / (2M)` the tone wraps, and the
estimate comes back wrong by exactly `symbol_rate / M` **with the same confidence as a correct
one**: 4.1 GHz is reported as −3.9 GHz, at a confidence of 23.1 against 24.0 for a correct
estimate. A test asserts the aliasing explicitly, rather than leaving it as a range nobody checks.

Something with no rotational symmetry at all has no power that strips it, so it is refused at run
time rather than answered with an argmax over noise.

### Acquisition, and why it cannot be a sweep

The obvious way to widen that ±4 GHz window is to try candidate offsets and keep whichever looks
best. **It cannot work, and the reason is worth stating because it is not an implementation
limit.** An alias of `symbol_rate / M` is a phase advance of exactly one `2π/M` turn per symbol —
and `M` is *defined* as the turn that maps the alphabet onto itself. So the aliased constellation is
not merely similar to the correct one, it is identical, symbol for symbol. A sweep would be
searching for a difference that is not there. The ambiguity belongs to the alphabet, not to the
estimator.

So [`CoarseFrequencyRecovery`](../src/maiman/components/dsp.py) does not look at the symbols at all. It
takes the **first circular moment of the received waveform's power spectrum** — where the band
*is*, before any of the modulation has been stripped:

```
offset = angle( Σ S(f)·exp(2πjf/fs) ) / 2π · fs
```

Circular rather than an ordinary centroid, and both reasons carry weight. A band straddling the
`+fs/2` wrap has no meaningful arithmetic mean and an exact circular one. And **a flat noise floor
contributes nothing to a circular moment**, because a constant spread evenly around the circle sums
to zero — so no floor has to be estimated, thresholded or subtracted, and *the block never has to
be told the signal's bandwidth or roll-off*. That last part is not a convenience. The obvious
alternative — slide a window of the signal's width and take the position holding the most energy —
must be told that width, and any window wider than the band sits on a plateau of equal energy whose
argmax is arbitrary: measured, assuming `1.6·R_s` for a band that was `1.2·R_s` put the answer
**3.9 GHz out**.

```
                       fine (M-th power)        coarse (circular moment)
reads                  symbols                  the waveform
unambiguous over       ±symbol_rate/(2M)        ±fs/2 — which is Nyquist, not the method
                       = ±4 GHz at 32 GBd       = ±256 GHz on the shipped link
accuracy               sub-MHz                  ~±150 MHz
past its range         confidently wrong        correct — the alias *is* the sampling
confidence figure      peak/median              resultant length, 0…1
```

**The two failure modes are opposites, and that is the whole argument for having both blocks.** The
fine stage's alias destroys the link and reports a healthy confidence while doing it. The coarse
stage's "alias" past `fs/2` is not an error at all: beyond Nyquist an offset *is* the wrapped value
in sampled data, and correcting to it is correct. Measured on the shipped link at 512 GHz sample
rate, +258 GHz reads as −254 GHz and the link still recovers to 7.50 % EVM with zero symbol errors.

**It has to run before dispersion compensation and before the matched filter**, and its port types
make it impossible to wire anywhere else — it takes a waveform and returns one. Both of those
stages are built around baseband: the compensator applies a quadratic phase measured from zero
frequency, and the matched filter is a root-raised-cosine centred there. A band sitting 20 GHz away
gets the wrong phase curve and is then largely filtered off.

Measured end to end on the shipped coherent link, detuning the LO and running the whole chain:

```
LO offset    coarse stage lands    fine stage residual    EVM      symbol errors
+20 GHz      +19.915 GHz           +85 MHz                7.41 %   0
+60 GHz      +59.890 GHz           +109 MHz               7.37 %   0
+200 GHz     +199.903 GHz          —                      7.41 %   0
```

Back to back is 7.39 %. Without the block the chain is broken past 4 GHz — at 4.1 GHz the same link
reads 6002 % EVM.

**It is deliberately coarse**, and the accuracy is set by the data's own spectral asymmetry rather
than by noise: a finite draw of random symbols does not have an exactly symmetric spectrum, and
±150 MHz over a 4096-symbol window is whatever that asymmetry is. That is ~25× finer than the range
the fine stage needs, which is the entire requirement — this block does not have to be accurate, it
has to be *unambiguous*. It also outlives the link: at −14 dBm launch the constellation is at 117 %
EVM and the acquisition estimate is still right to 226 MHz.

The **concentration** figure it reports is the resultant length of that same moment, and unlike the
M-th power's peak-to-median it is informative about the thing that actually goes wrong here, since
what degrades a circular mean is the spectrum ceasing to be a band. Walking the launch power down:
0.98 at +10 dBm, 0.88 at 0, 0.66 at −6, 0.43 at −10, 0.23 at −14.

`python examples/python/acquisition_link.py` runs one link with a matched filter in it through both
arrangements and prints the two tables side by side. Its middle rows are the interesting ones: the
fine stage reports −3.900 GHz for a +4.100 GHz offset at a confidence of 19.6, where a correct
estimate on the same link scores 23.2. Further out — at 20 and 100 GHz — the confidence does
collapse, to 4.4 and 3.3, but that is the matched filter having already destroyed the signal rather
than the estimator noticing anything. **Near the fold, which is where it matters, confidence does
not separate a right answer from a wrong one.**

**A large offset is not automatically a hard one**, which a test records so nobody re-reads it as
one. On a link that samples at the symbol rate with no matched filter, the sampler itself folds the
offset by `symbol_rate` before the fine stage sees it — at 60 GHz that fold lands on −4 GHz, inside
the fine stage's range, and at 64 GHz, exactly twice the symbol rate, it lands on zero. Both come
back with zero symbol errors and no acquisition stage at all. What makes an offset hard is a stage
downstream that is built around baseband.

### Both are in the reference design

The coherent link the studio opens with carries both stages, and its LO is tuned 200 MHz below the
transmitter. That is not an injected impairment — it is the removal of an idealisation. Two
free-running lasers are never on the same frequency to fifteen decimal places, and a good tunable
holds about this much; modelling them as exactly co-tuned was the unrealistic choice, in the same
way that modelling a zero-linewidth laser would be. The link already runs a realistic 100 kHz
linewidth for exactly that reason.

200 MHz is six thousandths of one per cent of the symbol rate. Here is what each stage is worth on
that link, with the detuning and without it — every row a real run of the shipped graph:

```
   LO      timing  frequency |    EVM      SNR      symbol errors
co-tuned     off      off    |   7.318%   22.71 dB      0/3968
co-tuned     on       on     |   7.387%   22.63 dB      0/3968
detuned      off      off    | 285.219%   -9.10 dB   3334/3968
detuned      on       off    | 292.993%   -9.34 dB   3473/3968
detuned      off      on     |  13.006%   17.72 dB      0/3968
detuned      on       on     |   7.342%   22.68 dB      0/3968
```

Three things are worth reading off it. **Frequency recovery is the difference between a link and
no link** — 3334 errors to none. **Timing recovery is worth 5.7 points of EVM, but only once the
offset is gone**: on its own it does nothing for a detuned link, because a spinning constellation
is not a timing problem. And **the two together return the detuned link to the co-tuned one**,
7.342 % against 7.318 %.

That last gap is the honest cost of the stages: on a link with an ideal LO they find 2.12 ps and
+0.000 MHz and correct them, and the correction is very slightly worse than leaving it alone —
0.07 points of EVM. A blind estimator has variance, and paying it is what buys the 278 points in
the row above. A receiver carries these stages because the impairment is normally there, not
because they are free when it is not.

### What carrier recovery is for

With ordinary 100 kHz lasers and no phase recovery, 16-QAM at 32 GBd does not close — and, more
tellingly, **launching more power stops helping**:

| launch | without recovery | with recovery |
| ---: | :--- | :--- |
| −14 dBm | 14.6 dB, BER 6e−3 | 22.1 dB, BER 4e−9 |
| −10 dBm | 15.1 dB, BER 4e−3 | 25.7 dB, BER 4e−18 |
| −6 dBm | 15.3 dB, BER 4e−3 | 28.7 dB, BER 1e−34 |
| −2 dBm | 15.9 dB, BER 2e−3 | 30.9 dB, BER 2e−56 |

Twelve dB of extra power buys 1.3 dB. Laser phase noise is a random walk, so it is not removable by
subtracting a constant or a line, and it puts a ceiling on SNR that no power budget lifts.
[`CarrierRecovery`](../src/maiman/components/coherent.py) removes the ceiling using the blind phase
search of Pfau et al. Both halves of that claim are [asserted](../tests/test_dsp.py) — the second
would be meaningless without the first.

### Reaching past the bench

Every coherent example above was back to back. `python examples/python/dispersion_link.py` puts the same
32 GBd 16-QAM link through real fiber, with loss and nonlinearity switched off so that chromatic
dispersion is the only thing acting:

```
            accumulated   spread          uncompensated              compensated
  span          [ps/nm]  [symbols]      EVM       SNR    errors      EVM       SNR
       0 km          0       0.0      1.68%    35.51 dB      0/3968     1.68%    35.51 dB
       5 km         85       0.8     17.04%    15.37 dB      5/3968     1.67%    35.52 dB
      20 km        340       3.3    217.64%    -6.75 dB   3336/3968     1.67%    35.56 dB
      80 km       1360      13.4   2542.45%   -28.11 dB   3698/3968     1.68%    35.52 dB
     400 km       6800      67.0   2686.86%   -28.58 dB   3692/3968     1.67%    35.55 dB
    1000 km      17000     167.4   3390.48%   -30.61 dB   3697/3968     1.68%    35.51 dB
```

Read the uncompensated column first, because it is the reason this block exists. **Five kilometres
— a metro hop — costs twenty decibels.** At 80 km the link is not degraded, it is gone: 3698 of
3968 symbols wrong is 93%, and blind guessing on 16-QAM gives 93.75%. The whole coherent phase had
been validated without ever meeting the impairment that dominates every real span.

The compensated column is the argument for coherent detection in one line. **Back-to-back quality
at every distance, with no penalty that grows with it.** Dispersion is an all-pass phase — it
rearranges the field in time and removes nothing — so a receiver that measures the field still
holds all of it, and one static filter puts it back. That is not error correction; it is inverting
an invertible operation. A direct-detection receiver squares the field at the photodiode, destroys
the phase, and can never do this at all, which is why it has to carry dispersion-compensating fiber
in the line instead.

The setting is sharp, and that is worth seeing rather than being told:

| compensator set to | error | EVM | SNR |
| ---: | ---: | ---: | ---: |
| 76 km | −68 ps/nm | 13.64% | 17.30 dB |
| 79 km | −17 ps/nm | 3.78% | 28.46 dB |
| **80 km** | **0** | **1.68%** | **35.52 dB** |
| 81 km | +17 ps/nm | 3.77% | 28.47 dB |
| 84 km | +68 ps/nm | 13.64% | 17.30 dB |

Being one kilometre out costs 7 dB. The symmetry of those flanks is also the sign check: a
compensator applying its correction the wrong way round would put the nominal setting at *twice*
the span, and the two sides would not match to three digits.

**Why this is a separate stage from the butterfly equaliser.** Both are linear filters, so one
adaptive filter could in principle do both jobs. It does not work. On the dual-polarization link
over 80 km, growing the butterfly from 7 taps to 65 — the longest the block allows, and nine times
the cost — leaves the link just as dead, because a blind modulus criterion has no gradient to
follow once the constellation is smeared into a Gaussian blob. The static block ahead of a 7-tap
filter restores back-to-back quality outright. Dispersion is static and *long*; polarization
mixing is fast and *short*; one filter serving both would have to be both, which is the worst of
each. That ordering is [asserted](../tests/test_cd_compensation.py), not quoted.

### Dual polarization

`python examples/python/dualpol_link.py` puts two independent 16-QAM tributaries on orthogonal
polarizations of one wavelength — **256 Gb/s** — and rotates the state the way a fibre does:

```
rotation    without equaliser              with equaliser
    0 deg  EVM    2.5 /    2.5 %      0 err      EVM 2.50 / 2.57 %    0 err
   15 deg  EVM   28.1 /   29.2 %   1416 err      EVM 2.51 / 2.58 %    0 err
   30 deg  EVM  218.0 /  123.5 %   6217 err      EVM 2.52 / 2.51 %    0 err
   45 deg  EVM  278.5 /  365.2 %   6763 err      EVM 2.48 / 2.55 %    0 err
   72 deg  EVM   85.7 /  165.6 %   5523 err      EVM 2.54 / 2.56 %    0 err  (swapped)
   90 deg  EVM    2.5 /    2.5 %      0 err      EVM 2.54 / 2.48 %    0 err  (swapped)
```

Past a few degrees the unequalised branches are not degraded — they carry no recoverable data at
all, because each is a *mixture* of both tributaries. The
[butterfly equaliser](../src/maiman/dsp.py) separates them blind, with no training sequence anywhere in
the link. Read the two end rows together: 90° is a clean swap rather than a mixture, so it needs no
equaliser at all and simply delivers the tributaries the other way round — which is also why
nothing blind can label them, and why a real link recovers the pairing from framing.

This is the increment that finally exercises `Ey`, which has been in the signal model since the
first commit for exactly this purpose.

### Shaping and differential encoding

Two transmitter refinements that close out the coherent phase.

**Root-raised-cosine shaping** bounds the spectrum. A held symbol has a sinc spectrum that never
ends — fine for one channel alone, useless once neighbours are packed onto a grid. At a 0.2
roll-off, **99.5%** of the shaped power falls inside ±0.6 symbol rates, against 83.6% for a held
symbol. The shaping is split into a root at each end, so the cascade is Nyquist (zero at every
symbol instant but its own) *and* the receiver's filter is matched to the transmitted pulse. One
end alone gives neither.

A shaped waveform overshoots between symbols, so its peak-to-average ratio is higher than the
constellation's and a full-swing drive clips — about 7% EVM at 16-QAM, falling to 1% backed off.
That is real, and backing off is what a transmitter does about it.

**Differential quadrant encoding** closes the quarter-turn ambiguity that every blind stage leaves
behind: the phase search cannot resolve it, and neither can the butterfly equaliser. Under plain
Gray labelling a quarter turn permutes the bits differently for every point; under a
quadrant-relative labelling it does exactly one thing, so differencing the quadrant makes it cancel.
A quarter turn that destroys **>75%** of absolutely-labelled symbols costs a differentially encoded
link nothing but its first symbol.

```python
result = sweep(graph, {("laser", "power"): [-24.0, -21.0, -18.0]}, runs=8)
q = result.metric(analyzer, lambda m: m.q_factor)     # shape (points, runs)
```

Repeats matter more than they look. At −20 dBm, eight runs of the same link give error counts of
37 to 58 — a 50% spread on the thing being measured, while Q itself is stable to ±1%. A single
BER at a marginal operating point is one sample, not an answer.

### Amplified and nonlinear

`python examples/python/amplified_link.py` runs a chain of 80 km spans, each amplified back to transparency:

```
  spans   reach     OSNR      vs. one span            Sech pulse over 4 soliton periods
      1     80 km   36.95 dB    +0.00  (theory -0.00)   configuration      width   peak
      2    160 km   33.94 dB    -3.01  (theory -3.01)   soliton (N = 1)    x1.00   x1.00
      4    320 km   30.93 dB    -6.02  (theory -6.02)   half the power     x2.62   x0.40
      8    640 km   27.92 dB    -9.03  (theory -9.03)   no nonlinearity    x4.12   x0.23
     16   1280 km   24.91 dB   -12.04  (theory -12.04)  no dispersion      x1.00   x1.00
```

The left table tracks `10·log10(N)` to a hundredth of a dB over sixteen spans, and nothing in the
model is written in those terms — the noise bins just accumulate. The single-span figure of
36.95 dB is `58 − 16 − 5`: the quantum floor, the span loss the amplifier has to make up, and its
noise figure.

The right table is the soliton. At N = 1 the chirp the Kerr effect imposes cancels the one
dispersion imposes and the pulse is unchanged after 29 km; halve the power and the balance breaks.
The last row is the honest caveat — self-phase modulation *alone* also preserves `|A(T)|`, so
shape invariance proves nothing by itself. It is invariance with both effects active that is the
result.

### What ASE beat noise is for

An amplifier's OSNR is only half an answer. A photodiode squares the field, so ASE arriving with
the signal *beats* against it rather than merely adding its power — and on any amplified link that
beat term is the noise floor. Without it this project could compute OSNR to a hundredth of a dB
over sixteen spans and then report a Q that had almost nothing to do with it:

```
8 x 80 km, each amplified to transparency, 10 Gb/s OOK

   NF       OSNR    Q before    Q now   Q from OSNR
  4 dB   25.96 dB     118.73    22.79         25.08
  8 dB   21.96 dB     113.72    14.50         15.59
 14 dB   15.96 dB      94.83     6.83          7.51
```

The middle column is the old model. Ten decibels of OSNR cost it **nothing at all** — Q moved from
118.7 to 94.8 while the link's optical margin collapsed — because the only thing ASE contributed
was mean power and its shot noise. The right-hand column is the textbook relation
`Q = 2·√(B_ref/B_e)·OSNR/(1+√(1+4·OSNR))`, checked first against its own known point: 14.5 dB must
give Q ≈ 6, which is the industry figure for 10 Gb/s at 1e-9.

The model now sits **just below** that limit at every point, which is the right side to be on — it
carries shot, thermal and finite-extinction effects the closed form omits.

Coherent detection has the same term, with ASE beating against the local oscillator instead, and
there the check is `SNR = 2·OSNR·B_ref/R_s`. The gap closes as ASE takes over, which is what makes
it a test of the beat term rather than of one operating point:

| noise figure | OSNR | SNR | optical limit | gap |
| ---: | ---: | ---: | ---: | ---: |
| 4 dB | 22.08 dB | 18.82 dB | 21.01 dB | 2.18 dB |
| 10 dB | 16.08 dB | 14.32 dB | 15.01 dB | 0.69 dB |
| 16 dB | 10.08 dB | 8.78 dB | 9.01 dB | **0.23 dB** |

Only ASE co-polarized with the signal beats with it, which is why a polarizer helps a receiver and
why a coherent front end needs no optical filter at all: it is filtered by its own electrical
bandwidth, so the ASE-ASE term that forces a direct-detection receiver to carry one is absent by
construction.

### Wavelength selection, and what a filter is really for

`python examples/python/wdm_demux.py` puts four channels on a 100 GHz grid through four amplified spans
and demultiplexes one. Because every band carries its own centre frequency, that is a real
wavelength-selective operation and not a choice of array index — a filter tuned *between* two
channels attenuates both.

The second job is the one that surprises people. An amplifier emits ASE across four terahertz and
every hertz of it reaches the photodiode and beats there:

| link | OSNR | ASE power | Q | vs OSNR limit |
| :--- | ---: | ---: | ---: | ---: |
| no demultiplexer | 29.93 dB | 0.3250 mW | 7.43 | 0.19× |
| 50 GHz demux | 26.92 dB | 0.0040 mW | 25.04 | **0.89×** |

Eighty times less ASE reaches the diode, and the demultiplexed link lands just under its own OSNR
limit — which is where a real receiver sits. **The OSNR figure barely moves**: it is quoted in a
fixed 12.5 GHz reference bandwidth, so it cannot see ASE removed outside that band. The cheapest
improvement available to a receiver is invisible to the number everyone quotes.

The filter's skirts are floored at a declared `extinction`, because a super-Gaussian's are not. A
third-order 50 GHz passband is `exp(-2838)` one channel spacing away — not a small number but
exactly zero in double precision, which would make rejection infinite and a chain of filters
accumulate no crosstalk at all. Real hardware specifies 30–50 dB and it is that floor, not the
shape, that decides what leaks through a long line of them.

The **[OSA](../src/maiman/components/filters.py)** finally makes the signal model visible: bands and
noise bins rendered onto one grid, the way an instrument shows them. Its resolution bandwidth is
not cosmetic — widening it raises the ASE trace decibel for decibel and leaves a carrier exactly
where it is, which is the clearest demonstration of why OSNR needs a stated reference bandwidth.

**It sweeps full span by default, because an instrument you have to aim is no use for finding
something.** It used to start at 1550 nm with a 1000 GHz window and drop everything outside, so a
laser at 1560 nm produced an empty trace and the only way to see it was to already know the
wavelength and type it in here as well — which is backwards, since the wavelength is usually the
thing being measured. Measured, before and after:

```
laser at   fixed window (1550, 1000 GHz)   automatic
1550 nm    1.0000 mW, peak 1550.00         1.0000 mW, peak 1550.00
1560 nm    0.0000 mW  — nothing at all     1.0000 mW, peak 1560.00
1310 nm    0.0000 mW  — nothing at all     1.0000 mW, peak 1310.00
```

Automatic covers what the signal actually occupies — every band's sampled bandwidth and every noise
bin, plus 5 % margin — so two carriers forty nanometres apart are both in the trace rather than the
loudest one deciding. `auto_span` off restores the declared window exactly, which is what you want
once the channel is known: `points` is fixed either way, so span buys coverage and costs
resolution, and that trade is the reason both modes exist.

**A default changed here, and it changes numbers.** An OSA in an existing project now sweeps a
different window, so its trace is not the trace it was — the shipped WDM demo went from a 6.40 nm
slice to the whole 35.27 nm comb. Set `auto_span` to false to get the old one back.

## A circuit is not a chain

A fibre link is a chain: each block takes a waveform and returns one, and the scheduler runs them in
order. A photonic integrated circuit is not. Light in a ring goes round, comes back to the coupler it
entered by, and interferes with itself; a resonance *is* that feedback at steady state. There is no
order to run the blocks in, because every port's answer depends on every other port's at once.

So [`maiman/circuit.py`](../src/maiman/circuit.py) does a different kind of solve. Every device is a
**scattering matrix** — for unit amplitude into port *j*, `s[i, j]` is what leaves port *i* — and a
circuit is which ports are wired to which. Split the ports into external and internal, write the
wires as a permutation `C` that hands each internal port's outgoing wave to its partner, and

```
a_int = C S_ii a_int + C S_ie a_ext        S = S_ee + S_ei (I - C S_ii)^-1 C S_ie
b_ext = S_ee a_ext + S_ei a_int
```

That is the whole solver. It is exact and not iterative: there is no convergence criterion to pick
and no number of round trips to truncate at. A resonance appears as `(I - C S_ii)^-1` growing large,
which is the same statement as the round trip approaching unit gain, reached in one step instead of
summed. The test suite checks it against the summation anyway, because that is a genuinely different
algorithm: **they agree to 1e-13 over three free spectral ranges.**

**The roadmap said to integrate a solver rather than write one, and that was worth checking before
believing.** For the *solver* it does not hold — the identity above is **twelve lines of numpy**,
inside a `solve` that is forty-six lines once the grid check, the port bookkeeping and the dropping
of dangling ports are counted. (This used to say "thirty lines", which was neither of those
numbers and was nobody's measurement.) What is genuinely large in that ecosystem is the PDK side,
and that is now joined from the other direction — see [the netlist
section](#what-a-layout-tool-knows-and-what-it-does-not). The other path was measured rather than
argued about: installing SAX resolves to **37 packages**, including jax
and a 66 MB jaxlib, plus matplotlib, pandas, scipy, sympy, xarray and pydantic — to perform one
`numpy.linalg.solve`. And `klujax`, its sparse back-end, is **LGPL-2.0-only**; this project already
refuses FFTW over exactly that question.

Refusing the dependency is not refusing the reference. SAX, installed in an environment of its own,
is given the same two device models, the same wiring and the same grid —
`python examples/python/sax_crossvalidation.py`:

| add-drop ring, 4001 frequencies over 2 THz | max &#124;maiman − sax&#124; |
| :--- | ---: |
| in → through | 4.8e-15 |
| in → drop | 3.5e-15 |
| add → through | 3.5e-15 |
| add → drop | 2.5e-15 |

About twenty units in the last place of double precision, across a spectrum containing three
resonances; a coupling moved by one part in a billion shows up as 7e-10. The comparison is not in
the CI that runs on every push — nothing that costs 70 MB and a licence review should be — but it
is a workflow of its own, *SAX cross-validation*, dispatched before each release, and it is the
reason those twelve lines are defensible.

### What comes out of it

`python examples/python/microring_filter.py`. Three blocks
([`maiman/components/photonic.py`](../src/maiman/components/photonic.py)) and two device models
([`maiman/photonics.py`](../src/maiman/photonics.py)), and the ring is **assembled rather than written
down** — two couplers and two arcs, wired into a loop and handed to the solver. The closed forms from
Yariv and Bogaerts live in the tests, on the other side of the comparison, where they can disagree;
they agree to 1e-13.

A 100 µm silicon ring at 3 dB/cm, walked across one resonance:

| offset | critical, κ = 0.00688 | under-coupled, κ/4 | over-coupled, 4κ |
| ---: | ---: | ---: | ---: |
| ±20 GHz | −0.01 dB | −0.00 dB | −0.03 dB |
| ±1 GHz | −2.08 dB | −0.57 dB | −3.07 dB |
| 0 | **−134 dB** | −4.42 dB | −4.39 dB |

**Critical coupling is a knife edge and a false friend.** Match the coupling to the round-trip loss
and the field coupled back out of the ring cancels the field that stayed on the bus, exactly. Miss it
and the notch fills in — by the *same* amount on either side, because the depth
`|t − a|² / |1 − t a|²` is symmetric in *t* and *a*. Solving for the partner of a given coupling,
`t₂ = (2a − t₁(1 + a²)) / (1 + a² − 2 a t₁)`, produces pairs that are indistinguishable from the
through port alone: κ = 0.0199 and κ = 0.00237 both notch −6.22 dB.

**The free spectral range is set by the group index and the resonance position by the effective
one**, and in silicon those differ by a factor of 1.7. Measured against `c / (n_g L)` across four
resonances: 713.80 GHz against 713.79. Change `n_eff` and the resonances move without the spacing
changing; double `n_g` and the spacing halves.

### A ring is an ASE gate, and that needs its linewidth

An amplifier emits across terahertz; a ring's comb passes about one linewidth in every free spectral
range. Reading the response at the bin's centre would return whatever that one frequency landed on,
so the noise a ring passes is *integrated* — and how finely is not a free parameter.

A ring 116 MHz wide inside a 714 GHz period is a loaded Q of 1.7 million: good, and buildable.
Averaging its drop response over a 4 THz amplifier bin, against a converged 2.444e-4:

| how the mean was taken | result | |
| :--- | ---: | ---: |
| 4096 points across the whole bin | 4.81e-4 | 97 % high |
| 4096 points across one free spectral range | 3.13e-4 or 2.14e-4 | 28 % high, or 12 % low |
| points set from the linewidth | 2.444e-4 | converged |

The middle row is the one worth staring at. **Which way it is wrong depends on where the amplifier's
band happened to sit** — high when the window landed on the resonance, low when it landed on a real
EDFA's centre — which is what makes an under-resolved integral worse than a merely inaccurate one.
So `RingResonator` hands its own linewidth down to the integrator, and the ceiling that stops the
point count running away is stated in the source rather than discovered.

On an ordinary ring the numbers come out where they should: a 12.44 GHz linewidth in a 714 GHz
period passes **2.40 %** of a flat spectrum, against 2.74 % for a Lorentzian of the same width — the
14 % the shape approximation costs once integrated rather than merely evaluated at half depth.

**What that 2.40 % is, and what it is not.** It is the fraction of an amplifier's *total* ASE the
drop port passes, and it is right. For a long time it was also, silently, the density the ring left
at *every* frequency, because a noise bin was flat and could hold nothing but that average. So an
OSNR meter on resonance read the ASE beside the carrier as having dropped 16 dB, and a test asserted
the ring improved OSNR by more than twelve. It does not. This ring's 12.44 GHz linewidth *is* the
12.5 GHz reference band, so on resonance it passes most of the ASE inside that band along with the
signal:

| ASE beside the carrier, read as | kept | OSNR change |
| :--- | ---: | ---: |
| the flat average over one free spectral range (before) | 2.40 % | +15.6 dB |
| integrated over the reference band (now) | 68.8 % | **+1.06 dB** |

Noise bins now carry the shape of what they went through, normalised to the mean they already held,
so every total-power figure in this section is unchanged and a density read at one frequency is
finally the density at that frequency: about 91 % of the input at a tooth and a few hundredths of a
percent between two. A detector downstream of a ring or an interferometer now beats against the ASE
that is really beside its carrier rather than against an average spread over hundreds of gigahertz —
which, for a ring like this one, had been understating its signal-spontaneous beat noise by some
16 dB.

Straightened out, the same waveguide is a delay line, and the arithmetic is bleak: 1 mm of silicon
holds 14.01 ps and costs 0.2 dB; 10 cm holds 1.40 ns and costs 20 dB. Optical buffering is expensive
and this is why.

### Two modes in one waveguide

This section used to say that one response was applied to both polarizations, and that fixing it
needed "a polarization-resolved scattering matrix, which is a second index on every device". **The
second half was wrong, and it was wrong in a way worth recording**: it had never been measured. The
solver needed *no changes at all*.

`SMatrix` identifies ports by **name**. So a device carrying two modes is not a device with an
extra index — it is a device with twice as many ports, and `Circuit.solve` already solved those.
What was missing was one combinator:

```python
dual_polarization({"te": guide_te, "tm": guide_tm})   # -> ports in@te, out@te, in@tm, out@tm
```

which stacks one matrix per polarization block-diagonally, plus `link_polarizations` and
`expose_polarizations` so a wire is wired on both modes in one call — the fault they exist to
prevent being a circuit connected on TE and not on TM, which yields a perfectly plausible spectrum
on one polarization and silence on the other.

**A strip waveguide's two modes are two different waveguides.** For 500 × 220 nm silicon at
1550 nm:

```
              n_eff    n_group    ring FSR (100 µm)
TE             2.44       4.20         713.8 GHz
TM             1.78       3.80         788.9 GHz
```

27 % apart in phase index. So the ring resonates at **two sets of wavelengths that are nowhere near
each other, on two different free spectral ranges** — and both match `c/(n_g·L)` to a part in a
thousand, each against its own group index. Measured on a graph: at 1558.16 nm the `Ex` component
sits in a notch and `Ey` passes; at 1561.71 nm it is the other way round. A ring like this is a
polarization-selective filter, which is what a real one is.

`python examples/python/birefringent_ring.py` prints both tables — where each comb sits against
`c/(n_g·L)`, and then the same ring as a block with the light launched at 45° so both axes carry
something for it to treat differently. Using TE's group index for TM would put that row 75 GHz out,
a tenth of a free spectral range, and it would still look like a perfectly plausible ring.

**The flag is off by default**, declared as an idealisation exactly the way `saturate` is on the
EDFA. Turning `birefringent` on changes the numbers of any link with a photonic block in it; that
should be a decision someone makes, not a default that moves under an existing project. A test
asserts that off, the two field components take *exactly* the path they took before the flag
existed — `rel=1e-12`, not approximately.

**Block diagonal is a choice the models make, not a shape the matrix can hold.** A bend, a sidewall
that is not vertical, and a mode converter placed there on purpose all convert TE to TM. Nothing
here produces such a term, so `dual_polarization` takes a `cross` argument for one — because the
solver never needed the block-diagonal assumption, and the one place where quietly acquiring it
would be invisible is here.

**What is still shared: loss and dispersion.** A real strip has a different propagation loss for
TM — it overlaps the sidewalls less and the substrate more — and a different dispersion with it.
Those are per-process numbers and this library will not invent one, so the two indices are what is
offered and a test records the rest as shared rather than leaving it to be discovered. A fitted set
belongs in a PDK, where `maiman.pdk` already refuses to extrapolate past the window it was fitted
in. The same goes for the couplers: a directional coupler's split ratio is polarization-dependent
and here it is not, so the two combs come out with the same notch depth where a real pair would
not.

**Coupling into the die is its own step.** Every photonic block takes `Ex` to be the chip's TE mode
and `Ey` its TM. `EdgeCoupler` and `GratingCoupler` are what put the signal into that frame, with a
die rotation, a per-polarization response and a loss — see *Getting light onto a chip*.

The MMI, the Y-junction, the Mach-Zehnder and PDK import are all downstream of this framework
rather than of new physics — an interferometer is already two couplers and two arms in a `Circuit`,
and the tests build one.

## A channel plan, and the crosstalk it produces

The roadmap said "DWDM MUX/DEMUX with crosstalk" and Phase 3 shipped without one. That was not
quite a gap in the physics — `Combiner`'s docstring calls itself a WDM multiplexer and
`OpticalFilter`'s first line calls itself the demultiplexer, and both are telling the truth. What
was missing was the *grid*: building an eight-channel demux meant a splitter, eight filters, and
channel arithmetic each project did for itself.

### The two ITU grids are not the same kind of grid

**G.694.1 is uniform in frequency. G.694.2 is uniform in wavelength.** That is the whole difference,
and it has a consequence people trip over:

```
   frequency      wavelength        step
  192.9000THz    1554.1340nm
  193.0000THz    1553.3288nm     0.8053nm
  193.1000THz    1552.5244nm     0.8044nm
  193.2000THz    1551.7208nm     0.8036nm
  193.3000THz    1550.9180nm     0.8028nm
```

The dense grid is anchored at **193.1 THz**, which is 1552.5244 nm, and every channel at every
spacing is an exact multiple of the spacing from it — that is what makes a 50 GHz plan a *superset*
of a 100 GHz one rather than something offset by half a channel. But the wavelength steps are not
equal and cannot be, because `Δλ = λ²Δf/c`: 100 GHz is 0.7808 nm at 1530 and 0.8170 nm at 1565.
Stepping a flat 0.8 nm to build a "100 GHz grid" loses most of a channel across the C band.

The coarse grid does the opposite — eighteen channels, 1271 to 1611 nm, exactly 20 nm apart — which
comes out **56 % uneven in frequency**, 3.654 THz at the blue end against 2.339 at the red.

**And the 20 nm is a specification, not a round number.** A CWDM channel is meant for an *uncooled*
DFB, and an uncooled laser walks about 0.1 nm per kelvin: seventy kelvin of case temperature is
7 nm of drift before manufacturing spread. 20 nm spacing is what makes a transmitter with no
thermoelectric cooler, no wavelength locker and no control loop into a legal channel — which is why
CWDM optics cost a fraction of DWDM optics, and why the grid reaches out to 1611 nm where no
amplifier will help it.

`dwdm_frequencies` refuses a spacing the standard does not define and names the function that will
build one anyway. `cwdm_wavelengths` stops at eighteen: 1631 nm is not a CWDM channel, it is past
the end of the grid, and continuing the arithmetic would invent one.

### The blocks are routers, not splitters

`Multiplexer` and `Demultiplexer` sit on that grid. Eight channels out and back:

```
  port      channel       kept    worst neighbour
     0   1550.0000nm    -8.000dB            -40.00dB
     1   1549.1990nm    -8.000dB            -40.00dB
     …
     7   1544.4105nm    -8.000dB            -40.00dB
```

−8 dB is 4 dB through the mux and 4 through the demux, and **it does not grow with the channel
count** — at 2, 4, 8, 16 and 32 channels it is −8.000 dB every time. An arrayed-waveguide grating
*routes*: each channel leaves by its own port. A broadcast-and-select demux really would divide
power N ways and cost 10·log10(N) — 15 dB at thirty-two channels — and that device is a `Splitter`
in front of filters, which is exactly what `Splitter` is for.

Every demultiplexer port is **the same function** the single-filter block is. `apply_passband` was
lifted out of `OpticalFilter` for that reason and a test asserts the two agree sample for sample: a
multi-port device whose per-channel response drifted from the single-filter block's would be the
worst kind of disagreement, since both would look right alone and the crosstalk number would depend
on which one happened to be used.

### A ROADM node: the add/drop switch

`WavelengthSelectiveSwitch` is one degree of a ROADM on the same grid. Every channel of the line
but one leaves by `out` through its own passband, the `dropped` one leaves by `drop`, and what
arrives on `add` takes its slot. Its `isolation` is how far below its arrival the dropped channel
still reaches `out`, and there it lands on the added channel's own frequency. The two are added
as fields, so the leak beats with the new channel at the receiver: in-band crosstalk, which no
filter downstream can remove. In example 3.9, 35 dB of isolation takes the added channel from
Q 41 to 26, and 20 dB takes it to 6.

### Which of two things sets the crosstalk

```
   spacing   passband    with floor    skirt alone
      200G        50G       -40.00dB          -infdB
      100G        50G       -40.00dB          -infdB
       50G        50G       -40.00dB       -192.66dB
       40G        50G       -40.00dB        -50.50dB
       25G        50G        -3.01dB         -3.01dB
```

Nothing in that table was declared. The right column is the super-Gaussian's own skirt, falling as
the sixth power of detuning at order 3 — and `-inf` there is not infinite rejection, it is the field
underflowing `complex64` past about −600 dB, which is a storage limit and not a physical one.

The left column is what a real system sees, and the point of it is that **the skirt is almost never
what matters**: at any sane plan it is already below the extinction floor, and it is the floor that
accumulates down a chain of these. At 25 GHz with a 50 GHz passband the neighbour is simply inside
the channel, and no floor is involved at all — which is a channel plan that does not work rather
than a model that does not.

`python examples/python/wdm_grid.py` prints all five tables.

## A mirror, and a graph that goes one way

Every optical block up to here is matched at both ends: light enters one port and leaves another,
and a dataflow edge is an arrow that says so. A fibre Bragg grating is the first one that is not.
Its useful output comes back out of the fibre it arrived on, and on a bench the only way to that
output is a circulator.

Two things had to be true for that to work, and one of them was not.

**The solver was already fine.** `circuit.py` has said since it was written that "a non-reciprocal
device (an isolator) and a reflecting one (a facet, a Bragg grating) both solve correctly" — the
reduction never transposes anything and never drops a term. Until now nothing outside a test made
it prove it. `FiberBraggGrating` puts its reflection on the diagonal of a 2×2 matrix and
`Circulator` is a cyclic permutation that is deliberately not symmetric, and both go through
unchanged.

**The scheduler was not.** The canonical drop is

```
signal → circ.in1 ; circ.out2 → fbg.in ; fbg.reflected → circ.in2 ; circ.out3 → receiver
```

which reads as `circ → fbg → circ` and was refused as a feedback loop. **It is not one.** At the
level of ports, `out3` is a function of `in2` alone and `out2` of `in1` alone; no light in that
picture travels backwards in time. The cycle was an artefact of scheduling whole components.

So a component may now declare that its ports fall into independent `PortGroup`s, and the
scheduler makes a node of each. The default is one group holding everything, which is the truthful
answer for every block that computes all its outputs from all its inputs in one call — that is,
every other block in the library, and the sort is bit-for-bit what it was. A real loop is still a
real loop: two attenuators wired to each other are refused, and so is a circulator whose three hops
are wired into a ring, because splitting a component into independent nodes cannot launder a cycle
that runs through all of them.

The cost of getting that claim wrong is a port that never runs or one that runs twice, so the
groups are checked to be a partition before anything executes rather than discovered mid-run.

### The grating is assembled, not written down

Two hundred per-section transfer matrices, multiplied. Erdogan's closed form for a uniform grating
appears nowhere in the model — it is in `tests/test_grating.py`, on the other side of the
comparison, and they agree to 2×10⁻¹¹ across two decades of grating strength. That is what buys
chirp and apodization, neither of which has a closed form:

```
        δn    κL    tanh²(κL)      model     max error
     1e-05  0.203    0.039981   0.039981      2.3e-13
     5e-05  1.013    0.588552   0.588552      3.4e-12
     1e-04  2.027    0.932915   0.932915      7.9e-12
     5e-04 10.134    1.000000   1.000000      2.3e-11
```

The first-null bandwidth lands on `λ²/(π n_eff L)·√((κL)²+π²)` to a per-mille, which is the
resolution of the measurement rather than of the model, and `R + T = 1` everywhere to 10⁻¹⁰ — a
transfer-matrix product that drifts off unity is the classic way this method fails, and it fails
quietly.

### Apodization is not mainly about sidelobes

The textbook reason to shape a grating's strength is crosstalk. The reason that decides whether a
*compensator* is usable is group-delay ripple, and it is an order of magnitude bigger:

```
       profile     peak R   sidelobe   GD ripple
       uniform     0.9329     -7.6 dB     101 ps
 raised-cosine     0.5886    -31.4 dB     6.5 ps
      gaussian     0.5823    -41.4 dB     8.1 ps
```

101 ps is a whole bit at 10 Gb/s. The peak comes down in the same move, because shaping the
coupling lowers its average — there is no free version of this.

### A compensator that needs no receiver

`DispersionCompensator` is DSP. It inverts the fibre's all-pass filter, which means it needs the
field, which means it needs coherent detection; a direct-detection receiver squares the field at
the photodiode and can never do it at all. A chirped grating is glass, and it works in front of a
photodiode.

10 cm chirped across 0.71 nm is −1360 ps/nm — 80 km of standard fibre — in a part the length of a
finger. A 30 ps Gaussian pulse, rms width, no amplifier:

```
     span       bare    + grating
      0 km    21.21 ps    44.81 ps
     40 km    29.46 ps    28.52 ps
     80 km    46.06 ps    21.34 ps
    120 km    64.89 ps    30.54 ps
```

Back to the launched width at the span it is matched to, over-compensated below it and
under-compensated above — which is what a fixed compensator does.

### The circulator charges twice, and says so

Its loss is quoted per hop because that is how a datasheet quotes it, and light reaching a
reflective device pays it going out and coming back. Dropping 1550 nm out of a pair 2 nm apart:

```
                1550 nm     1552 nm    accounting
       drop     -1.702 dB  -47.359 dB  two hops (-1.4) + R (-0.30)
    express    -12.434 dB   -0.700 dB  one hop (-0.7) + T (-11.73)
```

Every figure is arithmetic. With an ideal circulator the neighbour's 45 dB of rejection on the drop
port is **the grating's own sidelobe floor two nanometres out**, not an isolation figure anybody
declared — apodize the grating and the number moves. With a real circulator it is not the grating at
all; see below.

**A real circulator leaks, and this section used to get that wrong.** It said isolation could not be
a parameter because in this configuration the backward leak is a *loop* — light returning to the
grating, reflecting again, coming round once more — and a cavity the engine was right to refuse. It
is not a loop. Reflected light coming back into port 2 leaves at port 3, or back out of port 1
toward the source, and neither returns to the grating. `Circuit.solve`, which sums every bounce there
is, gives a drop port equal to the feed-forward `τ·r·τ + ι` with a difference of exactly zero.

What the leak really broke was `PortGroup`. Port 3 now hears port 2 *and* the direct leak from port
1, while port 2 hears port 1 alone: dependencies that overlap and contain no cycle, which a partition
of ports could not express. Outputs are still a partition; an input may now be read by more than one
group, and `Circulator` has an `isolation` parameter. The engine matches the exact solve to a few
micro-decibels at every setting:

```
    isolation     1550 nm      1552 nm    neighbour rejection
        ideal    -1.702 dB   -47.359 dB        45.7 dB
        60 dB    -1.702 dB   -46.781 dB        45.1 dB
        40 dB    -1.701 dB   -38.713 dB        37.0 dB
```

At 40 dB the floor on the drop port is the circulator's leak from port 1, not the grating — which is
the number a real drop would be specified by. **Return loss is what would be a cavity**: a reflection
back out of the port light entered by, bouncing into the grating again. At 40 dB of it the exact solve
leaves feed-forward by 8e-3 and ripples the drop port by about ±0.07 dB. That one is a real loop, and it has
its own section.

### A loop the engine will run

The scheduler orders a graph, and a cycle cannot be ordered — which for almost everything here is the
truth: a link goes one way, and a cycle in its graph is a wiring mistake. Two things that looked like
cycles turned out not to be, and `PortGroup` handles both without any loop control. Return loss is the
one that is. Light reflected back out of port 2 goes into the grating again, which reflects it again,
and the steady state is the geometric series `τ·r·τ / (1 − ρ·r) + ι` that nothing feed-forward can
sum.

`Feedback` is the explicit, declared way to run it. Put it on any one wire of the loop: its `out` feeds
the loop and its `in` takes what comes back, so the scheduler sees a source and a sink and the graph
has an order again. The whole graph then runs `passes` times. On the first pass `out` is dark; on every
later one it is whatever reached `in` the pass before. Without it, the same wiring is refused with a
`CycleError` that now names the fix — and a graph with no `Feedback` in it runs exactly once, as it
always did.

Against `Circuit.solve`, which sums every bounce exactly — 20 dB of return loss, 40 dB of isolation:

```
   passes   drop, 1550 nm    residual
        1      -40.000 dB         inf   nothing has been round yet: the leak alone
        2       -1.701 dB     9.6e-02   one trip to the grating: the feed-forward sum
        3       -1.671 dB     9.4e-03
        4       -1.752 dB     9.0e-04
        5       -1.752 dB     8.7e-05
        6       -1.751 dB     8.4e-06   the exact cavity, to 4e-7 dB
```

**The residual falls by 0.0966 every pass, and that number is the physics:** the echo's amplitude 0.1
times the grating's reflection 0.966, the loop's round-trip gain. The power does not fall so tidily —
three passes are further off than two, and five than four — because partial sums of a series whose
ratio carries a phase overshoot and undershoot the limit in turn. Watch the residual, not the power.

**What a pass is not.** It is not a lap in time. Every signal is a whole window in its own retarded
frame, so a trip round the loop adds no delay and every lap lands on top of the last. That is right for
a cavity short against the window — a reflection between parts centimetres apart — and wrong for a
recirculating loop a pulse goes round many times, where the laps should arrive one after another. That
needs a delay line, and this is not one — `DelayLine` is.

### A recirculating loop

`DelayLine` moves the field later by a fixed time, carrier phase included: the envelope of a band at
`f0` becomes `A(t − τ)·exp(−2πi f0 τ)`, so two paths of different length interfere the way they do in
glass. A delay that is a whole number of samples is a shift and exact; any other is a linear phase in
frequency. It has no loss and no dispersion of its own, and noise passes unchanged.

Put one in a `Feedback` loop and every pass becomes a lap that arrives one loop-time after the last.
A 10 ps pulse into a 3 dB coupler whose second pair of ports is joined through 800 ps and 1 dB:

```
   lap   arrives    peak power    accounting
     0      0 ps     0.500 mW     the straight half
     1    800 ps     0.199 mW     both crossings, one pass of loss
     2   1600 ps     0.079 mW     and another straight half and pass of loss
     3   2400 ps     0.031 mW     and another
```

Asserted to a part in ten thousand, and with the delay set to zero the same graph puts nothing at
800 ps at all — every lap lands on the first, which is the difference this block exists to make.

The window is periodic, because sources draw whole periods of their pattern: a lap that leaves the
end of the window comes back in at the start. Choose a window longer than the laps you want to watch.

### The named windows were never the limit

The section loop has always eaten two arrays — a coupling per section and a local Bragg wavelength
per section — and the three named windows and the linear chirp were only ever two ways of filling
them. Handing the arrays over directly costs the loop nothing and turns a grating *model* into
something closer to a grating *design* model. A named window and its array are checked to be the
same device bit for bit, not approximately, so the convenient path and the general one cannot
become two models with one of them tested.

**A sampled grating is one array.** Write the grating and erase it periodically — in practice,
expose it through an amplitude mask — and the single peak becomes a comb. It is the tuning element
of a sampled-grating DBR laser, a multi-channel dispersion compensator, an interrogator that reads
a whole array of sensors at once. Ten sampling periods over 20 mm:

```
  7 peaks, spacing                       0.41478 nm
  predicted λ²/(2·n_eff·Λs)              0.41494 nm
```

Four digits, against the same Fabry–Perot arithmetic as any other cavity of that length — because
the sampling period *is* the cavity. The model has nothing in it that knows about combs. It has a
coupling that is switched on and off.

**A phase-shifted grating is one more.** Break the periodicity once and the stop band acquires a
transmission window in the middle of it: two mirrors facing each other, which is a cavity. At π,
dead centre, on a 2 cm grating:

```
  window at            1550.000000 nm
  width                0.70 pm  =  87 MHz
  unbroken, the same grating transmits 1.8e-4 there
```

87 MHz is the narrowest feature anything in this library produces. It is the distributed-feedback
laser's cavity and the filter a laser locks itself to. Moving the break off centre makes the two
halves unequal mirrors and the resonance dies — 1.000, 0.705, 0.099 at positions 0.5, 0.45 and
0.35 — which is a real design sensitivity rather than a modelling artefact.

The phase rides on the **coupling** and not on the detuning, and that is not a detail. It means the
two off-diagonal terms take it with opposite signs, so a section stays unimodular and the device
stays lossless; and it means a phase constant along the whole length does nothing at all, which is
the physically right answer, since where a grating's fringes start is not observable. A phase on
the detuning instead would shift the entire spectrum and look perfectly plausible doing it.

**Two descriptions of one thing are refused rather than resolved.** `coupling_profile` alongside a
named `apodization`, or `bragg_profile` alongside `chirp`, is an error that names the array the
other one stands for. Profiles of different lengths are an error. A component is the exception and
for a stated reason: a project file and an inspector carry numbers and not arrays, so
`FiberBraggGrating` declares `sampled` and `phase_shifted` as flags that *build* the arrays — and
when a grating is both sampled and apodized it multiplies them, because a mask over an apodized
exposure is what the writing process actually does.

Sampling also raises the section count on its own. What has to be resolved is the mask rather than
the envelope, so the floor is twenty sections per sampling period — measured: the comb spacing is
unchanged from ten sections per period all the way to four hundred. Deriving that rather than
refusing a number it could compute itself is the difference between a component and a form.

### The same device is a strain gauge and a thermometer

A grating's period is a length. Lengths respond to being stretched and to being warmed, and the
reflection reports it — so nothing has to be added to the fibre, there is nothing electrical near
the measurement, and because the reading is a *wavelength* it does not drift when the light gets
dimmer. That is why these go into boreholes, composite spars and bridge decks.

The whole of it is one line, and both coefficients are a material number times λ_B:

```
Δλ/λ = (1 − p_e)·ε + (α + ξ)·ΔT
```

```
  strain        1.209 pm per microstrain     p_e = 0.22
  temperature  11.191 pm per kelvin          α + ξ = 7.22e-6 /K
```

Stretching lengthens the period *and* lowers the index, and the two fight: the index term cancels
22 % of the geometric one. Warming does two things too, and they are not the same size — thermal
expansion is 0.55e-6/K and the thermo-optic coefficient 6.67e-6/K, so this is a thermometer made of
glass rather than one made of geometry. It is also why **coating changes the thermal number a lot**
and the strain number not at all: a jacket adds its expansion to α and nothing to ξ.

**One grating produces one wavelength and there are two unknowns behind it.** Divide one
sensitivity by the other:

```
  9.26 microstrain per kelvin

  +10 K, no load        -> 1550.111910 nm
  92.6 ustrain, no heat -> 1550.111953 nm
  difference             43.4 fm
```

Nothing about the spectrum hints that an interrogator should try to tell those apart. Every real
installation is arranged around this.

**And a second grating at another wavelength does not fix it**, which is the trap worth seeing
measured. Both sensitivities scale with λ, so their ratio does not:

```
   lambda_B   pm/ustrain      pm/K     ratio
   1530.0nm       1.1934   11.0466     9.256
   1550.0nm       1.2090   11.1910     9.256
   1570.0nm       1.2246   11.3354     9.256

   condition number of the 1530/1570 pair: 4.6e+16
```

Two rows differing only by a scale factor invert into noise. What does work is a grating that
carries no load — loose in its tube beside the working one, measuring the temperature alone — which
is a genuinely independent second equation rather than a nearly dependent one.

### Interrogation is a sweep, and it recovers what was applied

The array is one fibre: the probe goes out through a circulator and the gratings sit in series,
each passing what it does not reflect. Stepping a tunable laser across each peak and taking the
centroid of the reflected power is what a real interrogator does, and 121 points over 0.6 nm — a
5 pm grid — recovers a shift to a fraction of a picometre:

```
     grating      nominal      measured       shift
     working   1545.0000nm   1545.64936nm     649.36pm
   reference   1555.0000nm   1555.16841nm     168.41pm

   recovered temperature    15.000 K    (error +0.0000)
   recovered strain        400.000 ue   (error +0.0001)
```

And the same array read as though the temperature were known to be zero, which is what an
uncompensated gauge assumes:

```
              applied   strain read       error
   400 ue and + 0.0 K      400.00ue      +0.00ue
   400 ue and + 1.0 K      409.26ue      +9.26ue
   400 ue and + 5.0 K      446.28ue     +46.28ue
   400 ue and +15.0 K      538.85ue    +138.85ue
```

9.26 microstrain per kelvin, arriving exactly on schedule.

**Why a sweep, and why an analyser now works too.** A tunable laser is the more precise
interrogator, and every sweep point is a real run of a real graph. The other standard instrument — a
white-light source and an OSA — could not be built here when this section was first written:
broadband light was a `NoiseBin` whose density was *flat* by construction, a response reached it as
one averaged number, and the reflection of ASE off a grating came back with no peak in it at all.

Noise bins now carry the shape of what they pass through (see the ring section above), so it can.
The reflection of flat amplifier ASE off a 10 mm grating draws its peak on an analyser at the Bragg
wavelength and at exactly the input density times `tanh²(κL)`; a nanometre away it is 25 dB down;
reflected and transmitted power add back to the input to nine digits; and a thousand microstrain
moves the drawn peak by 1.209 nm, the same shift the sweep reads.

One more thing the array turned up, and the engine was right about it. Wiring every grating's
reflection back onto one return fibre is what the hardware does, and `Combiner` **refuses** it: a
single-wavelength probe means every grating is reflecting a copy of the *same* band, and co-located
carriers have to be added as fields on a common grid rather than multiplexed. So each grating is
metered where it sits, which is the quantity being measured anyway.

`python examples/python/fbg_sensor.py` prints all five tables.

### And a sign that was wrong, in the fibre

The grating's own physics says a positive chirp puts the short wavelengths at the near end, so they
turn round first and the long ones arrive later: `dτ/dλ > 0`, which is `D > 0`, the sign standard
fibre has. A compensator is therefore the **negative** one, which is what the table above uses.

It did not use to be. For a while this device and `Fiber` disagreed about the sign, because
`kernels.propagate_dispersion` carried the opposite quadratic sign from
`photonics.propagation_constant` — something the latter's docstring had said in as many words since
long before the grating existed, with nothing ever putting the two in one graph to argue about it.

**The grating was that thing, and the kernel was the one that was wrong.** Its β₂ term was
Agrawal's `exp(+iβ₂ω²z/2)`, correct under the `exp(−iωt)` convention that book uses and wrong under
the `exp(+iωt)` that `numpy.fft.ifft` gives this one. A β₂-only model cannot feel the difference —
ω appears squared, so every width came out right — which is why nothing caught it for so long. The
β₃ term was never affected, because it had been *derived* from the expansion of β(ω) rather than
transcribed, and the docstring says so.

What could feel it was anything built on the other convention. Inside `propagate_coupled_ssfm` the
walk-off term and the β₃ term already followed `exp(−iβz)` and the β₂ term beside them did not, so
a WDM simulation slid its channels one way and dispersed each of them the other. Measured directly,
over 20 km at D = +17 ps/nm/km:

```
  a component 200 GHz above the carrier arrived   1089.89 ps LATE
  D · Δλ · L says it should arrive               1089.89 ps EARLY
  walkoff_from_dispersion agrees with physics
```

Correcting it moved three more signs that had been matched to the old one, and all three are
conventions rather than physics: the Kerr rotation in `propagate_coupled_ssfm` — which has to flip
with β₂ or the soliton stops balancing — `GaussianPulse`'s chirp parameter, so that `C > 0` still
means an up-chirp, and the trial phase the blind dispersion search builds in
`dsp.clock_tone_strength`. The flagship coherent link's EVM is unchanged at 7.39 %, which is the
right outcome: dispersion and its compensation both flipped, so nothing about a link's performance
moved. What moved is which way things point.

The test that caught it had been written as a *pin* rather than a claim, precisely so that
reconciling the kernel would fail and name the compensator as one of the things that moved with it.
It did exactly that.

`python examples/python/fbg_circulator.py` prints all seven tables.

## Cladding modes, and a grating that couples to them

A Bragg grating needs one number from its fibre, an effective index. A long-period grating needs a
mode solver: its period is hundreds of microns, so it couples the core mode *forwards* into modes of
the cladding — light guided by the glass-air boundary 62.5 µm out — at the wavelengths where
`(n_core − n_cladding,m) · period` is one wavelength. `maiman.modes` finds those modes, and
`LongPeriodGrating` is built on them.

### A mode solver with no SciPy in it

Linearly polarised modes of a three-layer step-index fibre: Bessel functions in the core, the
cladding and the surroundings, matched at both boundaries. The Bessel functions are computed from
their integral representations, because the package depends on NumPy alone. Nothing in it is taken
on trust:

```
   Bessel J, Y, K against tables                     within 4e-12
   LP11 appears at the zero of J0, V = 2.4048        within 0.1 %
   core mode against Gloge's approximation           6e-4, inside Gloge's own 1e-3
   8 cladding modes against finite differences       1e-10 in effective index
```

The finite-difference reference is built in the test from a tridiagonal matrix and shares no code
with the solver. That pins the numerics of the scalar equation. Whether the scalar equation is the
right one is a separate question, and `maiman.vector_modes` answers it.

### And the modes the scalar equation cannot tell apart

The LP approximation assumes every index step is small. The core's step is. The cladding-air step is
a third of an index, and there each LP mode stands in for a family of true modes that Maxwell's
equations split. `maiman.vector_modes` solves for those: concentric layers, Bessel functions for
`Ez` and `Hz` in each, all four tangential components matched at every interface, and a mode where
the solutions regular on the axis meet the ones that decay outside. For `nu = 0` that splits into TE
and TM; above it the modes are hybrid, HE and EH, and `LP_lm` is `HE_{l+1,m}` together with
`EH_{l-1,m}`.

```
   glass rod in air, exact equation of Snyder and Love   residual 1e-10
   core HE11, same equation                              residual 1e-12
   three layers with no core step = two layers           1e-15
   weak guidance against the scalar solver               2.5e-7 in n_eff, 0.1 % in coupling
   power flow between distinct modes                     orthogonal to 1e-10
```

Snyder and Love's characteristic equation is written out in the test and shares nothing with the
solver but the Bessel functions. **What the split is worth**, on standard fibre in air: the HE1m sit
1.3e-7 below their LP0m at the top of the band, 1e-5 ten modes down and 1.9e-4 thirty modes down —
and the core's own HE11 sits 5.2e-6 below LP01, which is the polarization correction the scalar
equation has no way to see.

Set `vector` on a `LongPeriodGrating` and it couples HE11 to the cladding's order-one modes instead:

```
   LP04 notch, scalar modes    1584.1 nm
   HE14 notch, vector modes    1583.0 nm     -1.10 nm, and as deep
```

A nanometre is not a rounding error on a device whose whole output is where its notch went. The
EH1m, which the scalar model has no mode for at this order, couple at about one percent of the HE1m
and cut their own shallow notches between them.

### And the glass disperses

By default a fibre's three indices are constants, and the only dispersion the solvers see is the
waveguide's. Set `material_dispersion` on a `StepIndexFibre`, a `LongPeriodGrating` or a
`TiltedFiberBraggGrating`, and the glass disperses:

- the cladding follows Malitson's Sellmeier equation for fused silica;
- the core follows germania-doped silica. Its Sellmeier coefficients are interpolated in mole
  fraction between silica and Fleming's pure germania, at the fraction that gives the quoted index
  step: 3.5 mol % for 0.0052.

The quoted indices hold at 1550 nm and nowhere else. The default 1.4440 is Malitson's silica
there to 2e-5, so at 1550 nm nothing moves at all.

The check is that the default fibre becomes the fibre it was meant to be:

```
                          constant indices     dispersing glass     G.652
   D at 1550 nm           −4.8 ps/(nm·km)      16.7 ps/(nm·km)      ≤ 18
   zero dispersion        none                 1308.3 nm            1300 – 1324 nm
```

Nothing was fitted to get there. Silica's own zero of dispersion sits at 1272.7 nm, and the core's
waveguide dispersion drags the fibre's zero thirty-five nanometres longward. At 1550 nm the
material and waveguide terms add to the total to within half a picosecond per nanometre per
kilometre.

A long-period grating's notches are set by the difference of two indices that disperse
differently, so they move by nanometres:

```
   cladding mode   constant indices   dispersing glass
   LP01            1354.9 nm          1350.7 nm        −4.2 nm
   LP02            1388.7 nm          1384.9 nm        −3.8 nm
   LP03            1455.7 nm          1453.0 nm        −2.7 nm
   LP04            1584.1 nm          1585.5 nm        +1.4 nm
```

Each notch still sits where the dispersive fibre's own indices phase match, to 1e-9. A tilted
grating's comb is read against indices that hold at 1550 nm, where the comb is, so it moves by
picometres: 16 pm on its Bragg line, and 12 pm twenty nanometres below it. The surrounding medium
stays constant.

### Where the notches land

A 500 µm period over 25 mm, with an index modulation of 3e-4, on standard fibre in air:

```
   cladding mode   notch       depth, one mode alone = cos²(κL)
   LP01            1354.9 nm   0.826
   LP02            1388.7 nm   0.479
   LP03            1455.7 nm   0.163
   LP04            1584.1 nm   0.011
```

The coupled equations have constant coefficients for a uniform grating, so they are solved exactly —
every mode at once, by diagonalising a small matrix at each frequency. With one cladding mode coupled
the transmission is the textbook closed form to 1e-10. With all of them coupled the neighbours move a
notch's sides by a few percent, which is why the model couples all of them: with all eight, the
block cuts a carrier sitting on the LP04 notch by 19.2 dB.

### A refractometer with nothing electronic in the fibre

The cladding modes feel what surrounds the fibre, and the core mode does not. Dip the grating in a
liquid and every notch moves shortward, faster the closer the liquid comes to the glass:

```
   surrounding index   LP04 notch   shift
   1.000 (air)         1584.1 nm
   1.333 (water)       1581.5 nm     -2.6 nm
   1.430               1572.0 nm    -12.1 nm
```

LP01 moves a tenth as far: the low-order cladding modes barely reach the boundary.

### Two of them, and the cladding light comes back

A single grating's cladding light is lost under the coating. Strip the coating between two gratings
and it is not lost. Set `pair` and the grating is written twice, `separation` apart over bare fibre:

1. The first grating splits the core's light between the core and the cladding.
2. The two run at different speeds through the gap.
3. The second grating recombines them.

That is a Mach-Zehnder interferometer inside one fibre, and the notch one grating would cut fills
with fringes. Each grating is solved as the full matrix exponential, cladding amplitudes and all,
and the gap is diagonal, so the pair is exact. `gap_loss` is what the cladding modes lose on the
way. Four checks hold it:

- **No gap** is one grating twice as long, to 1e-12.
- **At a 3 dB split** the fringes swing between `(1 + g)²/4` and `(1 − g)²/4`, where `g` is the
  cladding amplitude that survives the gap. So a lossless gap gives a perfect null; the minimum
  matches to 1e-4, at 0, 3 and 10 dB of loss.
- **The fringe spacing** is `λ²/(Δn_g d)`, where `Δn_g` is the two modes' group-index difference,
  taken from the mode solver. It is within 5 % at a 20 cm gap and 2 % at 80 cm. The gap is the
  interferometer, and the gratings' own length matters less as the gap grows.
- **At a full crossover** each grating on its own empties the core, and the pair passes all of it
  back.

## A grating that reads a liquid, because it is tilted

A Bragg grating reflects the core mode into itself and is blind to everything outside the glass.
Tilt its fringes by a few degrees and the fringe front now varies *across* the core, which lets the
core mode reach every backward cladding mode whose azimuthal shape matches the tilt — a comb of
narrow notches below the Bragg line, each at `(n_core + n_cladding,m) · period / cos θ`.
`TiltedFiberBraggGrating` is that device.

Expanding the tilted fringe across the core in azimuthal orders leaves each order a Bessel function
of `2π sin θ · r / period`, so the overlap is closed-form and the tilt is a divider:

```
   tilt    into itself   into LP15        dn = 1e-4
   0°        152.6 /m        0.0 /m
   2°        114.9 /m       28.4 /m
   4°         41.1 /m       28.4 /m
   6°          3.5 /m        8.5 /m
```

Square on the whole coupling is the mirror and nothing else exists, which is the Bragg grating the
library already had — the same number to 1e-12. The overlap is checked against the two-dimensional
integral of the fringe pattern itself, on a grid, with no expansion in it: 1e-5.

The coupled equations are contra-directional and every mode is coupled only through the core, so the
32 modes nearest phase matching are solved together by matrix exponential and the rest enter as the
shift they leave behind on the core. One mode alone is the textbook `1/cosh(κL)`; transmitted plus
reflected plus what went into the cladding is 1 to 1e-10, exactly, because the equations conserve it.

**Why anyone tilts a grating**: the comb feels what the fibre is dipped in and the Bragg line cannot,
while temperature moves both together. A 10 mm grating tilted 4°, moved from air into water:

```
   notch                    in air        moves by
   Bragg line               1551.243 nm      0.0 pm
   LP0,1, just below it     1550.020 nm      0.2 pm
   LP1,19, 12 nm down       1539.782 nm     71.2 pm
   LP2,19                   1539.273 nm     75.0 pm
```

Read at the bottom of its comb against the Bragg line at the top, it is a refractometer carrying its
own temperature reference. `python examples/python/tilted_grating_refractometer.py` prints all four tables.

### And each polarization sees its own comb

The modes above are scalar, and in the scalar picture the two polarizations are one. The cladding's
true modes are HE, EH, TE and TM. An LP mode is a family of them whose effective indices the
glass-air boundary splits, and a real tilted grating's comb changes as the input polarization
turns. Set `vector` on a `TiltedFiberBraggGrating` and it solves the vector modes and couples each
polarization on its own. `x` is in the plane of the tilt (p, the studio's TM), and `y` is across
it (s, TE).

With the fringes tilted in `x–z`, each vector mode of order `ν` comes in two orientations of one
effective index. p reaches one orientation and s the other, and the cross terms are odd in `φ`, so
they vanish. The overlap is:

```
π ∫_core [ J_{ν−1}(K_t r) S  ∓  J_{ν+1}(K_t r) D ] r dr
S = e_r e_r' + e_φ e_φ'     D = e_r e_r' − e_φ e_φ'     minus for p, plus for s
```

That one sign is the whole of the polarization dependence:

- For `ν = 0` it leaves p reaching TM0m and never TE0m, and s the reverse, exactly.
- For the other orders it weights HE and EH differently.

Four checks hold it:

- **Untilted**, it is the long-period grating's vector coupling to 1e-12.
- **The core's own reflection** matches the scalar core's to 0.2 % for either polarization.
- **In weak guidance** (the fibre in a liquid of index 1.43), each LP family's vector modes together
  carry what the scalar LP mode does, to 2 %.
- **LP3**, reached a thousandth as strongly as LP1 at 4°, converges on its LP value as the step at
  the cladding closes, rather than matching it outright. The families it is built from are reached
  through `J₁` as well as `J₃`, and those `J₁` terms cancel only in the limit.

Near the cladding's cutoff the split is largest. There TE and TM part by tens of picometres, and a
30 mm grating's notches are narrow enough to separate them. Five nanometres below the Bragg line,
the two polarizations' transmission differs by up to 14 dB. The block sends the field's `x` through
p's comb and `y` through s's.

The vector modes cost about ten times the scalar ones: a minute rather than seconds for a
spectrum. The two polarizations share one mode solve, so the second costs a fraction of a second.

## A laser that chirps because it is being modulated

Every other transmitter here leaves the laser alone. A directly modulated laser is the current itself
— one chip instead of two, a fraction of the power — and what it gives up is in its output: the
carrier density has to move for the power to move, the index moves with the carriers, and the optical
frequency moves with the index. `DirectlyModulatedLaser` integrates the single-mode rate equations,
so the chirp is a consequence rather than a parameter.

```
   threshold current        23.57 mA      q V N_th / tau_n
   slope efficiency          0.160 W/A    eta h nu / q, measured to 1 %
   relaxation frequency      8.54 GHz     sqrt(g0 S0 / tau_p) / 2 pi, at 90 mA
   adiabatic chirp, ones   +12.96 GHz     kappa P, to 0.5 %
   transient peak          +20.58 GHz
```

The chirp is **measured from the field's own phase** and compared against
`alpha/4pi · d(lnP)/dt + kappa·P`, which correlates at 0.99997 across a step — neither term appears
anywhere in the integration. The ringing is checked against a spectrum of the step response, and the
confinement factor is *not* in that formula: it cancels between the two equations, and leaving one in
underestimates a 1310 nm laser's ringing by nearly a factor of two, which the measured step caught.

What it costs, at 10 Gb/s and 1550 nm against a Mach-Zehnder carrying the same power and extinction:

```
   span      DML Q   external Q
     0 km    15.57       283.09
     5 km    31.70       192.85
    10 km     8.85        98.12
    20 km     2.73        39.68
    40 km     1.35        29.86
```

**The penalty is not monotone**, and that is the interesting part: the eye at 5 km is better than back
to back. Dispersion acts on a signal whose instantaneous frequency moves with its power, and over a
short span that partly undoes the laser's own ringing before it starts spreading the pulses. Past
10 km there is nothing left to undo, which is the reach limit a DML is specified with.
`examples/python/dml_reach.py` prints all three tables.

### And the noise it makes on its own

Spontaneous emission is random as well as a rate. With `noise` set, the rate equations carry the
textbook Langevin forces — correlated photon and carrier noise and a phase kick of `1/sqrt(P)` per
spontaneous photon, and nothing else. What the textbooks derive from them then comes *out*:

```
   linewidth at twice threshold     Schawlow-Townes   Henry        measured
   alpha = 0                        3.23 MHz          3.23 MHz     3.08 MHz
   alpha = 2                                          16.14 MHz    16.46 MHz
   alpha = 4                                          54.88 MHz    56.18 MHz
   intensity noise peaks at 5.07 GHz; the relaxation frequency is 5.09
```

Henry's factor `1 + alpha^2` — seventeen, for `alpha = 4` — is written nowhere in the integration.
Each spontaneous photon moves the intensity, the gain restores it by moving the carriers, the index
follows them, and the phase takes a second kick: that is where it comes from, and it is why a DFB's
line is tens of megahertz and not the three that phase diffusion alone would draw. Gain compression
moves the line without widening it, and the measurement takes that offset out.

### And the heat it makes on its own

Most of a laser's drive becomes heat: `I(V_j + I R_s)` goes in and only the light carries any of it
away. The junction warms by `R_th` times what is left, and a warm junction emits longer — a DFB's
grating moves 0.09 nm/K. Set `thermal` and the block computes both halves of that, which live on
two different time axes.

**The bias sets where the line sits.** Settled, with 45 K/W and the defaults here:

```
   bias    junction   wavelength     against cold
   20 mA     0.90 K   1310.0810 nm    -14.1 GHz
   35 mA     1.60 K   1310.1437 nm    -25.1 GHz
   50 mA     2.36 K   1310.2128 nm    -37.2 GHz
   80 mA     4.20 K   1310.3782 nm    -66.1 GHz
```

That is 4 to 6 pm per milliamp, which is what a DFB is specified at, and a quarter of a 100 GHz
channel at the default bias. It belongs to the **band's own centre frequency**, not to the phase: a
window of a few hundred nanoseconds cannot hold 25 GHz as a phase ramp without aliasing it.

**The pattern moves it, slowly.** The junction follows its dissipation through one pole of about a
microsecond, so a long mark warms it and the line drifts red through it. That part *is* carried in
the phase, because it is what changes inside a window, and it is thermal chirp — the low-frequency
tail under the adiabatic chirp the carriers make, and the reason a line code's run length matters to
a directly modulated laser:

```
   mark, 30 to 50 mA      junction    line drifts
     100 ns               +0.095 K      -1.34 GHz
     500 ns               +0.394 K      -6.03 GHz
    2000 ns               +0.865 K     -13.45 GHz
```

The temperature is a third state beside the carriers and the photons, integrated with them and
started settled as they are. A test runs it to rest against `R_th P_diss` — two code paths, one
closed form — and another measures the exponential's time constant back out of a step, to 5 %.

**A Fabry-Perot laser has more than one mode**, and `maiman.laser.integrate_multimode` gives it them:
one carrier reservoir, a parabolic gain curve, a Langevin force per mode drawn against the one carrier
force. The modes trade power, so the total is quiet and each alone is not — for seven modes a
nanometre apart the total's noise is a twentieth of the sum of theirs. Dispersion delays each mode by
its own amount, and `dispersed_power` undoes the cancellation:

```
   D L            received noise (relative variance)
   0              3.8e-3    the laser's own
   0.3 s/m        1.1e-2    18 km of standard fibre at 1550 nm
   1 s/m          2.5e-2    60 km
   3 s/m          5.0e-2    180 km, most of every mode's
```

A floor no received power lifts: mode partition noise. `examples/python/laser_noise.py` prints the tables.

### The same noise, in the link

`dispersed_power` does it at the laser, where the modes' delays are a number. In a link they are not:
every band is an envelope in its own retarded frame, where the constant group delay a span gives it
has been divided out, and a detector summing several of them would add them as though they had all
arrived at once — and report the laser's quiet total.

So the delay is **carried**. `FabryPerotLaser` emits one band per longitudinal mode; a span with
`carry_walkoff` writes each band's `D L dlambda` into the signal's `WalkoffHistory`, beside the
dispersion and Kerr histories it already carries; and the detector spends it, delaying each band's
power by its own amount before the sum. Nothing is applied inside a band, where it would cancel.

```
   7 modes, 1.1 nm apart, 17 ps/nm/km, 12.8 ns window

   span     modes arrive over   received noise (relative variance)
   0 km          0 ps           1.58e-2   the laser's own, the partition cancelled
   5 km        561 ps           1.29e-2   still largely correlated
   20 km      2244 ps           3.33e-2
   50 km      5610 ps           6.47e-2   against a ceiling of 8.58e-2
```

The ceiling is the sum of the modes' own variances — what a receiver sees once nothing cancels at all
— and it is reached from below. The dip at 5 km is real and not a glitch: a delay shorter than the
fluctuations' own correlation time leaves them partly aligned, and partly cancelling still.

Held to `dispersed_power` where that function's approximation holds — each mode's bandwidth far below
the spacing between them — to within a thousandth of the mean, two code paths that share nothing but
the physics. Everything is behind `carry_walkoff`, which starts false: a link that does not ask sees
exactly what it saw before.

## A PAM4 lane, and the equaliser that makes it work

Data-centre optics carry a lane on PAM4: two bits a symbol, four intensities, one photodiode. It halves
the symbol rate NRZ would need and pays three ways — a third of the eye height, a modulator whose
`cos²` compresses the outer eyes, and a receiver bandwidth that was generous for NRZ smearing each
symbol into the next. `PAM4Driver` Gray-codes the bits and, with `predistort`, chooses voltages that
put the modulator's four *powers* evenly apart. `FFEDFEEqualizer` takes one sample a symbol and
equalises it with feed-forward taps and decision feedback.

The error rate is `ser_pam`, which is one axis of `ser_qam` — square QAM is two PAMs, and the tests hold
the two to each other exactly — and matches errors counted in Gaussian noise to 5 %. The equaliser is
trained on the reference, as a compliance receiver is, and then **measured with its taps frozen and its
feedback fed its own decisions**, so a DFE's error propagation is in the count. Its taps converge to the
least-squares solution for the same samples, and a feedback tap to the postcursor it cancels to 1e-3.

```
   26.5625 GBd PAM4 into a 7 GHz receiver, 4096 symbols
   equaliser        SNR       symbol errors
   none              9.70 dB  642
   FFE, 5 taps      31.51 dB    0
   FFE, 9 taps      38.64 dB    0
```

**Two bugs this found before it shipped.** A photocurrent is ten thousand times smaller than a symbol
level, so normalised LMS — which divides by the regressor's length — spent almost all of that length on
the bias and the feedback, and a nine-tap FFE equalised no better than one tap: 14.55 dB either way. The
input is standardised before adaptation now. And the test that was to catch a DFE cancelling a
postcursor had built a precursor, which no DFE can reach. `examples/python/pam4_lane.py` prints the table.

**A receiver in the field has neither a reference nor a choice of sampling phase**, so the
equaliser has both answers. `fractional` takes two samples a symbol and spaces the taps half a symbol
apart: sampled half a symbol late, a symbol-spaced FFE is left with an MSE of 0.32 and six percent of
its symbols wrong, and a T/2-spaced one of the same span decides every symbol at any phase. `blind`
adapts on its own decisions (decision-directed LMS), the reference used only to count: it lands on the
reference-trained taps exactly — even from a start where a third of the raw decisions are wrong — up
to a postcursor of 0.4. At 0.5 it does not converge at all while the reference-trained one still
decides every symbol, and the tests keep that edge in view rather than hide it.

## Getting light onto a chip

Every photonic block here used to assume the die was perfectly aligned to the fibre and that
reaching it cost nothing. In a real photonic link budget the coupling is often the largest line — a
few decibels per facet, where a ring filter costs tenths. Two blocks are that step, and both project
the signal onto the die's axes first: `rotation` decides how much of it is TE.

### An edge coupler is two Gaussian beams and two facets

The fibre's mode and the chip's are each a Gaussian, and their overlap — with a lateral offset, a
tilt, and a free-space gap the fibre's beam diffracts across — is a Gaussian integral in closed form.
It is checked against the integral itself, with the beam propagated across the gap by FFT, to 2e-7,
and against the textbook offset, mismatch and tilt limits exactly. The Gaussian that stands in for a
fibre's mode (Marcuse's spot size) is checked against the mode: 0.7 % wider than the best fit to the
solver's own LP01 field, and 99.2 % of it in power.

```
   standard fibre (10.4 um mode) onto a 3 um spot-size converter
   mode mismatch                     -5.47 dB
   two facets in air                 -0.31 dB
   through the joint                 -5.78 dB

   matched modes, misaligned
   1 um lateral offset               -0.16 dB
   2 um                              -0.64 dB
   10 um air gap                     -0.04 dB
   50 um air gap                     -0.82 dB
   bare silicon facet, both facets   -1.75 dB
```

By default the two facets are independent losses — the average over an etalon fringe. Set
`etalon` and they are a cavity: the chip mode receives a sum of beams, the `n`-th having crossed the
gap `2n + 1` times and diffracted over all of it, each projected with its own overlap and its own
Gouy phase. With modes too wide to diffract that is the Airy function of a thin film, to 5e-5; at
zero gap it is the fibre touching the chip, one interface. Against a brute-force propagation of the
same cavity — an angular-spectrum FFT, twelve bounces summed by hand — it agrees to 2e-4.

**Tilting the fibre walks the cavity off.** The fibre's facet is square to the fibre, so a tilt turns
one mirror against the other; unfolded, each round trip turns the beam by `2θ` and carries it
sideways, and each bounce overlaps the chip mode less:

```
   standard fibre onto a 3 um mode, 50 um of air
   square on           0.49 dB ripple, fringes 3.0 THz apart = c / 2g
   fibre at 8 degrees  0.04 dB
   index-matched       none — nothing to reflect
```

Which is why fibre arrays are polished at an angle. Not modelled: any difference between a chip
mode's TE and TM sizes.

### A grating coupler is phase matching, and its stack

Where a grating coupler couples best is physics: the fibre's tangential wavenumber plus one grating
vector is the guided mode's, with that mode's dispersion in it. A 611 nm period in 220 nm silicon,
fibre at 10 degrees in air, centres at 1549.8 nm and tunes by **7.0 nm per degree** of fibre angle.
The same grating without dispersion would tune by 10.5, which is why the group index is a parameter.

How well and over how wide a band is by default a compact model — a peak loss and a 1 dB
bandwidth, a Gaussian passband between them — which is what a foundry's PDK publishes. Set
`from_stack` and it is computed instead, from three things that multiply:

* **What goes up rather than into the substrate.** The grating is a sheet source in the middle of
  the silicon; the silicon, the buried oxide and the substrate beneath are a thin-film stack, and the
  share leaving upward follows from the reflection each half presents. Checked against a direct solve
  of every layer's boundary conditions to 1e-12. The oxide is a cavity: the share swings between 0.39
  and 0.77 as it goes from one micron to three, and the standard 2 microns sits at 0.59.
* **What the fibre accepts.** The grating's beam is exponential and the fibre's Gaussian; the best
  they can do is 80.1 %, the ceiling every uniform grating coupler lives under.
* **How fast the beam turns.** The emission angle moves with wavelength by phase matching and the
  fibre's does not, so away from the centre the two beams meet at an angle. That is the passband.

```
   220 nm silicon, 2 um oxide, air above, fibre at 10 degrees
   peak                          -3.19 dB at 1545.9 nm
   1 dB bandwidth                35.7 nm
   2.2 um oxide instead          -2.08 dB
   20 um fibre mode              24.6 nm     accepts fewer angles
   group index 3.6               39.5 nm     turns more slowly
```

The compact model's defaults were 3 dB and 35 nm; the stack arrives at 3.19 and 35.7 from nothing
but its geometry. It peaks 4 nm short of the phase-matched centre, because the share going up is
still rising there. TM is rejected by `tm_extinction`: an x-polarized carrier onto a die rotated by
90 degrees couples 25 dB worse than onto one aligned to it.

### What its teeth send back, and what the gap above it does

Four more things the stack can answer, each off by default.

**The reflection into the waveguide.** The same corrugation that radiates on its first order couples
the forward mode to the backward one on its second: a Bragg grating written in the chip, at
`2 n_eff Λ / 2`. Set `from_teeth` and it is solved rather than quoted, from the tooth's index step
and its duty cycle. Three things fall out of it:

```
   fibre angle   period that radiates there   reflection
   0 degrees     572.0 nm                      -1.6 dB     on its own Bragg line
   2 degrees     579.4 nm                     -12.2 dB
   5 degrees     591.0 nm                     -19.4 dB
   10 degrees    611.1 nm                     -21.5 dB
   15 degrees    632.3 nm                     -26.2 dB
```

- **The angle is the whole defence.** A vertical coupler reflects its own Bragg line; the detuning
  that keeps a tilted one quiet is `2π n sin θ / λ`, the same phase matching that sets the emission
  angle. Tilting the fibre is not only about the beam.
- **A duty cycle of exactly one half reflects nothing**, because `sin(2π · 0.5)` is zero and the
  second harmonic with it. Either side of a half it comes back, symmetrically.
- It is checked against this library's own transfer-matrix grating, which shares no algebra with the
  closed form, to 1e-9 across the band.

**The height the fibre sits at.** `fibre_height` is the gap between facet and chip, and three
things follow. The beam is wider where it lands, so a fibre 150 µm up couples more than a decibel
worse. It arrives curved, which the overlap integral carries as a phase. And the gap is an etalon
between the fibre's facet and the chip's own reflection, ringing at `λ²/(2nh)` — measured against
that closed form to 5 %. The tilt walks each round trip `2h tan θ` sideways, so the ripple falls
from 0.81 dB at normal incidence to 0.06 dB at 12 degrees, which is the same reason an edge
coupler's facet is angled.

**A mirror under the oxide.** `substrate_extinction` makes the substrate absorbing — aluminium is
`1.44 + 16j` — and what would have gone down comes back up: the share leaving upward rises from
**0.59 to 0.99**. An index with the other sign would be a gain medium, and is refused rather than
quietly flipped.

**Apodization.** `grating_strength_end` tapers the radiation strength along the grating. The
emitted profile follows from energy conservation alone, `sqrt(2α(x)) exp(−∫α)`, and a taper makes
it rounder — closer to the fibre's Gaussian than the uniform grating's 80.1 % ceiling:

```
   uniform, 0.14 /um                 -3.19 dB
   tapered, 0.05 to 0.4 /um          -2.63 dB
   with a mirror as well             -0.48 dB
```

Half a decibel from the taper and two more from the mirror, from geometry and a complex index.

## What a layout tool knows, and what it does not

A `.pdk` here carries a foundry's *fitted numbers*. gdsfactory draws the mask. Joining them is the
last thing the roadmap had open, and doing it started with two measurements — because this project
has twice now found that a stated obstacle was never measured.

**The first measurement was the licence.** gdsfactory itself is MIT. Its dependency tree is not:

```
gdsfactory  →  kfactory  →  klayout          GPL-3.0-or-later
86 packages resolved in total
```

That is the exact ground this project already refuses FFTW (GPL-2.0-or-later) and klujax
(LGPL-2.0-only) on, and `pyproject.toml` says licences are checked before adoption rather than
after. So gdsfactory is **not a dependency, not even an optional one** — nothing here imports it.

**The second measurement changed what the join could be.** A gdsfactory `CrossSection` carries
`sections` (width, offset, layer), `radius`, `radius_min`, `bbox_layers` — and `extra="forbid"`, so
a kit author cannot even attach anything else. There is no effective index, no group index, no
propagation loss and no dispersion anywhere in the package; the only `neff` in it computes grating
tooth pitch. **A layout tool does not know what any of it does optically**, so importing a PDK
*from* gdsfactory cannot produce the numbers a `.pdk` is made of. They were never there.

So the join runs the other way round, and it is the natural division of labour:

```
netlist (gdsfactory)  →  topology + geometry: what is wired, how long the router made it
.pdk       (maiman)   →  which model each cell is, and what this process measures
Circuit.solve()       →  the answer
```

An instance says it is a `straight` drawn in cross-section `strip`; the kit says `straight` is a
`Waveguide` in this process, that its `o1` is that model's `in`, and that `strip` is 2.44 / 4.20 /
2 dB/cm. **That mapping lives in the file and not in this library** — another process draws the
same geometry under another name, and a table of cell names compiled into a simulator would be
wrong for every kit but one.

`python examples/python/netlist_circuit.py` solves a netlist **gdsfactory actually emitted**, shipped
unmodified in `tests/data/` with its MIT attribution:

```
instances                       cell         length
bend_euler_R10_A90_P0p5_2f1…    bend_euler   16.637 um
straight_L10_N2_CSstrip_5000…   straight     10 um

transmission              -0.0053 dB
26.637 um at 2 dB/cm      -0.0053 dB
```

**The bend's 16.637 µm is its arc length**, and the radius alone does not give it — you need the
Euler parameter too. The kit asks for it with `from_netlist: {length: info.length}`, so every
routed bend and connecting straight carries the length the router gave *that instance* rather than
one nominal value standing in for all of them.

The second circuit in that example is a racetrack ring, hand-written in the same schema because
gdsfactory's shipped samples contain no ring and no coupler, and a circuit without feedback does
not exercise the one thing a scattering-matrix reduction is for. Its free spectral range comes back
at **713.74 GHz against `c/(n_g·L)` = 713.79** for the 100.000 µm its five instances sum to.

**It refuses rather than skips.** A cell the kit does not model, a layout port the kit does not
map, a port wired twice, a circuit with nothing facing outward — each is an error naming the thing.
A reader that quietly dropped what it did not recognise would hand back a circuit that solves,
plots like a spectrum, and is not the circuit on the mask, which is the worst of the three
outcomes.

**YAML is not a dependency either.** The reader works over a parsed mapping and handles JSON with
the standard library; YAML is read when a YAML parser happens to be installed, and the error names
the one-liner that converts a file when it is not. `pyyaml` is in the `dev` extra so CI exercises
that path, and the shipped package still depends on numpy and nothing else.

**What is still missing.** Routing. A schematic's `routes:` section names links between ports, and
the bends and straights that implement them do not exist until the route is built — so a schematic
read at this level is the circuit without its interconnect, and its lengths are missing. Read a
netlist extracted from the built layout, which is what the shipped sample is. And bend loss: a
bend is modelled as a straight waveguide of the same arc length, which a tight bend in a real
process beats by some margin the kit would have to carry.

## The kernels do not know what they are running on

The propagation kernels are the only part of this worth a GPU: a loop over FFTs on a long array,
where everything else is scalar arithmetic or a closed form evaluated a few dozen times.
[`maiman/kernels.py`](../src/maiman/kernels.py) was written as array-to-array functions from the
beginning so that this could be added without touching anything above it, and
[`maiman/backend.py`](../src/maiman/backend.py) is the whole of the addition.

**The arrays decide, not a setting.** A kernel handed CuPy arrays runs on CuPy and returns CuPy
arrays; handed NumPy arrays it runs on NumPy. There is no global mode and no flag on the context,
which matters because the kernels are pure functions and a hidden mode would be the one piece of
state that could make the same inputs give different answers. Dispatch is on the array's own type —
`type(a).__module__` names the package it came from — so there is no registry to keep in sync.

**CuPy is not exercised here**, and saying otherwise would be the kind of claim this project exists
to avoid: there is no device and no install in CI. What *is* tested is the half that would actually
break a port. A second array library — [`tests/hostile_backend.py`](../tests/hostile_backend.py) — sets
`__array_function__` to `None`, which is NumPy's own way for a type to say it is not NumPy's, so
`np.fft.fft` on one of its arrays raises. Universal functions are deliberately left working, because
`np.exp` on a CuPy array dispatches and comes back a CuPy array; refusing them would be testing a
rule that is not true. What breaks a port is anything that *allocates* — `np.fft.fftfreq` and
`np.zeros` build on the host, and a kernel calling one inside its loop pays a transfer every step.

Every kernel is then run on both libraries and the results compared, and the names the second one
was asked for are recorded. That set is the contract, and it is asserted as an equality rather than
a lower bound:

    abs  complex128  conj  exp  float64  max  pi
    fft.fft  fft.fftfreq  fft.ifft  fft.irfft  fft.rfft  fft.rfftfreq

Thirteen names. A change that reaches for something only NumPy has fails in this repository rather
than on somebody's GPU, and one that stops needing something fails too. CuPy provides all thirteen.

Only the propagation path is converted. The four-wave-mixing closed forms and the 2×2 Jones algebra
are scalar work a device would slow down, and there is a test naming which functions are in and
which are out, so the line is a decision rather than an oversight.

## 400G and 800G, and what they cost

Three reference transceivers, all dual-polarization coherent, all derived from one number and
arithmetic. 400G is DP-16QAM at 59.84 GBd — the shape a 400ZR module has. 800G is that payload
doubled, and there are two ways to double it:

| configuration | GBd | line rate | payload | slot | b/s/Hz |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 400G DP-16QAM | 59.84 | 479 Gb/s | 400 G | 75 GHz | 5.33 |
| 800G DP-16QAM | 119.68 | 957 Gb/s | 800 G | 150 GHz | 5.33 |
| 800G DP-64QAM | 79.79 | 957 Gb/s | 800 G | 100 GHz | 8.00 |

Line rate is baud × bits × 2 polarizations; a 400 Gb/s payload inside 479 leaves 16.4 % for forward
error correction and framing. **Nothing here is quoted from a standard** — the 800G symbol rates
follow from the 400G one, and the required OSNR below is *measured*: a noise-loaded link, bisected
until the **counted** bit error rate lands on the threshold, then compared against a relation that
knows nothing about any of it.

| configuration | closed form | ideal DSP | penalty | blind equaliser |
| :--- | ---: | ---: | ---: | ---: |
| 400G DP-16QAM | 19.47 dB | 19.78 dB | +0.31 | 20.24 dB |
| 800G DP-16QAM | 22.48 dB | 22.86 dB | +0.38 | 23.32 dB |
| 800G DP-64QAM | 26.41 dB | 27.22 dB | +0.81 | 28.56 dB |

The closed form assumes a perfect transmitter, perfect DSP and a noiseless receiver, so the fourth
column is the transmitter's implementation penalty — growing with the format order, because a
denser constellation is less forgiving of the modulator's curvature.

**The fifth column is what the DSP costs on top**, and it used to be a finding. Nothing rotates the
polarization on this bench, so a perfect separator would be the identity and the blind butterfly
equaliser should be free. At 64-QAM it cost eleven decibels.

Chasing that down: it is not the step and not the decision — freezing the radius-directed stage
entirely still left 3249 symbol errors, which identified the *first* stage. A polarization rotation
is **memoryless**: a 2×2 complex matrix, four numbers. Fitting it with seven taps per path means
twenty-eight, and the twenty-four that are not needed fill with gradient noise, which the
single-radius cost has no per-sample truth to hold down. At 64-QAM that noise is the whole margin;
a single tap had already been measured to leave a residual of 0.0013 where seven left 0.29.

Adapting only the centre tap while that stage runs takes 64-QAM through a rotated channel from 3568
symbol errors to **zero at every angle** — and breaks the other case, where the eye is closed by
*memory* rather than by rotation and every tap has to move: at 60 ps/nm of residual dispersion it
costs 16-QAM 1955 errors where letting them all run costs 44. Neither wins outright, and which one a
channel needs is not knowable in advance, so the block runs both and keeps whichever lands closer to
the constellation's rings. That is not a new criterion — it is the statistic the second stage already
steers by, read as a score. Over ten cases it picks right in all of them, and 16-QAM at fifteen taps
improved from 544 errors to 53 as well.

Five other approaches were measured and moved nothing: gating the second stage's decision on whether
the ring it picked was credible, normalising the update by the window energy, annealing the second
stage's step, leaking the weights towards zero, and shortening the first stage.

With the DSP out of the way, the 800G choice is two numbers:

    twice the baud, same format    +3.08 dB of OSNR, twice the spectrum
    denser format, less baud       +4.36 dB of OSNR, two thirds of it

The first is the price of bandwidth and is 3 dB in theory: twice the symbol rate collects twice the
noise and nothing else changes. The second buys a third of the spectrum back, and is what a link
with filled fibre and optical SNR to spare pays for it.

`maiman.analysis` carries the bridge these rest on — `snr_from_osnr`, `snr_for_ber` and
`required_osnr` — and the two reference designs ship as
[`examples/maiman/zr400.maiman`](../examples/maiman/zr400.maiman) and
[`examples/maiman/zr800.maiman`](../examples/maiman/zr800.maiman), laid out and openable in the studio.
[`examples/python/reference_rates.py`](../examples/python/reference_rates.py) builds them and prints the tables.

## Mixing products add in field, not in power

A link is not one fibre. The four-wave mixing product a span generates arrives on top of the one
the span before it generated, and whether those add or cancel is decided by how far the pumps and
the product have drifted apart over the fibre already behind them. That drift is the phase mismatch
integrated over the distance travelled, and it used to be thrown away: every span drew its products
a fresh random phase, so they added in power and a four-span link came out four times one span
instead of sixteen.

Three carriers at −30 dBm, four 80 km spans against one, each span's loss exactly undone:

| span | D = 0 | × one span | D = 17 | × one span |
| ---: | ---: | ---: | ---: | ---: |
| 78 km | −104.06 dBm | 16.00 | −180.98 dBm | 0.03 |
| 80 km | −104.04 dBm | 16.00 | −154.84 dBm | 13.17 |
| 82 km | −104.02 dBm | 16.00 | −166.46 dBm | 0.89 |
| 85 km | −103.99 dBm | 16.00 | −160.89 dBm | 3.12 |

**The sharp form of the claim is that cutting a span in half must change nothing.** A span boundary
is a bookkeeping decision, not a physical one, so 320 km of lossless fibre must give the same
product whether it is run as one block or as eight — and it now does, to under a hundredth of a
decibel, at every dispersion. Adding the pieces in power put eight of them 9 dB below one.

Note what is *not* claimed: that dispersion suppresses the build-up. At the right span length it
does the opposite — the rotation per span comes back round to a multiple of 2π and the spans stack
again, which is the 13.17 in the table. That periodic re-phasing is a real property of a
dispersion-managed link and is why the map is designed rather than chosen; a model that reported
"less" would be reporting something false.

**One number carries all of it.** The mismatch is a difference of four propagation constants at
frequencies satisfying `ω_i + ω_j = ω_k + ω_F`, so the constant and group-delay terms cancel
identically and only the β₂ term survives. The signal therefore carries a single accumulated
`Σ β₂·L` and nothing about any band's absolute phase — which is fortunate, because `β₀·L` is of
order 10¹¹ radians over 80 km and reducing that modulo 2π in double precision would leave about
five digits of the answer. A dispersion-compensating span is a fibre with negative D, so it
subtracts from that sum on its own; nothing special is done for it.

What remains drawn is one phase per *triplet*, standing in for the pump phase combination that
modelling the pumps by their powers has thrown away. It is keyed on the three pump frequencies and
on nothing else — not on the block, not on which span it is — because the same three pumps make the
same product wherever they are, and a phase redrawn per block is precisely what made the spans
average instead of add.

### And the pumps' own phase

Self- and cross-phase modulation turn a product's drive, `A_i A_j A_k*`, and the carrier the product
lands on by different amounts. With cross-phase modulation on, the difference is the textbook
`−γ(P_i + P_j − P_k − P_F)`: the cross-phase of everything else turns every carrier alike, so it
cancels, and so does the orthogonal polarization's. That difference shifts the mismatch.

Set `pump_phase` on the fibre and the block carries the shift in three places:

1. **Inside a span**, as a rate the mixing integral carries down the span as the pumps decay, like
   `L_eff(z)` rather than `z`. It is summed as a series in `r/α`, or by quadrature beyond it; the
   two agree to 1e-12.
2. **Between spans**, as how far the drive has turned against the product's carrier in the spans
   before. The signal now carries a Kerr history, the nonlinear counterpart of its accumulated
   dispersion: each carrier's angle, plus the angle a weak carrier on the same path would have.
3. **In the product's frame.** The split-step keeps turning the band a product was added to, so a
   new span's share has to arrive turned the same way, or the spans do not add as the fibre adds
   them.

What checks it is the split-step solution of the same fibre with every tone in one band. That
solution has no mixing model at all: the products grow out of `|A|²A`. The test uses one strong pump
and one weak signal, so the products stay small and do not mix again among themselves. Then the
only thing the linear model leaves out is the Kerr phase. At 20 mW, the product 100 GHz below the
pump after four amplified 80 km spans:

| D [ps/(nm·km)] | linear mismatch ÷ split-step | with `pump_phase` ÷ split-step |
| ---: | ---: | ---: |
| +4 | 23.9 | 1.04 |
| −4 | 0.044 | 1.01 |
| +17 | 0.021 | 0.99 |

A single span of 10 km moves it by 12 to 29 %, and with it the error is 2 %. The flag is off by
default, because turning it on changes which way the drawn phase and the mismatch combine, and so
moves every product. A join of two paths that went through different Kerr fibre is recorded rather
than refused, and only a span that asks for `pump_phase` refuses it.

## The short wavelengths pump the long ones

A photon can scatter off a silica vibration and come out at a lower frequency, and the process is
stimulated — light already there at the lower frequency makes it more likely. So in a comb the
short-wavelength channels pump the long-wavelength ones, and a flat launch does not arrive flat.
Set `raman_gain_slope` on the fibre. One 80 km span of standard fibre, everything launched at
0 dBm:

| comb | total | span | tilt |
| :--- | ---: | ---: | ---: |
| 4 × 100 GHz | 6.0 dBm | 0.30 THz | +0.00 dB |
| 40 × 100 GHz | 16.0 dBm | 3.90 THz | +0.40 dB |
| 80 × 50 GHz | 19.0 dBm | 3.95 THz | +0.81 dB |
| 80 × 100 GHz | 19.0 dBm | 7.90 THz | +1.63 dB |

A filled C band loses most of a decibel across itself every span, and it accumulates. That is a
large fraction of the margin a link is designed with, and it is why a line system is built with a
tilt to undo rather than assumed flat. A four-channel comb — which is what every other WDM number
here is measured on — moves by three thousandths of a decibel, which is why this was never missed.

**Power is moved, not lost.** The tilt is a closed form (Zirngibl, *Electron. Lett.* 34(8), 1998)
and the sum over channels is unchanged to floating point; the quantum defect the lattice keeps is a
part in ten thousand at these separations and is not modelled. That conservation is also what makes
a sign error impossible to hide — the two ends have to move in opposite directions or the sum
cannot come out.

The gain rises linearly with separation up to its 13.2 THz peak, and inside that the closed form is
exact. The C and L bands together are *not* past it — 1530 to 1610 nm is 9.7 THz, which these notes
used to get wrong. It takes the S band as well. A comb wider than the peak is integrated instead, with
silica's measured gain shape and photons rather than watts conserved, and the difference is not small:

```
   S + C + L, 1460 to 1625 nm, 209 channels at 0 dBm, one 80 km span
   straight line past the peak       11.19 dB of tilt
   silica's shape                     6.39 dB
```

The shape is a coarse trace of the measured spectrum, good to tens of percent past the peak. The
`diagnostics` port reports the tilt in dB, so what happened is a number rather than an assumption.

## What the pumps pay

Four-wave mixing used to make its products out of nothing: the pumps left a span exactly as bright as
if they had made no products at all, so a lossless span came out with more power than it was given.
Each product now takes its photons from the two pumps that made it and gives one to the idler, which
is parametrically amplified. A lossless span conserves energy to floating point, and
`diagnostics.fwm_depletion` is the fraction the products hold.

The product itself is still the undepleted-pump formula, and the honest check is the full nonlinear
Schrödinger equation — two tones in *one* band, where the split-step solves the Kerr term exactly:

```
   gamma P L    pump loss, model / exact
       0.065                    1.002
       0.195                    1.019
       0.390                    1.079
```

Within a percent where products are small against the pumps, and drifting high as they are not. At
link powers — four channels at 0 dBm through 80 km of standard fibre — the pumps give up parts per
million. Where the formula would ask a pump for more than it holds, the span is refused rather than
balanced by a fiction.

## An orthogonal neighbour is not an absent one

The Kerr coupling used to be scalar per polarization: a channel was modulated by its neighbours'
co-polarized power and by nothing else, so a channel polarized across it counted for zero. In an
isotropic medium orthogonal power counts for exactly two thirds — which, since co-polarized
cross-phase modulation already carries its factor of two, makes orthogonal cross-phase modulation
exactly **one third** of co-polarized. Set `cross_polarization` on the fibre:

| neighbour | axes uncoupled | axes coupled |
| :--- | ---: | ---: |
| co-polarized | 2.000 γPL | 2.000 γPL |
| orthogonal | 0.000 γPL | 0.667 γPL |

The same coefficient does two more things, because it is the same coefficient. A channel's own
orthogonal component modulates it at two thirds the rate, so power split evenly between the axes
turns each of them by `(0.5 + ⅔·0.5)·γPL` instead of `0.5·γPL`. And the two axes, no longer
accumulating the same phase, rotate the state of polarization as the power moves — nonlinear
birefringence, which falls by a factor of three, and cross-polarization modulation, which is that
rotation being driven by a *different* channel's power.

The value is the fixed-axis one, from the χ⁽³⁾ tensor rather than from any averaging. Two further
settings finish the picture, both off by default so no earlier result moves.

**`coherent_polarization`** keeps the `A_x* A_y²` term, which moves power between the axes rather
than only dephasing them. It is not a phase in the x/y basis, but in the circular basis the self and
cross terms become pure phases again, so the step stays exact — checked against a direct Runge-Kutta
integration of the x/y equations to 1e-12. What it changes is measured, not asserted:

```
   circular light, nonlinear phase over gamma P L    2/3 with it, 5/6 without
   linear light                                      exactly 1 either way
   an ellipse                                        its axes turn at (2/3) gamma S3 per metre,
                                                     with its power and ellipticity unchanged
```

**`interleave_pmd`** applies the PMD waveplates along the span, between Kerr steps, rather than all
after it — so the Kerr effect acts on a state of polarization that is still rotating. With the Kerr
effect off the two arrangements are the same operator, and agree to 2e-16.

**What neither changes is the Manakov average**, and the tests say so rather than pretending
otherwise. Scramble the state fast enough — 800 waveplates across the span — and the averaged
nonlinearity lands on 8/9 of `gamma P` *with or without* the coherent term. Averaged over every
polarization state the two forms have the same invariant part, `(4/9) S0²`; the coherent term decides
how the state evolves on the way, not that average. An earlier draft of this note claimed the
phase-only form could not reach 8/9. The measurement said otherwise, and the algebra agrees.

Off by default, and with all the light on one axis it changes nothing at all — on the samples,
which is what makes it safe to leave on for a dual-polarization link and pointless for a
single-polarization one.

## D is not one number

Standard fibre gains 0.058 ps/nm/km of dispersion for every nanometre up the
band. Set `dispersion_slope` and the fibre stops pretending otherwise — and the same
coefficient shows up in two places at once, because it is the same coefficient.

**Across a comb**, channels no longer share a dispersion, so one compensator setting cannot serve
all of them. Over 80 km, with the compensator set for 1550 nm:

| channel | D there | D·L there | mismatch | EVM, one setting | EVM, its own |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1550 nm | 17.000 | 1360.0 | +0.0 | 1.68 % | 1.68 % |
| 1554 nm | 17.232 | 1378.6 | +18.6 | 4.07 % | 1.68 % |
| 1558 nm | 17.464 | 1397.1 | +37.1 | 7.76 % | 1.68 % |
| 1566 nm | 17.928 | 1434.2 | +74.2 | 15.23 % | 1.68 % |

That is the same size of penalty as the mis-settings table above it, arriving without anybody
having mis-set anything. It is why a single dispersion-compensating fibre never flattened a whole
C-band, and why coherent receivers carry a per-channel setting.

**Within one channel** the slope is a cubic phase, and cubic phase broadens a pulse
*asymmetrically* where β₂ broadens it evenly. It is honestly small at these baud rates: over
1000 km the cubic phase across a 32 GBd band is about 0.04 radians and costs a tenth of a point of
EVM. `accumulated_slope` on the compensator removes it anyway, and the test that it does is worth
having because a slope compensation with the wrong sign *doubles* the residue rather than removing
it — and nothing about the width of a pulse could ever tell you.

**β₃ is not the slope by another name.** Even at zero slope a fibre has a nonzero β₃, because
holding D flat across wavelength is itself a statement about how β₂ varies:

    beta3 = (lambda^2 / 2*pi*c)^2 * (S + 2*D/lambda)

For standard fibre at 1550 nm, D = 17 and S = 0.058 give β₃ = 0.13 ps³/km, the value the
literature quotes. Feeding it S = 0.09 — the slope at the *zero-dispersion* wavelength, which is
the number datasheets lead with — gives 0.18, and is the easiest way to be forty percent wrong.

The cubic term's sign is derived from the same Taylor expansion of β(ω) that gives the group delay
and the dispersion, not written down, because with this module's transform pair the quadratic and
cubic terms land on *opposite* signs. Broadening is even in β₃, so a sign error produces exactly
the right width and exactly the wrong skew; the tests measure the skew.

Left at zero the slope changes nothing. Every number taken before it existed still comes out, on
the same samples.

## Finding the dispersion without being told it

A dispersion compensator has to be set to within a few ps/nm. Over 80 km at 32 GBd the true value
is 1360, and being 17 out — one kilometre of fibre — takes the EVM from 1.7 % to 3.8 %, while 136
out lands the symbols at chance. It is not a knob to be roughly right about, because the residual
after a mismatch *is* the mismatch.

No deployed receiver is ever told the number. It measures it during acquisition, from the signal,
before the equaliser or the carrier loop have converged. Set `estimate` on the compensator and it
does the same:

| span | accumulated | estimated | error | EVM declared | EVM blind |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 km | 0 ps/nm | −7.4 | −7.4 | 1.68 % | 2.23 % |
| 20 km | 340 ps/nm | 336.4 | −3.6 | 1.67 % | 1.82 % |
| 80 km | 1360 ps/nm | 1355.3 | −4.7 | 1.68 % | 1.92 % |
| 400 km | 6800 ps/nm | 6793.9 | −6.1 | 1.67 % | 2.05 % |
| 1000 km | 17000 ps/nm | 16991.5 | −8.5 | 1.68 % | 2.39 % |

The last two columns are the ones to read together: being right in ps/nm is not the claim, leaving
behind a link as good as one that was handed the answer is.

**Two stages, because no one statistic does both jobs.** The intensity of a modulated signal
repeats at the symbol rate, so `|A|²` carries a line there — and dispersion gives every pair of
frequencies a symbol rate apart a relative phase that grows with frequency, so their contributions
cancel and the line fades. Scanning that line across ±20000 ps/nm acquires the value with an
enormous capture range and an error of tens of ps/nm. Then a second stage minimises how Gaussian
the intensity looks, which has curvature exactly where the tone is flat.

The scan never compensates anything. The line at the symbol rate is a correlation of the spectrum
with itself shifted by that rate, and compensation only multiplies the spectrum by a phase, so
every candidate is one dot product against a product computed once from a single transform. It is
an identity, held to 1e-12 against compensate-then-transform in the tests, and it made 400
candidates over a 65536-sample window twenty times cheaper.

**It needs excess bandwidth**, and says so: at zero roll-off the tone this searches for does not
exist at all. Between there and roll-off 0.1 it exists but acquisition comes in a consistent
950 ps/nm low, and what recovers those runs is refinement walking its window in rather than any
warning from the contrast figure — which scored 29 on a run that was wrong by 950 and 32 on one
that was right. Resolution also scales as 1/R\_s²: the same link at 10 GBd lands 44 ps/nm out where
32 GBd lands 5. All of that is in
[`estimate_dispersion`](../src/maiman/dsp.py)'s docstring, measured rather than asserted, and the
third table of [`examples/python/dispersion_link.py`](../examples/python/dispersion_link.py) prints it.

## What channels do to each other

Until recently bands propagated independently through the fiber, which made this a good model of
one channel and an optimistic model of a comb. They no longer do. The same `|A|²A` term that gives
a channel its own self-phase modulation lets every *other* channel rotate its phase — and the
coefficient is not free. Expanding the term for a sum of carriers, a channel's own power appears
once and a neighbour's appears twice, because there are two ways to choose which un-conjugated
factor belongs to the neighbour and one way when it is the channel itself.

That factor of two is measurable, and it is exact:

| co-propagating channels | mean nonlinear phase | vs one channel | closed form |
| :--- | ---: | ---: | ---: |
| 1 | 0.027518 rad | 1.000× | 1× |
| 2 | 0.082559 rad | 3.000× | 3× |
| 3 | 0.137600 rad | 5.000× | 5× |
| 4 | 0.192640 rad | 7.000× | 7× |

**Dispersion is the cure here, not the disease.** Chromatic dispersion is normally introduced as
something to compensate. Between channels it is the only thing keeping them apart: it makes them
travel at different speeds, so a neighbour's bit pattern *slides past* instead of sitting on top of
the channel it is modulating. Walk-off is therefore not a separate parameter — it is the
group-delay term of the same expansion of β(ω) that produces the dispersion, so setting D to zero
removes both at once. At 17 ps/nm/km a 100 GHz neighbour separates by 13.62 ps/km, and has slid 140
symbols by the end of a 320 km link.

What walk-off removes is not the cross-phase modulation but its *variation*. The mean phase shift
is fixed by the neighbour's average power and no amount of sliding changes it — sliding
redistributes in time, it does not destroy. That split is the whole mechanism, because a constant
phase offset is absorbed by carrier recovery for free and it is the variation that closes an eye.
Measured on channel 1 of a four-channel QPSK comb over 4 × 80 km, as EVM after carrier recovery:

| launch/channel | D = 0 | D = 17 ps/nm/km |
| :--- | ---: | ---: |
| −3 dBm | +0.18 % | +0.00 % |
| 0 dBm | +0.49 % | +0.03 % |
| +3 dBm | **+2.84 %** | **+0.29 %** |

Ten times less penalty for having dispersion in the fiber.

**Four-wave mixing lands on the channels.** Products appear at `f_i + f_j − f_k`, and on an equally
spaced grid those frequencies *are* channel frequencies — so the crosstalk arrives in band, where
no filter downstream can reach it. The model folds such a product into the channel it lands on,
with a phase drawn from the run's generator for the same reason PMD is drawn: it is set by fiber
details nobody measures. Dispersion suppresses mixing too, by dephasing it, and the phase mismatch
grows as the *square* of the channel spacing:

| channel spacing | D = 0 | D = 2 | D = 17 |
| :--- | ---: | ---: | ---: |
| 25 GHz | 0.0 dB | −4.4 dB | −21.2 dB |
| 50 GHz | 0.0 dB | −14.7 dB | −33.1 dB |
| 100 GHz | 0.0 dB | −26.7 dB | −45.4 dB |
| 200 GHz | 0.0 dB | −38.6 dB | −57.4 dB |

Zero-dispersion fiber is perfectly phase matched at every spacing, which is the whole reason
dispersion-shifted fiber was abandoned for WDM.

The two effects are computed differently, and the difference is worth knowing before reading a
number off the block. Cross-phase modulation is solved on the waveform inside the split-step,
because it depends on the neighbour's instantaneous power sliding past; four-wave mixing is solved
in closed form from the band powers and injected as tones, because the products land at frequencies
no band is sampled at. Not putting the channels on one grid is what makes a WDM comb affordable at
all, and that choice has to be paid for somewhere. See
[`examples/python/wdm_nonlinear.py`](../examples/python/wdm_nonlinear.py).

## A foundry's numbers, not a foundry's code

A process design kit here is **not** a component library. It adds no devices, carries no layout,
and reading one executes nothing — a `.pdk` is JSON, read the way a `.maiman` project is, and for
the same reason: opening a file somebody sent you must not be equivalent to running their code.
`"component": "os.system"` is refused like any other name that is not in the registry.

What it carries is the half of a real design the engine cannot derive. `maiman.photonics` knows
what a directional coupler *is*. It has no way to know that this process makes a nominal 3 dB
coupler measuring 0.48 at 1550 nm, or that the nitride guide beside the silicon one is twenty times
quieter. Those numbers come off a wafer.

```python
from maiman import load_pdk

pdk = load_pdk("examples/python/silicon_220nm.pdk.json")
coupler = pdk.make("dc_3db", label="split")        # 0.48, not 0.5
splitter = pdk.make("mmi_1x4", wavelength=1565.0)  # the fits, evaluated there
```

**A value in a kit may be a fit rather than a constant** — a polynomial in `λ − λ_ref` in
nanometres, which is the form a foundry quotes one in. It is evaluated when the component is built.
The device models here are frequency-flat by construction, so what a kit buys is the right constant
for the band you are working in, chosen by the file rather than guessed. That "3 dB" coupler is a
1.5 / 5.5 dB split at 1500 nm and 4.9 / 1.7 at 1600 — which is the entire reason the MMI sits
beside it in the kit.

**A fit has a window, and running outside it is refused.** This is the part worth the code. That
first-order C-band fit, extrapolated to 1310 nm, returns **minus 0.46** — a negative power fraction,
out of arithmetic that never complained once. `valid_wavelengths` turns it into a sentence naming
the range and the device. Everything else a kit can get wrong — a component that is not registered,
a parameter the component does not declare, a cross-section that does not exist, a cross-section
attached to a lumped device with no waveguide to apply it to — is refused at load, naming the file
and the entry, because a kit is read once and used a hundred times.

And nominal is not what you build. The same 1 × 4 splitter, twice:

    out1     out2     out3     out4
    textbook         -6.021   -6.021   -6.021   -6.021   total +0.000 dB   spread 0.000 dB
    silicon-220nm    -6.298   -6.414   -6.531   -6.648   total -0.450 dB   spread 0.350 dB

See [`examples/python/pdk_import.py`](../examples/python/pdk_import.py) and the kit it reads,
[`examples/python/silicon_220nm.pdk.json`](../examples/python/silicon_220nm.pdk.json) — representative of the open
multi-project-wafer processes and drawn from published literature, not from anyone's confidential
kit. Replace it with yours.
