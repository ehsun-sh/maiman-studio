# ruff: noqa: E501, RUF001 -- prose and formulas, with the dashes and minus signs they need
"""The teaching material the shipped projects carry: a lesson, and marks on the canvas.

Every project the studio opens from the File menu is written by a script in
this folder, and each of those scripts hands its lesson to ``save`` from here.
Keeping the prose in one module rather than in each script means the physics
of a script stays readable, and a correction to a formula is made once.

Two things per project, both ignored by the engine:

* ``NOTES[key]`` -- the lesson the studio opens beside the canvas. The body is
  Markdown with TeX between dollar signs (``$...$`` inline, ``$$...$$`` on its
  own line), and ``[[label]]`` names a block: in the studio it is a link that
  selects that block, so the prose and the schematic point at each other.
* ``marks(key, layout)`` -- frames around the stages of the link and the
  occasional sticky note, placed from the *layout* rather than at fixed
  coordinates, so that moving a block in a script moves its frame with it.

A formula here is a claim like any other in this repository. Each one is the
textbook relation the block implements or the one the reading on screen is
judged by, and where it is a rule of thumb rather than the model it says so.
"""

from __future__ import annotations

from typing import Any

#: The size of a block on the canvas, as the studio draws it.
NODE_W, NODE_H = 116.0, 52.0

Layout = dict[str, dict[str, float]]


def frame(layout: Layout, ids: list[str], label: str, tone: str = "binary") -> dict[str, Any]:
    """A rectangle around the blocks ``ids``, with room above them for its label."""
    xs = [layout[i]["x"] for i in ids]
    ys = [layout[i]["y"] for i in ids]
    pad = 9.0
    x, y = min(xs) - pad, min(ys) - pad - 18.0
    return {
        "kind": "rect",
        "x": x,
        "y": y,
        "w": max(xs) + NODE_W + pad - x,
        "h": max(ys) + NODE_H + pad - y,
        "text": label,
        "tone": tone,
    }


def note(x: float, y: float, text: str, w: float = 210.0, h: float = 96.0) -> dict[str, Any]:
    """A sticky note at a fixed place on the canvas."""
    return {"kind": "note", "x": x, "y": y, "w": w, "h": h, "text": text, "tone": "electrical"}


def arrow(x1: float, y1: float, x2: float, y2: float, text: str = "") -> dict[str, Any]:
    """An arrow from (x1, y1) to (x2, y2), optionally labelled at its middle."""
    return {"kind": "arrow", "x1": x1, "y1": y1, "x2": x2, "y2": y2, "text": text, "tone": "metric"}


def marks(key: str, layout: Layout) -> list[dict[str, Any]]:
    """The marks drawn over project ``key``, numbered in the order they are drawn."""
    return [{"id": f"a{n}", **mark} for n, mark in enumerate(_MARKS[key](layout), start=1)]


def _ook_eye(at: Layout) -> list[dict[str, Any]]:
    fib = at["fib"]
    return [
        frame(at, ["prbs", "drv", "tx", "mzm"], "Transmitter", "optical"),
        frame(at, ["fib"], "Channel", "binary"),
        frame(at, ["pin", "lpf"], "Receiver", "electrical"),
        frame(at, ["eye", "ber"], "Instruments", "metric"),
        note(
            fib["x"] - 40.0,
            fib["y"] + 110.0,
            "Lengthen the fibre and run again: loss lowers the eye, dispersion "
            "smears it sideways. $\\Delta\\tau \\approx D\\,L\\,\\Delta\\lambda$",
            w=230.0,
        ),
        arrow(fib["x"] + 40.0, fib["y"] + 108.0, fib["x"] + 58.0, fib["y"] + NODE_H + 4.0),
    ]


def _wdm_osa(at: Layout) -> list[dict[str, Any]]:
    edfa = at["edfa0"]
    return [
        frame(at, ["ch0", "ch1", "ch2", "ch3", "mzm0", "mzm3"], "4 transmitters", "optical"),
        frame(at, ["f0", "edfa0", "f1", "edfa1"], "2 amplified spans", "binary"),
        frame(at, ["osa", "pm", "osnr"], "Instruments", "metric"),
        note(
            edfa["x"] - 20.0,
            edfa["y"] + 110.0,
            "Each EDFA adds noise as well as gain. The floor it leaves under the "
            "channels is what the OSNR meter reads.",
            w=230.0,
        ),
        arrow(edfa["x"] + 50.0, edfa["y"] + 108.0, edfa["x"] + 58.0, edfa["y"] + NODE_H + 4.0),
    ]


def _dwdm_link(at: Layout) -> list[dict[str, Any]]:
    dcf = at["dcf"]
    return [
        frame(at, [f"ch{i}" for i in range(8)] + ["mzm0", "mzm7"], "8 transmitters", "optical"),
        frame(at, ["mux", "booster", "span", "dcf", "preamp"], "Line", "binary"),
        frame(at, ["demux", "drop_power", "pin", "lpf", "ber"], "Drop one channel", "electrical"),
        note(
            dcf["x"] - 60.0,
            dcf["y"] - 190.0,
            "The DCF is sized to cancel the span exactly: "
            "$D_\\text{span}L_\\text{span} + D_\\text{DCF}L_\\text{DCF} = 0$",
            w=240.0,
        ),
        arrow(dcf["x"] + 40.0, dcf["y"] - 92.0, dcf["x"] + 58.0, dcf["y"] - 34.0),
    ]


def _coherent_sdfec(at: Layout) -> list[dict[str, Any]]:
    sd = at["sd"]
    return [
        frame(at, ["fec", "map", "drv", "tx", "mod"], "Transmitter", "optical"),
        frame(at, ["lo", "rx", "pm"], "Coherent front end", "electrical"),
        frame(at, ["smp", "sd", "dec", "cd"], "Soft decisions and decoding", "symbol"),
        note(
            sd["x"] - 20.0,
            sd["y"] + 100.0,
            "The demapper hands the decoder a confidence for every bit, not a bit: "
            "$L(b)=\\ln\\frac{P(b=0\\mid y)}{P(b=1\\mid y)}$",
            w=250.0,
        ),
        arrow(sd["x"] + 50.0, sd["y"] + 98.0, sd["x"] + 58.0, sd["y"] + NODE_H + 4.0),
    ]


_MARKS = {
    "ook_eye": _ook_eye,
    "wdm_osa": _wdm_osa,
    "dwdm_link": _dwdm_link,
    "coherent_sdfec": _coherent_sdfec,
}


NOTES: dict[str, dict[str, str]] = {
    "ook_eye": {
        "title": "Direct detection: reading an eye",
        "body": r"""
A 10 Gb/s pseudo-random pattern [[prbs]] drives a Mach–Zehnder modulator [[mzm]] that switches
the laser [[tx]] on and off. After 20 km of fibre [[fib]], a PIN photodiode [[pin]] and a
filter [[lpf]] turn the light back into a current. The eye [[eye]] folds that current over
two bit periods, and [[ber]] counts errors against the original pattern.

## The modulator

Driven push–pull, an MZM transmits

$$P_\text{out} = P_\text{in}\cos^2\left(\frac{\pi V}{2V_\pi}\right)$$

so a drive swinging between $0$ and $V_\pi$ takes the light from full to nearly dark. How dark
is set by the extinction ratio, $\mathrm{ER} = 10\log_{10}(P_1/P_0)$.

## The fibre

Power falls exponentially with length: $P(L) = P(0)\,10^{-\alpha L/10}$, with $\alpha$ in dB/km.
At 0.2 dB/km, 20 km costs 4 dB. Chromatic dispersion spreads each pulse by roughly

$$\Delta\tau \approx D\,L\,\Delta\lambda$$

With $D = 17$ ps/(nm·km) and the 0.08 nm a 10 Gb/s signal occupies, that is about 27 ps,
a quarter of the 100 ps bit slot.

## The photodiode and its noise

The photocurrent is $I = \mathcal{R}P$. Two noises ride on it: shot noise, which grows with the
current, and the thermal noise of the load resistor, which does not.

$$\sigma^2 = 2q\,(I + I_d)\,B + \frac{4k_BT}{R_L}\,B$$

## From the eye to a bit error rate

With the mean and spread of the one and zero rails read off the eye,

$$Q = \frac{I_1 - I_0}{\sigma_1 + \sigma_0}, \qquad \mathrm{BER} = \frac{1}{2}\,\mathrm{erfc}\left(\frac{Q}{\sqrt{2}}\right)$$

$Q = 6$ is the classic target: a BER of about $10^{-9}$.

## Try this

- Stretch [[fib]] to 60, 80 and 100 km. Does the eye lose height first, or width?
- Lower the power of [[tx]] until errors appear. That power is the receiver's sensitivity.
- Narrow [[lpf]] to 3 GHz. Less noise gets through, but neighbouring bits start to overlap.
""".strip(),
    },
    "wdm_osa": {
        "title": "WDM: four channels and an OSNR",
        "body": r"""
Four lasers [[ch0]]…[[ch3]] sit on the ITU 100 GHz grid, each modulated by its own MZM, and the
combiner [[mux]] puts them on one fibre. Two spans of 80 km each lose 16 dB, and two EDFAs
[[edfa0]] [[edfa1]] put it back. The analyser [[osa]] draws the spectrum and [[osnr]] reads the
signal-to-noise ratio off it.

## The grid

Channels are spaced evenly in *frequency*, not wavelength:

$$\nu_n = \nu_0 + n\,\Delta\nu, \qquad \Delta\lambda \approx \frac{\lambda^2}{c}\,\Delta\nu$$

At 1550 nm, 100 GHz is about 0.8 nm.

## Where the noise floor comes from

An amplifier with gain $G$ adds amplified spontaneous emission. In a reference bandwidth
$B_\text{ref}$, over both polarisations,

$$P_\text{ASE} = 2\,n_{sp}\,h\nu\,(G-1)\,B_\text{ref}$$

and at high gain the noise figure is $\mathrm{NF} \approx 2n_{sp}$.

## OSNR

$$\mathrm{OSNR} = \frac{P_\text{ch}}{P_\text{ASE}} \quad \text{in } B_\text{ref} = 12.5\ \text{GHz}\ (0.1\ \text{nm})$$

For a chain of $N$ identical spans, the rule of thumb at 1550 nm (a rule of thumb, not this
model) is

$$\mathrm{OSNR_{dB}} \approx 58 + P_\text{ch,dBm} - L_\text{span,dB} - \mathrm{NF_{dB}} - 10\log_{10}N$$

## Resolution bandwidth

An OSA reports power *per resolution bandwidth*. Widen it from $B_1$ to $B_2$ and the ASE floor
rises by $10\log_{10}(B_2/B_1)$ dB, while a channel narrower than the window reads the same.

## Try this

- Change the resolution bandwidth of [[osa]] from 12.5 to 50 GHz. How far does the floor move?
- Raise the noise figure of [[edfa0]] by 3 dB. Does the OSNR fall by 3 dB?
- Add a third span and amplifier. The rule of thumb predicts a loss of $10\log_{10}(3/2)$ dB.
""".strip(),
    },
    "dwdm_link": {
        "title": "Eight-channel DWDM link",
        "body": r"""
Eight 10 Gb/s channels on the 100 GHz grid are multiplexed [[mux]], boosted [[booster]], sent
through 80 km of fibre with the Kerr effect on [[span]], dispersion-compensated [[dcf]] and
amplified again [[preamp]]. One channel is dropped at [[demux]] and received on [[pin]].

## Dispersion compensation

The compensating fibre has a large negative $D$, and its length is chosen so the accumulated
dispersion cancels:

$$D_\text{span}L_\text{span} + D_\text{DCF}L_\text{DCF} = 0 \;\Rightarrow\; L_\text{DCF} = \frac{17 \times 80}{100} = 13.6\ \text{km}$$

## The Kerr effect

Light changes the refractive index it travels through. Over a span, a channel picks up a
phase proportional to its own power, and twice as much per watt from each neighbour:

$$\phi_j = \gamma\,L_\text{eff}\Big(P_j + 2\sum_{k \ne j} P_k\Big), \qquad L_\text{eff} = \frac{1 - e^{-\alpha L}}{\alpha}$$

With $\alpha$ in 1/km (0.2 dB/km is 0.046 /km), $L_\text{eff}$ of an 80 km span is about 21 km:
the nonlinearity happens near the launch, where the power is high.

## Noise against nonlinearity

More launch power means a better OSNR at the receiver, and more nonlinear phase in the span.
Somewhere between the two is the best launch power for the link.

## Try this

- Switch off cross-phase modulation on [[span]] and compare the bit error rate and run time.
- Step the gain of [[booster]] from 4 to 16 dB. Where is the error rate lowest?
- Shorten [[dcf]] by half and watch the dropped channel's eye at [[ber]].
""".strip(),
    },
    "coherent_sdfec": {
        "title": "Coherent 16-QAM with soft-decision FEC",
        "body": r"""
The encoder [[fec]] adds parity, [[map]] turns every 4 bits into one of 16 points, and [[drv]]
with [[mod]] write those points onto the field of [[tx]]. The coherent receiver [[rx]] beats the
signal against a strong local oscillator [[lo]], [[smp]] takes one sample per symbol, [[sd]]
turns each sample into soft bits and [[dec]] corrects them.

## Writing a point onto light

Biased at null, each arm of the IQ modulator has a sine-shaped field response, so

$$E_\text{out} \propto \sin\left(\frac{\pi V_I}{2V_\pi}\right) + j\,\sin\left(\frac{\pi V_Q}{2V_\pi}\right)$$

The driver pre-distorts with an arcsine so the points land on a square grid.

## Coherent detection

A 90° hybrid and balanced photodiodes return the real and imaginary parts of the beat:

$$i_I \propto \mathcal{R}\,\mathrm{Re}\{E_s E_\text{LO}^*\}, \qquad i_Q \propto \mathcal{R}\,\mathrm{Im}\{E_s E_\text{LO}^*\}$$

The current scales with $\sqrt{P_s P_\text{LO}}$, which is why a −25 dBm signal is readable:
the 10 dBm oscillator lifts it far above the thermal noise of the receiver.

## Quality of the constellation

$$\mathrm{EVM}_\text{rms} \approx \frac{1}{\sqrt{\mathrm{SNR}}}, \qquad \mathrm{BER}_\text{16QAM} \approx \frac{3}{8}\,\mathrm{erfc}\sqrt{\frac{\mathrm{SNR}}{10}}$$

The second assumes Gray coding and Gaussian noise.

## Soft decisions

Instead of a 0 or a 1, the demapper gives the decoder a log-likelihood ratio,

$$L(b) = \ln\frac{P(b=0\mid y)}{P(b=1\mid y)} \approx \frac{1}{2\sigma^2}\Big(\min_{s \in S_1}|y-s|^2 - \min_{s \in S_0}|y-s|^2\Big)$$

where $S_0$ and $S_1$ are the points whose bit $b$ is 0 or 1. A sample near a decision boundary
gives an LLR near zero, and the decoder knows to distrust that bit.

## Try this

- Lower the power of [[tx]] step by step. The pre-FEC error rate rises smoothly; the post-FEC one
  stays at zero until it falls off a cliff.
- Cut the iterations of [[dec]] from 8 to 2. Where does the cliff move?
""".strip(),
    },
    "coherent_16qam": {
        "title": "Coherent 16-QAM over 80 km",
        "body": r"""
This is the link the studio opens with: 32 GBd 16-QAM, 128 Gb/s raw, over 80 km of fibre with
real lasers. The transmitter and local oscillator [[tx]] [[lo]] each have a 100 kHz linewidth
and sit about 200 MHz apart, so the receiver has to find the carrier on its own. The bottom row
of the schematic is that search, in the order a real receiver performs it.

## 1. Undo the dispersion — [[cdc]]

The fibre is an all-pass filter whose phase grows with the square of frequency,

$$H_\text{fibre}(\omega) = \exp\left(j\,\frac{\beta_2 L}{2}\,\omega^2\right), \qquad \beta_2 = -\frac{D\lambda^2}{2\pi c}$$

and the compensator applies its conjugate. $17$ ps/(nm·km) over 80 km accumulates 1360 ps/nm.

## 2. Find the sampling instant — [[tr]], [[smp]]

Timing recovery estimates where in each symbol to sample, and the sampler applies the
root-raised-cosine matched filter (roll-off 0.2) and keeps one sample per symbol.

## 3. Remove the frequency offset — [[fo]]

Raising a QPSK-like constellation to the fourth power strips the data and leaves the carrier:

$$\Delta\hat f = \frac{1}{8\pi T}\,\arg\sum_k \big(y_k\,y_{k-1}^*\big)^4$$

## 4. Track the phase — [[cr]], [[pqr]]

Laser phase noise is a random walk whose step variance per symbol is $2\pi(\Delta\nu_\text{tx} + \Delta\nu_\text{lo})\,T$. Blind phase search
tries [[cr]]'s test phases and keeps the one that lands samples closest to the grid; known pilot
symbols, one every 64, settle which of the four quarter turns is the right one.

## 5. Decide, softly — [[sd]], [[fdec]]

The demapper turns each symbol into four log-likelihood ratios and the decoder corrects them.
[[vsa]] measures EVM, SNR and BER, and [[cd]] draws the constellation in the dock below.

## Try this

- Set the linewidth of both lasers to 1000 kHz. What happens to the constellation?
- Move [[lo]] further from 1550 nm. Where does frequency recovery give up?
- Change the format to 64-QAM (bits per symbol 6) on [[map]]: every block that cares follows.
""".strip(),
    },
}
