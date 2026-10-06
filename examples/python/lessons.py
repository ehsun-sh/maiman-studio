# ruff: noqa: E501, RUF001 -- prose and formulas, with the dashes and minus signs they need
"""The teaching material the shipped projects carry: a lesson, and marks on the canvas.

Every project the studio opens from the Examples menu is written by a script in
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


# ---------------------------------------------------------------------------
# Phase 1: first light


def _laser_meters(at: Layout) -> list[dict[str, Any]]:
    att = at["att"]
    return [
        frame(at, ["laser"], "Source", "optical"),
        frame(at, ["pm_tx", "osa", "pm_rx"], "Instruments", "metric"),
        note(
            att["x"] - 20.0,
            att["y"] + 100.0,
            "Halving the power is 3 dB: $10\\log_{10}(1/2) = -3.01$ dB. In dBm, losses subtract.",
            w=230.0,
        ),
        arrow(att["x"] + 50.0, att["y"] + 98.0, att["x"] + 58.0, att["y"] + NODE_H + 4.0),
    ]


def _loss_budget(at: Layout) -> list[dict[str, Any]]:
    span = at["span"]
    return [
        frame(at, ["laser"], "Transmitter", "optical"),
        frame(at, ["patch_tx", "span", "patch_rx"], "Everything between", "binary"),
        frame(at, ["pm_rx"], "Receiver", "metric"),
        note(
            span["x"] - 40.0,
            span["y"] + 110.0,
            "$P_\\text{rx} = P_\\text{tx} - \\alpha L - \\sum L_\\text{conn}$, "
            "all in dB. 0 − 10 − 1 = −11 dBm.",
            w=240.0,
        ),
        arrow(span["x"] + 50.0, span["y"] + 108.0, span["x"] + 58.0, span["y"] + NODE_H + 4.0),
    ]


def _mzm_curve(at: Layout) -> list[dict[str, Any]]:
    mzm = at["mzm"]
    return [
        frame(at, ["laser", "bias"], "Light and a voltage", "optical"),
        frame(at, ["mzm"], "Modulator", "electrical"),
        frame(at, ["pm_out"], "Meter", "metric"),
        note(
            mzm["x"] - 10.0,
            mzm["y"] + 120.0,
            "$P_\\text{out} = P_\\text{in}\\cos^2(\\pi V/2V_\\pi)$. "
            "Sweep the voltage of bias from 0 to 8 V to draw it.",
            w=240.0,
        ),
        arrow(mzm["x"] + 50.0, mzm["y"] + 118.0, mzm["x"] + 58.0, mzm["y"] + NODE_H + 4.0),
    ]


def _pulse_spreading(at: Layout) -> list[dict[str, Any]]:
    scope = at["after_10km"]
    return [
        frame(at, ["pulse"], "10 ps pulse", "optical"),
        frame(at, ["fibre_a", "fibre_b"], "2 × 5 km, lossless", "binary"),
        frame(at, ["launch", "after_5km", "after_10km"], "Scopes", "metric"),
        side_note(
            scope,
            "Run, then open the Scope tab: three pulses, one energy. "
            "$T_1/T_0 = \\sqrt{1 + (z/L_D)^2}$",
        ),
        side_arrow(scope),
    ]


def _chirp_compression(at: Layout) -> list[dict[str, Any]]:
    up = at["up_out"]
    return [
        frame(at, ["up", "down"], "Opposite chirps", "optical"),
        frame(at, ["fibre_up", "fibre_down"], "Same fibre", "binary"),
        frame(at, ["up_out", "launch", "down_out"], "Scopes", "metric"),
        side_note(
            up,
            "$C\\beta_2 < 0$: the chirp and the fibre undo each other and the pulse "
            "narrows. Tick *Show instantaneous frequency* in the Scope tab to see why.",
            h=112.0,
        ),
        side_arrow(up),
    ]


def _soliton(at: Layout) -> list[dict[str, Any]]:
    kerr = at["with_kerr"]
    return [
        frame(at, ["soliton"], "sech pulse, N = 1", "optical"),
        frame(at, ["kerr_on", "kerr_off"], "Kerr on and off", "binary"),
        frame(at, ["with_kerr", "launch", "dispersion_only"], "Scopes", "metric"),
        side_note(
            kerr,
            "$N^2 = \\gamma P_0 T_0^2/|\\beta_2| = 1$: self-phase modulation cancels "
            "the dispersion's chirp exactly.",
        ),
        side_arrow(kerr),
    ]


def _splitters(at: Layout) -> list[dict[str, Any]]:
    total = at["pm_sum"]
    return [
        frame(at, ["laser_a", "laser_b"], "Two colours", "optical"),
        frame(at, ["split4", "tap", "combiner"], "Dividing and joining", "binary"),
        frame(at, ["pm_quarter", "pm_sum", "pm_tap"], "Meters", "metric"),
        side_note(
            total,
            "Powers add in milliwatts, not in dBm: 0.25 + 0.90 = 1.15 mW = +0.61 dBm.",
        ),
        side_arrow(total),
    ]


def side_note(block: dict[str, float], text: str, h: float = 96.0) -> dict[str, Any]:
    """A note to the right of ``block``, clear of the frame around it."""
    return note(block["x"] + NODE_W + 50.0, block["y"] - 20.0, text, w=240.0, h=h)


def side_arrow(block: dict[str, float]) -> dict[str, Any]:
    """From a :func:`side_note` back to the block it is about."""
    y = block["y"] + NODE_H / 2.0
    return arrow(block["x"] + NODE_W + 48.0, y, block["x"] + NODE_W + 14.0, y)


def below_note(block: dict[str, float], text: str, w: float = 240.0) -> dict[str, Any]:
    """A note under ``block``, clear of the frame around it."""
    return note(block["x"] - 40.0, block["y"] + NODE_H + 60.0, text, w=w)


def below_arrow(block: dict[str, float]) -> dict[str, Any]:
    """From a :func:`below_note` up to the block it is about."""
    x = block["x"] + NODE_W / 2.0
    return arrow(x, block["y"] + NODE_H + 58.0, x, block["y"] + NODE_H + 14.0)


# ---------------------------------------------------------------------------
# Phase 2: direct-detection links


def _receiver_sensitivity(at: Layout) -> list[dict[str, Any]]:
    apd = at["apd"]
    return [
        frame(at, ["prbs", "drv", "tx", "mzm"], "10 Gb/s transmitter", "optical"),
        frame(at, ["att", "split"], "Variable loss", "binary"),
        frame(at, ["pin", "lpf_pin", "ber_pin", "eye_pin"], "PIN receiver", "electrical"),
        frame(at, ["apd", "lpf_apd", "ber_apd", "eye_apd"], "APD receiver", "electrical"),
        below_note(
            apd,
            "Gain $M$ lifts the signal over the thermal noise; it also multiplies shot "
            "noise by $M^2F(M)$. Sweep the gain: there is a best one.",
        ),
        below_arrow(apd),
    ]


def _dml_reach(at: Layout) -> list[dict[str, Any]]:
    scope = at["chirp"]
    return [
        frame(at, ["drv_dml", "dml"], "Directly modulated laser", "optical"),
        frame(at, ["cw", "drv_ext", "mzm"], "CW laser + Mach–Zehnder", "optical"),
        frame(at, ["fibre_dml", "fibre_ext"], "20 km each", "binary"),
        frame(
            at,
            ["pin_dml", "eye_dml", "pin_ext", "eye_ext"],
            "Identical receivers",
            "electrical",
        ),
        side_note(
            scope,
            "Run, open the Scope tab and tick *Show instantaneous frequency*: "
            "every edge chirps by gigahertz.",
        ),
        side_arrow(scope),
    ]


def _mode_partition(at: Layout) -> list[dict[str, Any]]:
    fp = at["eye_fp"]
    return [
        frame(at, ["tx_dfb", "tx_fp_steady", "tx_fp"], "Three lasers", "optical"),
        frame(
            at,
            ["mzm_dfb", "mzm_fp", "fibre_dfb", "fibre_fp"],
            "Same modulation, 3 km each",
            "binary",
        ),
        frame(
            at,
            ["pin_dfb", "pin_fp", "eye_dfb", "eye_fp"],
            "Identical receivers",
            "electrical",
        ),
        side_note(
            fp,
            "Same laser as the row above, with its modes trading power. "
            "Raise its power by 10 dB: the Q hardly moves.",
        ),
        side_arrow(fp),
    ]


def _pam4_lane(at: Layout) -> list[dict[str, Any]]:
    eq = at["eq"]
    return [
        frame(at, ["prbs", "pam4", "tx", "mzm"], "PAM4 transmitter", "optical"),
        frame(at, ["pin", "lpf"], "7 GHz receiver", "electrical"),
        frame(at, ["eye", "eq_off", "eq"], "Eye and equalisers", "metric"),
        side_note(
            eq,
            "9 feed-forward taps and 2 of decision feedback undo what the 7 GHz filter smeared.",
        ),
        side_arrow(eq),
    ]


def _cwdm4(at: Layout) -> list[dict[str, Any]]:
    fibre = at["fibre"]
    return [
        frame(
            at,
            ["tx0", "tx3", "mzm0", "mzm3", "prbs", "drv"],
            "4 × 25G lanes, 1331 to 1271 nm",
            "optical",
        ),
        frame(at, ["mux", "fibre", "demux"], "One fibre, no amplifier", "binary"),
        frame(
            at,
            ["pin0", "pin3", "eye0", "eye3"],
            "4 receivers",
            "electrical",
        ),
        below_note(
            fibre,
            "Near 1310 nm $D \\approx S_0(\\lambda - \\lambda_0)$: from +1.9 to −3.6 "
            "ps/(nm·km) across the four lanes.",
        ),
        below_arrow(fibre),
    ]


def _gpon(at: Layout) -> list[dict[str, Any]]:
    split = at["split32"]
    return [
        frame(at, ["prbs", "drv", "olt", "mzm"], "OLT, at the exchange", "optical"),
        frame(at, ["feeder", "split32"], "Outside plant", "binary"),
        frame(at, ["pm_onu", "apd", "lpf", "ber", "eye"], "One ONU, at a home", "electrical"),
        below_note(
            split,
            "1:32 costs $10\\log_{10}32 = 15.05$ dB before any excess loss: more than "
            "the 20 km of fibre.",
        ),
        below_arrow(split),
    ]


# ---------------------------------------------------------------------------
# Phase 3: amplified and WDM links


def _edfa_basics(at: Layout) -> list[dict[str, Any]]:
    edfa = at["edfa"]
    return [
        frame(at, ["pm_in", "osa_in"], "Before", "metric"),
        frame(at, ["pm_out", "osnr", "osa_out"], "After", "metric"),
        below_note(
            edfa,
            "20 dB of gain, and a floor of ASE 33 dB under the signal in every 0.1 nm: "
            "$58 + P_\\text{in} - NF$.",
        ),
        below_arrow(edfa),
    ]


def _amplified_chain(at: Layout) -> list[dict[str, Any]]:
    return [
        frame(at, ["span1", "edfa4"], "Spans 1 to 4", "binary"),
        frame(at, ["span5", "edfa8"], "Spans 5 to 8", "binary"),
        side_note(
            at["osa8"],
            "Every amplifier adds the same noise; the signal stays the same. "
            "Twice the spans, half the OSNR: 3 dB.",
        ),
        side_arrow(at["osa8"]),
    ]


def _launch_power(at: Layout) -> list[dict[str, Any]]:
    booster = at["booster"]
    return [
        frame(at, ["prbs", "drv", "tx", "mzm"], "10 Gb/s transmitter", "optical"),
        frame(at, ["span1", "edfa3"], "Spans 1 to 3, each compensated", "binary"),
        frame(at, ["span4", "edfa5"], "Spans 4 and 5", "binary"),
        frame(at, ["pin", "lpf", "ber", "eye"], "Receiver", "electrical"),
        below_note(
            booster,
            "Sweep this gain from 0 to 15 dB. Too little and noise wins; too much and "
            "the Kerr effect does. The best launch is near +3 dBm.",
        ),
        below_arrow(booster),
    ]


def _fwm_dsf(at: Layout) -> list[dict[str, Any]]:
    return [
        frame(at, ["ch0", "ch1", "ch2", "ch3"], "4 × CW, 100 GHz apart, +6 dBm", "optical"),
        frame(at, ["fibre_dsf", "osa_dsf"], "G.653: D = 0 at 1550 nm", "binary"),
        frame(at, ["fibre_nzdsf", "osa_nzdsf"], "G.655: D = 4 ps/(nm·km)", "binary"),
        side_note(
            at["osa_dsf"],
            "New lines at $f_i + f_j - f_k$: on a uniform grid they also land on the "
            "channels, where no filter can take them out.",
        ),
        side_arrow(at["osa_dsf"]),
    ]


def _roadm(at: Layout) -> list[dict[str, Any]]:
    wss = at["wss"]
    return [
        frame(at, ["prbs", "drv", "tx0", "tx3", "mzm0", "mzm3"], "4 channels", "optical"),
        frame(at, ["span_a", "edfa_a"], "Span in", "binary"),
        frame(at, ["pin_drop", "lpf_drop", "ber_drop", "eye_drop"], "Drop", "electrical"),
        frame(at, ["prbs_add", "drv_add", "tx_add", "mzm_add"], "Add, new data", "optical"),
        frame(at, ["span_b", "edfa_b", "demux_b"], "Span out", "binary"),
        frame(at, ["pin_add", "lpf_add", "ber_add", "eye_add"], "Added channel", "electrical"),
        note(
            wss["x"] - 40.0,
            wss["y"] - 150.0,
            "Express three, drop one, add one in its slot. What leaks of the dropped "
            "channel sits on top of the new one.",
            w=240.0,
        ),
        arrow(wss["x"] + 58.0, wss["y"] - 52.0, wss["x"] + 58.0, wss["y"] - 12.0),
    ]


_MARKS = {
    "ook_eye": _ook_eye,
    "wdm_osa": _wdm_osa,
    "dwdm_link": _dwdm_link,
    "coherent_sdfec": _coherent_sdfec,
    "laser_meters": _laser_meters,
    "loss_budget": _loss_budget,
    "mzm_curve": _mzm_curve,
    "pulse_spreading": _pulse_spreading,
    "chirp_compression": _chirp_compression,
    "soliton": _soliton,
    "splitters": _splitters,
    "receiver_sensitivity": _receiver_sensitivity,
    "dml_reach": _dml_reach,
    "mode_partition": _mode_partition,
    "pam4_lane": _pam4_lane,
    "cwdm4": _cwdm4,
    "gpon": _gpon,
    "edfa_basics": _edfa_basics,
    "amplified_chain": _amplified_chain,
    "launch_power": _launch_power,
    "fwm_dsf": _fwm_dsf,
    "roadm": _roadm,
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
    "laser_meters": {
        "title": "1.1 A laser, a power meter and a spectrum",
        "body": r"""
A laser [[laser]] puts out one milliwatt at 1550 nm. Three instruments read it: a power meter
[[pm_tx]], an optical spectrum analyser [[osa]], and a second meter [[pm_rx]] behind a 3 dB
attenuator [[att]].

## dBm

Optical power is quoted against one milliwatt, on a log scale:

$$P_\text{dBm} = 10\log_{10}\frac{P}{1\ \text{mW}}, \qquad P = 1\ \text{mW}\times10^{P_\text{dBm}/10}$$

So 0 dBm is 1 mW, +10 dBm is 10 mW and −30 dBm is 1 µW. A loss in dB *subtracts*: 3 dB halves
the power, which is why [[pm_rx]] reads −3.00 dBm, 0.501 mW.

## What the analyser draws

An OSA sweeps a narrow filter across the band and reports the power that falls in it. The
filter's width is the **resolution bandwidth** (RBW), 12.5 GHz here (0.1 nm at 1550 nm).

This laser's linewidth is 100 kHz, five orders of magnitude narrower than the filter. So the line
on screen is the *filter's* shape, not the laser's, and its height is the laser's whole power:
0 dBm. Converting a width in frequency to one in wavelength:

$$\Delta\lambda \approx \frac{\lambda^2}{c}\,\Delta\nu \quad\Rightarrow\quad 12.5\ \text{GHz} \approx 0.1\ \text{nm}$$

## Try this

- Set [[laser]] to +3 dBm. What do the two meters read, in mW?
- Widen the resolution bandwidth of [[osa]] to 50 GHz. The line gets wider; does it get taller?
- Change [[att]] to 10 dB, then 20 dB. Each 10 dB is a factor of ten.
""".strip(),
    },
    "loss_budget": {
        "title": "1.2 A loss budget",
        "body": r"""
The first sum anyone designing a link does. A 0 dBm laser [[laser]] goes through a patch
connector [[patch_tx]], 50 km of fibre [[span]] and another connector [[patch_rx]] to a
meter [[pm_rx]].

## Loss in fibre

Power falls exponentially with length, which is a straight line in decibels:

$$P(L) = P(0)\,10^{-\alpha L/10} \quad\Leftrightarrow\quad P_\text{dBm}(L) = P_\text{dBm}(0) - \alpha L$$

with $\alpha$ in dB/km: 0.2 dB/km is typical of standard fibre at 1550 nm, so 50 km costs 10 dB.

## The budget

Because everything is in dB, the budget is an addition:

$$P_\text{rx} = P_\text{tx} - \alpha L - \sum L_\text{conn} = 0 - 10 - 2\times0.5 = -11\ \text{dBm}$$

The **margin** is what is left over the receiver's sensitivity. A receiver that needs
−20 dBm leaves 9 dB here, and a designer would keep 3 dB of it for ageing and repairs.

## Sweep it

Open the **Sweep** button, choose [[span]] and its length, and run 0 to 100 km. The meter
falls 5 dB every 25 km, from −1 dBm (the connectors alone) to −21 dBm:

| Length | 0 km | 25 km | 50 km | 75 km | 100 km |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Received | −1 dBm | −6 dBm | −11 dBm | −16 dBm | −21 dBm |

## Try this

- At 1310 nm fibre loses about 0.35 dB/km. Change [[span]]'s attenuation and find the length
  that leaves −20 dBm.
- Add a 1:8 splitter: how much shorter must the fibre be for the same received power?
""".strip(),
    },
    "mzm_curve": {
        "title": "1.3 The Mach–Zehnder transfer curve",
        "body": r"""
A Mach–Zehnder modulator [[mzm]] splits the light of [[laser]] into two arms, shifts the phase
of one against the other with a voltage, and recombines them. A DC source [[bias]] sets that
voltage and [[pm_out]] measures what comes out.

## The cosine

The two arms interfere. With a phase difference $\Delta\phi = \pi V/V_\pi$,

$$P_\text{out} = P_\text{in}\cos^2\left(\frac{\Delta\phi}{2}\right) = P_\text{in}\cos^2\left(\frac{\pi V}{2V_\pi}\right)$$

$V_\pi$ is the voltage that takes the output from full to dark: 4 V here.

## Bias points

| Drive | Name | Output |
| :-- | :-- | :-- |
| $0$ | peak | 0 dBm |
| $V_\pi/2 = 2$ V | quadrature | −3 dBm |
| $V_\pi = 4$ V | null | −30 dBm |

Quadrature is where the curve is steepest and most nearly straight, so an analogue or a small
digital signal is biased there. An on–off keyed signal swings from peak to null instead.

## Extinction ratio

A real modulator never goes fully dark: its arms do not split exactly 50/50. The **extinction
ratio** says how dark,

$$\mathrm{ER} = 10\log_{10}\frac{P_\text{max}}{P_\text{min}}$$

30 dB on [[mzm]], which is why the null reads −30 dBm and not $-\infty$.

## Sweep it

Open **Sweep**, pick [[bias]] and its voltage, and run 0 to 8 V in 0.25 V steps. The curve
falls to the null at 4 V and comes back at 8 V: the cosine is periodic in $2V_\pi$.

## Try this

- Lower the extinction ratio of [[mzm]] to 15 dB. Which bias points move?
- Set $V_\pi$ to 2 V and find the new quadrature point.
""".strip(),
    },
    "pulse_spreading": {
        "title": "1.4 A pulse spreading in fibre",
        "body": r"""
A Gaussian pulse [[pulse]] with $T_0 = 10$ ps goes through two 5 km pieces of standard fibre
[[fibre_a]] [[fibre_b]]. Three oscilloscopes, [[launch]], [[after_5km]] and [[after_10km]],
draw it on one time axis in the **Scope** tab. Loss is set to zero so that dispersion is the
only thing happening.

## Why it spreads

A short pulse is made of many frequencies, and in fibre each travels at its own speed. That is
group-velocity dispersion, written as $D$ in ps/(nm·km) on a datasheet and $\beta_2$ in the
equations:

$$\beta_2 = -\frac{D\,\lambda^2}{2\pi c} \approx -21.7\ \text{ps}^2/\text{km} \quad (D = 17,\ \lambda = 1550\ \text{nm})$$

## How fast

The distance over which dispersion matters is the **dispersion length**,

$$L_D = \frac{T_0^2}{|\beta_2|} = \frac{(10\ \text{ps})^2}{21.7\ \text{ps}^2/\text{km}} = 4.6\ \text{km}$$

and an unchirped Gaussian widens as

$$\frac{T_1}{T_0} = \sqrt{1 + \left(\frac{z}{L_D}\right)^2}$$

| Scope | $z$ | $T_1/T_0$ | FWHM ($1.665\,T$) |
| :-- | :-- | :-- | :-- |
| [[launch]] | 0 | 1 | 16.65 ps |
| [[after_5km]] | 5 km | 1.48 | 24.6 ps |
| [[after_10km]] | 10 km | 2.39 | 39.8 ps |

The scopes' FWHM readings are these numbers. The area under each trace is the same energy, so as
the pulse widens its peak falls by the same factor.

## Try this

- Halve the width of [[pulse]] to 5 ps. $L_D$ falls four times: how wide is it after 10 km?
- Set [[fibre_b]]'s dispersion to −17. What does [[after_10km]] show, and why?
- Tick *Show instantaneous frequency* in the Scope tab: dispersion leaves the pulse chirped. In standard fibre blue
  travels faster, so the leading edge is bluer than the tail.
""".strip(),
    },
    "chirp_compression": {
        "title": "1.5 Chirped pulse compression",
        "body": r"""
Two Gaussian pulses with the same $T_0 = 10$ ps but opposite chirp, [[up]] ($C = +2$) and
[[down]] ($C = -2$), go through identical 1.84 km fibres [[fibre_up]] [[fibre_down]]. The
**Scope** tab overlays the launched pulse [[launch]] and both outputs [[up_out]] [[down_out]].

## Chirp

A chirped pulse has a frequency that changes across it. With

$$A(0, T) = \sqrt{P_0}\,\exp\left(-\frac{1 - jC}{2}\,\frac{T^2}{T_0^2}\right)$$

the instantaneous frequency is $\delta f = C\,T/(2\pi T_0^2)$: for $C > 0$ the tail is bluer than
the head (an up-chirp). Tick *Show instantaneous frequency* in the
Scope tab to see that straight line.

## The sign matters

In standard fibre $\beta_2 < 0$, and blue light travels *faster*. An up-chirped pulse has its
blue at the back, which catches up with the red at the front: the pulse first narrows. A
down-chirped pulse is already arranged the wrong way and spreads faster than an unchirped one.

$$\frac{T_1}{T_0} = \sqrt{\left(1 + \frac{C\beta_2 z}{T_0^2}\right)^2 + \left(\frac{\beta_2 z}{T_0^2}\right)^2}$$

The narrowest point is at $z = L_D\,C/(1 + C^2) = 0.4\,L_D = 1.84$ km, where the width is
$T_0/\sqrt{1+C^2}$:

| Scope | $C$ | $T_1/T_0$ | FWHM |
| :-- | :-- | :-- | :-- |
| [[launch]] | — | 1 | 16.65 ps |
| [[up_out]] | +2 | 0.447 | 7.45 ps |
| [[down_out]] | −2 | 1.84 | 30.7 ps |

This is how a directly modulated laser's chirp limits its reach, and how pulse compressors and
chirped-pulse amplifiers work.

## Try this

- Make [[fibre_up]] twice as long. The pulse passes its narrowest point and spreads again.
- Set the chirp of [[up]] to 0. Now both outputs spread the same way.
""".strip(),
    },
    "soliton": {
        "title": "1.6 The fundamental soliton",
        "body": r"""
A hyperbolic-secant pulse [[soliton]] with $T_0 = 10$ ps is launched at exactly the power that
makes it a soliton, into two identical 14.5 km fibres: [[kerr_on]] with the Kerr effect and
[[kerr_off]] without it. The **Scope** tab overlays the launch [[launch]] and the two outputs
[[with_kerr]] [[dispersion_only]].

## Two chirps that cancel

Dispersion alone spreads the pulse and chirps it (in anomalous fibre, blue ahead). The Kerr
effect makes the refractive index depend on intensity, $n = n_0 + n_2 I$, so the pulse imposes
a phase on itself,

$$\phi_\text{NL}(T) = \gamma\,|A(T)|^2\,z$$

which chirps it the *other* way. At the right peak power the two chirps cancel at every point of
the pulse and it travels unchanged.

## The soliton condition

$$N^2 = \frac{\gamma P_0 T_0^2}{|\beta_2|} = 1 \quad\Rightarrow\quad P_0 = \frac{|\beta_2|}{\gamma T_0^2} = \frac{21.7\ \text{ps}^2/\text{km}}{1.3\ \text{W}^{-1}\text{km}^{-1}\times(10\ \text{ps})^2} = 167\ \text{mW}$$

The fibre is two soliton periods long, $z_0 = \frac{\pi}{2}L_D = 7.24$ km each.

| Scope | FWHM | Peak |
| :-- | :-- | :-- |
| [[launch]] | 17.6 ps ($1.763\,T_0$) | 167 mW |
| [[with_kerr]] | 17.6 ps | 167 mW |
| [[dispersion_only]] | 49.5 ps | 65 mW |

## Try this

- Double the peak power of [[soliton]] (+3 dB). Now $N = \sqrt2$: the pulse narrows and breathes.
- Quadruple it (+6 dB) for $N = 2$, and set [[kerr_on]] to 3.62 km ($z_0/2$). A second-order
  soliton splits and re-forms once a period.
- Put 0.2 dB/km of loss on [[kerr_on]]. As the power falls, the soliton slowly widens.
""".strip(),
    },
    "splitters": {
        "title": "1.7 Splitters, couplers and a combiner",
        "body": r"""
Three ways to divide or join light. A 1:4 splitter [[split4]] shares [[laser_a]] four ways.
A directional coupler [[tap]] takes 10 % of [[laser_b]] off to a monitor. A combiner
[[combiner]] puts one quarter of the first laser and the through port of the coupler onto one
fibre, and [[pm_sum]] measures them together. The spectrum [[osa]] keeps apart what the meter
adds: two lines, 1 nm apart, each at its own power.

## Splitting costs 3 dB per halving

An ideal $1{:}N$ splitter gives each output $1/N$ of the power:

$$L_\text{split} = 10\log_{10}N \quad\Rightarrow\quad 1{:}4 = 6.02\ \text{dB}, \quad 1{:}32 = 15.05\ \text{dB}$$

[[pm_quarter]] reads −6.02 dBm, 0.25 mW. A passive optical network's 1:32 split costs more than
70 km of fibre.

## A coupler conserves power

A coupler with power coupling $\kappa$ sends $\kappa$ across and keeps $1-\kappa$ on the straight
path:

$$P_\text{cross} = \kappa P_\text{in}, \qquad P_\text{through} = (1-\kappa)\,P_\text{in}$$

With $\kappa = 0.1$, the monitor [[pm_tap]] gets −10 dBm and the through port keeps 0.9 mW
(−0.46 dBm). The two add back to the input: nothing is lost in an ideal coupler.

## Powers add in milliwatts

The combiner joins two *different* wavelengths, 1550 and 1551 nm, so their powers simply add:

$$P_\text{sum} = 0.25 + 0.90 = 1.15\ \text{mW} = +0.61\ \text{dBm}$$

Adding the dBm values, −6.02 + (−0.46), would be meaningless. Two beams at the *same* wavelength
would interfere instead, which is why [[combiner]] refuses them: that needs a coupler.

## Try this

- Change [[split4]]'s excess loss to 0.5 dB, as a real splitter has.
- Set the coupling of [[tap]] to 0.5 for a 3 dB coupler. What does [[pm_sum]] read?
- Set [[laser_b]] to 1550 nm and run: read the combiner's message in the Log tab.
""".strip(),
    },
    "receiver_sensitivity": {
        "title": "2.2 Receiver sensitivity: PIN against APD",
        "body": r"""
A 10 Gb/s on–off keyed signal [[tx]] [[mzm]] goes through a variable attenuator [[att]] and is
split three ways [[split]]: one copy to a power meter [[pm_rx]], one to a PIN photodiode
[[pin]], and one to an avalanche photodiode [[apd]]. Each receiver has the same 7 GHz filter
and an error counter, so the only difference between them is the detector. Their eyes,
[[eye_pin]] and [[eye_apd]], are in the Eye tab: raise [[att]] and watch the PIN's close first.

## What limits a PIN receiver

The photocurrent is $I = \mathcal{R}P$. At low power the dominant noise is the thermal noise of
the load resistor, which does not depend on the signal at all:

$$\sigma_T^2 = \frac{4k_BT}{R_L}\,B$$

So $Q$ falls in proportion to the received power: halve the power and $Q$ halves.

## What an APD changes

An avalanche photodiode multiplies the photocurrent by $M$ before the load sees it, so the signal
grows $M$ times while thermal noise stays put. The multiplication is random, which costs an
excess noise factor on the shot noise:

$$\sigma_s^2 = 2q\,M^2F(M)\,\mathcal{R}P\,B, \qquad F(M) = kM + \left(2 - \frac{1}{M}\right)(1 - k)$$

$k$ is the ionisation ratio, 0.5 on [[apd]] (InGaAs).

## Sensitivity

The **sensitivity** is the received power for $Q = 6$, a bit error rate of $10^{-9}$.
Open **Sweep**, choose [[att]] and its attenuation, and run 4 to 30 dB:

| Receiver | Sensitivity at $Q = 6$ |
| :-- | :-- |
| PIN [[pin]] | −19.2 dBm |
| APD [[apd]], $M = 10$ | −28.1 dBm |

The APD buys almost 9 dB, which is 45 km of fibre at 0.2 dB/km.

## The optimum gain

More gain is not always better: signal grows as $M$, shot noise as $M^2F(M) \approx kM^3$. At
[[att]] = 20 dB (−27.8 dBm) the APD's $Q$ is 0.8 at $M = 1$, 6.4 at 10, peaks at 7.3 near
$M = 20$ and falls back to 6.3 at 40.

## Try this

- Sweep the gain of [[apd]] from 1 to 40 at [[att]] = 20 dB, and find the peak.
- Set its ionisation ratio to 0.02 (silicon). Where does the best gain move?
- Narrow [[lpf_pin]] to 5 GHz. Thermal noise falls with $B$; does the eye stay open?
""".strip(),
    },
    "dml_reach": {
        "title": "2.3 The reach of a directly modulated laser",
        "body": r"""
The same 10 Gb/s pattern [[prbs]] leaves two transmitters. On top, the drive current of a
directly modulated laser [[dml]] is switched. Below, a CW laser [[cw]] is switched by a
Mach–Zehnder [[mzm]]. The two are matched: the same average power and the same extinction ratio
(4.84 dB), so the only difference is how the light was modulated. Each goes through 20 km of
standard fibre into an identical receiver. Compare the two eyes, [[eye_dml]] and
[[eye_ext]], in the Eye tab.

## Why a DML chirps

The light comes out when the carrier density in the laser moves, and the refractive index moves
with the carriers. So every change in power is also a change in frequency. The rate equations
give it as

$$\Delta\nu(t) = \frac{\alpha}{4\pi}\left(\frac{d}{dt}\ln P(t) + \kappa\,P(t)\right)$$

The first term is the **transient** chirp, at every edge. The second is the **adiabatic** chirp:
a one sits at a different frequency from a zero. $\alpha$ is the linewidth enhancement factor,
4 on [[dml]]. The oscilloscope [[chirp]] shows it: the frequency swings by about 21 GHz on the
rising edges.

## What dispersion does with it

Each part of a bit now travels at its own speed. A frequency excursion $\Delta\nu$ is a
wavelength excursion $\Delta\lambda = \lambda^2\Delta\nu/c$, which dispersion turns into a delay:

$$\Delta\tau = D\,L\,\Delta\lambda \approx 17 \times 20 \times 0.17\ \text{nm} \approx 58\ \text{ps}$$

That is more than half a 100 ps bit slot.

| Transmitter | $Q$ after 20 km |
| :-- | :-- |
| [[dml]] | 2.7 |
| [[cw]] + [[mzm]] | 39.8 |

A Mach–Zehnder driven push–pull changes only the amplitude of the field, so it does not chirp.
That is why long-reach links use external modulation, and why 10G DMLs are sold for 10 to
20 km.

## Try this

- Shorten [[fibre_dml]] to 0, 5 and 10 km. The eye at 5 km is better than back to back: the
  fibre first undoes some of the laser's ringing.
- Raise [[dml]]'s bias to 120 mA. It rings faster; does the $Q$ at 20 km improve?
- Set both fibres' dispersion to 0. What is left of the gap is the laser's ringing, not its
  chirp.
""".strip(),
    },
    "mode_partition": {
        "title": "2.4 Laser noise: mode partition in a Fabry–Perot laser",
        "body": r"""
One 2.5 Gb/s pattern modulates three lasers at 1550 nm, each through 3 km of standard fibre
into the same receiver:

- [[tx_dfb]], a single-frequency DFB laser;
- [[tx_fp_steady]], a seven-mode Fabry–Perot laser with its noise switched off (its modes sit at
  their steady powers);
- [[tx_fp]], the same Fabry–Perot laser with its noise on.

Each receiver has its eye: [[eye_fp]] is the one whose rails thicken.

## Many modes, one reservoir

A Fabry–Perot cavity supports a comb of longitudinal modes, 1.1 nm apart here, and the gain
keeps several of them lasing. They all draw on the same carriers, so when one mode gains power
another loses it. The *sum* is quiet. Each mode on its own is not.

## Dispersion splits them

Each mode travels at its own group velocity. Seven modes span $6 \times 1.1 = 6.6$ nm, so after
3 km at 17 ps/(nm·km) they arrive

$$\Delta\tau = D\,L\,\Delta\lambda = 17 \times 3 \times 6.6 \approx 340\ \text{ps}$$

apart, most of a 400 ps bit. That alone smears the eye, and the steady laser [[tx_fp_steady]]
shows how much.

## Partition noise

Once the modes no longer arrive together, the power they were trading no longer cancels at the
detector. What is left is noise proportional to the signal itself, so turning up the power
cannot beat it. The textbook estimate (Ogawa; a rule of thumb, not this model) is a relative
noise of

$$r_\text{mpn} = \frac{k}{\sqrt2}\left[1 - e^{-(\pi B D L \sigma_\lambda)^2}\right]$$

with $k$ the partition coefficient and $\sigma_\lambda$ the laser's rms spectral width. It sets a
ceiling on $Q$ of about $1/r_\text{mpn}$, whatever the received power.

| Laser | $Q$ after 3 km |
| :-- | :-- |
| [[tx_dfb]] | 104 |
| [[tx_fp_steady]] | 28.6 |
| [[tx_fp]] | 13.0 |

The difference between the last two rows is partition noise and nothing else.

## Try this

- Raise the power of [[tx_fp]] by 10 dB. Its $Q$ barely moves: a noise floor, not a power limit.
- Set [[fibre_fp]]'s dispersion to 0, as at 1310 nm. The modes arrive together and $Q$ doubles,
  to about 26; what is left is the laser's total intensity noise. That is why Fabry–Perot lasers
  live in the O-band.
- Cut [[tx_fp]]'s modes from 7 to 3.
""".strip(),
    },
    "pam4_lane": {
        "title": "2.5 A 53 Gb/s PAM4 lane with an equaliser",
        "body": r"""
A PAM4 driver [[pam4]] turns every two bits of [[prbs]] into one of four voltages, and a
Mach–Zehnder [[mzm]] writes them onto 1310 nm light at 26.5625 GBd: 53 Gb/s, one lane of a
200G data-centre link. The receiver [[pin]] has only 7 GHz of bandwidth [[lpf]], about a quarter
of the symbol rate. The eye [[eye]] shows the result, and two equalisers measure it: [[eq_off]]
with one tap (no equalisation) and [[eq]] with nine feed-forward taps and two of decision
feedback.

## Four levels on a cosine

The modulator's power is $\cos^2$ of its voltage, so equally spaced volts give unequal steps of
power and the outer eyes close first. The driver **pre-distorts**: it puts each level at the
voltage whose power is evenly spaced,

$$V_k = \frac{2V_\pi}{\pi}\arccos\sqrt{P_k/P_\text{max}}$$

## Inter-symbol interference

A filter narrower than the symbol rate spreads each symbol into its neighbours. Sampled at the
best instant, what arrives is

$$y_n = h_0 a_n + \sum_{k \ne 0} h_k a_{n-k} + \text{noise}$$

and the sum is the ISI. With four levels only a third of the eye height is left between them, so
ISI that a two-level signal would survive closes PAM4.

## The equaliser

The feed-forward equaliser is a short filter that inverts the channel,
$z_n = \sum_j c_j y_{n-j}$, trained to minimise the error. The decision-feedback part subtracts
the ISI of symbols already decided, $z_n - \sum_k b_k \hat a_{n-k}$. A DFE cannot reach a
*precursor*: a symbol it has not decided yet.

| Receiver | SNR | Symbol errors |
| :-- | :-- | :-- |
| [[eq_off]] | 9.9 dB | 609 / 4096 |
| [[eq]] | 38.7 dB | 0 / 4096 |

## Try this

- Turn off predistortion on [[pam4]]. Which of the three eyes closes first?
- Widen [[lpf]] to 20 GHz and see how much the equaliser still has to do.
- Set [[eq]] to 3 feed-forward taps and no feedback.
""".strip(),
    },
    "cwdm4": {
        "title": "2.6 CWDM4: four lanes in the O-band",
        "body": r"""
Four 25.78 Gb/s lanes on the coarse WDM grid, [[tx0]] to [[tx3]] at 1331, 1311, 1291 and 1271
nm, are combined by [[mux]], sent through 10 km of standard fibre [[fibre]] with no amplifier,
split by [[demux]] and received on four PIN receivers. This is how a 100G data-centre optic
carries its four lanes. The spectrum is on [[osa]], and each lane's eye, [[eye0]] to [[eye3]],
in the Eye tab.

## The coarse grid

ITU-T G.694.2 places channels 20 nm apart in *wavelength* (dense WDM is uniform in frequency
instead). Each 20 nm slot leaves about 13 nm of passband, wide enough for a laser with no
temperature control to drift in, which is what makes CWDM optics cheap. [[mux]] steps its
channels in nanometres for this.

## Why the O-band

Standard fibre's dispersion crosses zero near 1310 nm. Close to that point it grows linearly
with the slope $S_0 \approx 0.092$ ps/(nm²·km):

$$D(\lambda) \approx S_0\,(\lambda - \lambda_0)$$

| Lane | $\lambda$ | $D$ [ps/(nm·km)] | $Q$ |
| :-- | :-- | :-- | :-- |
| [[ber0]] | 1331 nm | +1.9 | 25.3 |
| [[ber1]] | 1311 nm | +0.1 | 26.0 |
| [[ber2]] | 1291 nm | −1.7 | 25.9 |
| [[ber3]] | 1271 nm | −3.6 | 20.3 |

The lane furthest from the zero loses the most. With the slope set to zero all four read about
26.

## The budget

Fibre loss is higher here, about 0.35 dB/km, so 10 km costs 3.5 dB. Per lane:

$$0\ \text{dBm} - 3\ (\text{modulator}) - 2\ (\text{mux}) - 3.5\ (\text{fibre}) - 2\ (\text{demux}) \approx -10.5\ \text{dBm}$$

## Try this

- Set [[fibre]]'s dispersion slope to 0 and compare the four $Q$ values.
- Lengthen [[fibre]] to 40 km. Which lane fails first, and why?
- Move [[tx3]] by 7 nm and watch [[demux]] cut it off.
""".strip(),
    },
    "gpon": {
        "title": "2.7 GPON downstream: one fibre, 32 homes",
        "body": r"""
A passive optical network shares one fibre between many homes with no powered equipment in the
street. The optical line terminal at the exchange, [[olt]] with [[mzm]], sends 2.488 Gb/s at
1490 nm through 20 km of feeder fibre [[feeder]] to a 1:32 splitter [[split32]]. Every home's
ONU receives the whole stream and keeps its own part. One ONU is drawn here: a meter [[pm_onu]]
and an APD receiver [[apd]], whose eye is [[eye]].

## The split is the budget

An ideal $1{:}N$ splitter costs $10\log_{10}N$:

$$1{:}32 \;\Rightarrow\; 15.05\ \text{dB}$$

plus about 1.5 dB of excess in a real planar splitter. That is more than the 5 dB of the 20 km
feeder at 0.25 dB/km.

$$P_\text{ONU} = 3 - 3\ (\text{modulator}) - 5\ (\text{feeder}) - 16.55\ (\text{split}) = -21.55\ \text{dBm}$$

[[pm_onu]] reads −21.56 dBm. A GPON class B+ ONU must work at −28 dBm, which is why ONUs use an
APD (lesson 2.2), and the link here has 6.5 dB to spare: [[ber]] reads $Q = 35.6$.

## Two directions, two wavelengths

Downstream is 1490 nm and broadcast. Upstream is 1310 nm, and the ONUs take turns: each sends a
short burst in its own time slot. A burst arrives at the OLT with a power that depends on that
home's distance and splitter, so the upstream receiver has to settle on each burst's level within
a few bits. That needs a **burst-mode receiver**, which is not in the palette yet.

## Try this

- Add 3 dB to the excess loss of [[split32]]: that is a 1:64 split. What is left of the margin?
- Lengthen [[feeder]] to 40 km. Does the APD still hold $Q = 6$?
- Swap [[apd]] for a PIN photodiode and find the longest feeder that still works.
""".strip(),
    },
    "edfa_basics": {
        "title": "3.1 One EDFA, before and after",
        "body": r"""
A weak CW line [[laser]] at −20 dBm goes into an erbium-doped fibre amplifier [[edfa]] with 20 dB
of gain and a 5 dB noise figure. Meters and spectra read both sides: [[pm_in]] and [[osa_in]]
before, [[pm_out]], [[osnr]] and [[osa_out]] after.

## Gain, and what comes with it

The signal leaves 20 dB stronger, at 0.00 dBm. The amplifier also adds amplified spontaneous
emission (ASE): broadband light in both polarisations, with a power per bandwidth $B$ of

$$P_\text{ASE} = 2\,n_\text{sp}\,h\nu\,(G - 1)\,B, \qquad NF \approx 2\,n_\text{sp}$$

On [[osa_out]] it is the flat floor either side of the line.

## Reading the numbers

| Reading | Value |
| :-- | :-- |
| [[pm_in]] | −20.00 dBm |
| [[pm_out]] | +0.65 dBm |
| ASE floor on [[osa_out]], per 0.1 nm | −32.95 dBm |
| [[osnr]] in 0.1 nm | 32.95 dB |

[[pm_out]] reads 0.65 dB more than the signal, because a power meter has no filter. It counts the
ASE across the whole 4 THz band, 0.16 mW of it. The OSNR compares the signal with the ASE in
0.1 nm (12.5 GHz) only. At 1550 nm, $10\log_{10}(h\nu \cdot 12.5\,\text{GHz})$ is −58 dBm, so

$$\text{OSNR} \approx 58 + P_\text{in} - NF = 58 - 20 - 5 = 33\ \text{dB}$$

The gain cancels out: the OSNR is set by the power arriving at the amplifier, not the power
leaving it.

## Try this

- Raise [[laser]] to −10 dBm. The OSNR rises by the same 10 dB.
- Set [[edfa]]'s noise figure to 3 dB, the quantum limit for a phase-insensitive amplifier.
- Turn on *saturate* on [[edfa]] and raise [[laser]] to +5 dBm. Does it still give 20 dB?
""".strip(),
    },
    "amplified_chain": {
        "title": "3.2 A chain of amplified spans",
        "body": r"""
A 0 dBm line [[laser]] goes through eight 80 km spans, [[span1]] to [[span8]]. After each one an
EDFA ([[edfa1]] to [[edfa8]]) makes up exactly the 16 dB that span lost. The OSNR is read after 1,
2, 4 and 8 spans by [[osnr1]], [[osnr2]], [[osnr4]] and [[osnr8]], and the spectrum after the
first and the last by [[osa1]] and [[osa8]].

## Noise adds up, the signal does not

Every span is identical, so every amplifier receives the same −16 dBm and adds the same ASE.
The signal leaves each amplifier at 0 dBm again, but the ASE from earlier amplifiers is carried
along. After $N$ spans there is $N$ times the noise of one:

$$\text{OSNR}_N = \text{OSNR}_1 - 10\log_{10}N$$

| Spans | Reach | OSNR in 0.1 nm |
| :-- | :-- | :-- |
| 1 | 80 km | 36.95 dB |
| 2 | 160 km | 33.94 dB |
| 4 | 320 km | 30.93 dB |
| 8 | 640 km | 27.92 dB |

Each doubling costs 3.01 dB. On [[osa1]] the ASE floor is 37 dB under the line, and on
[[osa8]] it has risen to 28 dB under it.

## Try this

- Lengthen every span to 100 km and raise every gain to 20 dB. How much OSNR does the extra
  4 dB of span loss cost?
- Raise [[laser]] to +3 dBm. Every OSNR rises by 3 dB. 3.5 is why that cannot go on.
""".strip(),
    },
    "launch_power": {
        "title": "3.5 Optimum launch power",
        "body": r"""
A 10 Gb/s OOK signal [[tx]] [[mzm]] is launched by a booster [[booster]] into five 80 km spans of
standard fibre. Each span is followed by a spool of dispersion-compensating fibre ([[dcf1]] to
[[dcf5]]) that undoes its 1360 ps/nm, and an amplifier that makes up the 22.8 dB the two lost.
The launch power is read on [[pm_launch]], and the receiver ends in [[osnr]], [[ber]] and [[eye]].

## Two limits

More launch power means more OSNR, decibel for decibel. But standard fibre's Kerr
nonlinearity, $\gamma = 1.3$ /(W·km), shifts the phase of the signal by its own power:

$$\phi_\text{NL} = \gamma P L_\text{eff}, \qquad L_\text{eff} = \frac{1 - e^{-\alpha L}}{\alpha} \approx 21\ \text{km}$$

The phase varies with the bit pattern, and the dispersion of the next span turns that phase into
distortion of the power. Too little power and the noise closes the eye; too much and the
distortion does.

## The bathtub

Sweep the gain of [[booster]] from 0 to 15 dB in steps of 3 (*Sweep* in the toolbar):

| Booster gain | Launch | OSNR | Q |
| :-- | :-- | :-- | :-- |
| 0 dB | −3.0 dBm | 20.2 dB | 5.90 |
| 3 dB | 0.0 dBm | 23.1 dB | 9.74 |
| 6 dB | +3.0 dBm | 26.1 dB | 12.24 |
| 9 dB | +6.0 dBm | 29.1 dB | 8.71 |
| 12 dB | +9.0 dBm | 32.1 dB | 4.18 |
| 15 dB | +12.0 dBm | 35.0 dB | 1.67 |

The OSNR keeps rising to the last row, but the Q peaks at +3 dBm and falls on either side.
Designing a line system starts from this curve.

## Try this

- Set the nonlinearity of every span to 0 and sweep again. The Q now only rises.
- Compare [[eye]] at 6 dB of gain with 12 dB.
""".strip(),
    },
    "fwm_dsf": {
        "title": "3.6 Four-wave mixing on a zero-dispersion fibre",
        "body": r"""
Four CW channels [[ch0]] to [[ch3]], 100 GHz apart from 1550 nm at +6 dBm each, are combined by
[[mux]] and split [[split]] into two fibres of 80 km. [[fibre_dsf]] is dispersion-shifted fibre
(G.653), with its zero dispersion at 1550 nm. [[fibre_nzdsf]] is non-zero dispersion-shifted
fibre (G.655), with 4 ps/(nm·km). The spectra are on [[osa_dsf]] and [[osa_nzdsf]].

## Light nobody launched

The Kerr effect mixes every three channels into a fourth at

$$f_{ijk} = f_i + f_j - f_k$$

With four channels on a uniform grid, those frequencies are the channels themselves and the
grid slots on either side. The mixing is efficient only while the channels stay in phase with
each other, and dispersion is what moves them out of phase:

$$\Delta\beta = \frac{2\pi\lambda^2}{c}\,D\,\Delta f_{ik}\,\Delta f_{jk}$$

At $D = 0$ nothing dephases them.

## The two spectra

Each channel arrives at about −13 dBm per 0.1 nm.

| Slot | G.653 | G.655 |
| :-- | :-- | :-- |
| below the channels | −24.1 dBm | −63.2 dBm |
| above the channels | −23.4 dBm | −62.0 dBm |
| channels | −13.4 to −13.7 dBm | −13.0 dBm |

On G.653 the mixing products are only about 10 dB below the channels, and the channels
themselves have changed: products landed on them and beat with them. A few ps/(nm·km) of
dispersion suppresses the products by 39 dB. That is why G.653, built to have no dispersion
at 1550 nm, was abandoned for WDM, and why G.655 was designed to keep a little.

## Try this

- Lower every channel to 0 dBm. Each product falls by 3 dB per 1 dB of channel power.
- Set [[fibre_nzdsf]]'s dispersion to 17, standard fibre. How far down do the products go?
""".strip(),
    },
    "roadm": {
        "title": "3.9 A ROADM add/drop node",
        "body": r"""
Four 10 Gb/s channels [[tx0]] to [[tx3]], 100 GHz apart, cross a span [[span_a]] [[edfa_a]] and
reach a node. There a wavelength-selective switch [[wss]] lets three channels through
(*express*), takes channel 1 off to a local receiver (*drop*, [[ber_drop]]) and puts a new
channel with different data on the same wavelength (*add*, [[tx_add]]). The line continues
through [[span_b]] [[edfa_b]] to [[demux_b]], where the added channel is received by
[[ber_add]]. The spectra on [[osa_a]] and [[osa_b]] look alike, but channel 1 now carries other
data.

## In-band crosstalk

A switch does not block perfectly. What it leaks of the dropped channel, its *isolation* below
the arrival, reaches the out port on exactly the frequency of the added channel. There no filter
can separate them. The two fields add, and the receiver sees their beat:

$$P = |E_\text{add} + E_\text{leak}|^2 = P_\text{add} + P_\text{leak} + 2\sqrt{P_\text{add}P_\text{leak}}\cos\Delta\phi$$

The beat term is in amplitude, so crosstalk 35 dB down in power is only 17.5 dB down in field.

| [[wss]] isolation | Q, added channel |
| :-- | :-- |
| 80 dB | 41.2 |
| 35 dB | 26.2 |
| 25 dB | 10.8 |
| 20 dB | 6.2 |

The dropped channel reads Q 32.1 on [[ber_drop]] whatever the isolation is. The leak only
harms the channel that takes its place.

## Try this

- Sweep the *isolation* of [[wss]] from 15 to 40 dB.
- Lower [[tx_add]]'s power by 5 dB. The leak is now 5 dB closer to the added channel.
""".strip(),
    },
}
