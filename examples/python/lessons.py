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


# ---------------------------------------------------------------------------
# Phase 4: coherent links


def _qpsk_b2b(at: Layout) -> list[dict[str, Any]]:
    vsa = at["vsa"]
    return [
        frame(at, ["prbs", "map", "drv", "tx", "mod"], "QPSK transmitter", "optical"),
        frame(at, ["voa", "ase"], "Noise loading", "binary"),
        frame(at, ["lo", "rx", "smp"], "Coherent receiver", "electrical"),
        frame(at, ["osnr", "osa"], "Optical instruments", "metric"),
        frame(at, ["vsa", "const"], "Constellation", "metric"),
        side_note(vsa, "$\\mathrm{EVM} \\approx 1/\\sqrt{\\mathrm{SNR}}$: 11.75 dB reads 25.9 %."),
        side_arrow(vsa),
    ]


def _dualpol(at: Layout) -> list[dict[str, Any]]:
    eq = at["eq"]
    middle = eq["x"] + NODE_W / 2.0
    return [
        frame(at, ["prbs_x", "map_x", "drv_x", "mod_x"], "Tributary x", "optical"),
        frame(at, ["prbs_y", "map_y", "drv_y", "mod_y"], "Tributary y", "optical"),
        frame(at, ["pbc", "rot"], "Channel", "binary"),
        frame(at, ["lo", "rx", "smp_x", "smp_y"], "Dual-pol receiver", "electrical"),
        frame(at, ["eq", "cr_x", "cr_y"], "DSP", "symbol"),
        note(
            eq["x"] - 10.0,
            eq["y"] + 230.0,
            "Four filters, $x$ and $y$ in, $x$ and $y$ out: the inverse of the "
            "rotation, learned blind.",
            w=240.0,
            h=80.0,
        ),
        arrow(middle, eq["y"] + 228.0, middle, eq["y"] + NODE_H + 14.0),
    ]


def _acquisition(at: Layout) -> list[dict[str, Any]]:
    acq = at["acq"]
    return [
        frame(at, ["prbs", "map", "drv", "tx", "mod"], "16-QAM transmitter", "optical"),
        frame(at, ["lo", "rx"], "LO 20 GHz away", "electrical"),
        frame(at, ["acq", "tr", "smp", "fo"], "Coarse, then fine", "symbol"),
        below_note(
            acq,
            "Finds the band on the waveform's spectrum, before the matched filter throws it away.",
        ),
        below_arrow(acq),
    ]


def _zr(at: Layout) -> list[dict[str, Any]]:
    osnr = at["osnr"]
    return [
        frame(at, ["prbs_x", "map_x", "drv_x"], "Data x", "binary"),
        frame(at, ["prbs_y", "map_y", "drv_y"], "Data y", "binary"),
        frame(at, ["tx", "pbs", "mod_x", "mod_y", "pbc"], "DP-IQ transmitter", "optical"),
        frame(at, ["fib", "edfa"], "One 80 km span", "binary"),
        frame(at, ["lo", "rx", "cdc_x", "cdc_y", "smp_x", "smp_y"], "Receiver", "electrical"),
        frame(at, ["eq", "cr_x", "cr_y", "vsa_x", "vsa_y"], "DSP", "symbol"),
        side_note(
            osnr,
            "The span leaves 23.2 dB of OSNR. Compare it with what the rate needs.",
            h=70.0,
        ),
        side_arrow(osnr),
    ]


def _loop(at: Layout) -> list[dict[str, Any]]:
    scope = at["scope"]
    return [
        frame(at, ["delay", "loss", "loop"], "The loop: 800 ps, 1 dB", "binary"),
        below_note(scope, "One pulse in, a train out: each lap 800 ps later and weaker."),
        below_arrow(scope),
    ]


def _pcs(at: Layout) -> list[dict[str, Any]]:
    shaped = at["map_s"]
    bottom = at["mod_s"]["y"] + NODE_H + 40.0
    middle = shaped["x"] + NODE_W / 2.0
    return [
        frame(at, ["prbs_u", "map_u", "drv_u", "tx_u", "mod_u"], "Uniform 16-QAM", "optical"),
        frame(at, ["prbs_s", "map_s", "drv_s", "tx_s", "mod_s"], "Shaped 16-QAM", "optical"),
        frame(at, ["voa_u", "ase_u", "voa_s", "ase_s"], "Same noise", "binary"),
        frame(at, ["vsa_u", "const_u", "vsa_s", "const_s"], "Mutual information", "metric"),
        note(
            shaped["x"] - 60.0,
            bottom,
            "Inner points more often: $p(x) \\propto e^{-\\lambda |x|^2}$, "
            "3.7 bit per symbol instead of 4.",
            w=240.0,
            h=80.0,
        ),
        arrow(middle, bottom - 2.0, middle, shaped["y"] + NODE_H + 14.0),
    ]


# ---------------------------------------------------------------------------
# Phase 5: photonic integrated circuits


def above_note(block: dict[str, float], text: str) -> list[dict[str, Any]]:
    """A narrow note over ``block`` and an arrow down to it, clear of its neighbours."""
    middle = block["x"] + NODE_W / 2.0
    return [
        note(block["x"] - 27.0, block["y"] - 150.0, text, w=170.0, h=96.0),
        arrow(middle, block["y"] - 52.0, middle, block["y"] - 12.0),
    ]


def _source(at: Layout) -> dict[str, Any]:
    return frame(at, ["seed", "ase", "osa_in"], "White light", "optical")


def _coupler_length(at: Layout) -> list[dict[str, Any]]:
    dc = at["dc"]
    return [
        _source(at),
        frame(at, ["pm_bar", "osa_bar", "pm_cross", "osa_cross"], "Bar and cross", "metric"),
        *above_note(
            dc, "$\\kappa = \\sin^2(\\pi\\,\\Delta n\\,L/\\lambda)$: 9.69 µm is half of $L_c$."
        ),
    ]


def _mzi(at: Layout) -> list[dict[str, Any]]:
    device = at["mzi"]
    return [
        _source(at),
        frame(at, ["osa_bar", "osa_cross"], "Two complementary combs", "metric"),
        *above_note(device, "One arm 200 µm longer: a comb every $c/(n_g\\,\\Delta L)$ = 357 GHz."),
    ]


def _ring(at: Layout) -> list[dict[str, Any]]:
    device = at["ring"]
    return [
        _source(at),
        frame(at, ["osa_through", "osa_drop"], "Through and drop", "metric"),
        *above_note(device, "A resonance every $c/(n_g L)$ = 714 GHz, 12.2 GHz wide."),
    ]


def _birefringent_ring(at: Layout) -> list[dict[str, Any]]:
    biref = at["ring_biref"]
    return [
        _source(at),
        frame(at, ["ring_plain", "osa_plain"], "One polarisation model", "binary"),
        frame(at, ["ring_biref", "osa_biref"], "TE and TM apart", "symbol"),
        below_note(biref, "TE every 714 GHz, TM every 789 GHz."),
        below_arrow(biref),
    ]


def _chip_couplers(at: Layout) -> list[dict[str, Any]]:
    return [
        frame(at, ["tx", "edge_in", "wg_edge", "edge_out", "pm_edge"], "Edge couplers", "optical"),
        frame(at, ["gc_in", "wg_gc", "gc_out"], "Grating couplers", "optical"),
        frame(at, ["seed", "ase", "osa_in", "pm_ase"], "White light", "binary"),
        frame(at, ["pm_gc", "osa_gc"], "Out", "metric"),
    ]


def _pdk_splitter(at: Layout) -> list[dict[str, Any]]:
    kit = at["kit"]
    return [
        frame(at, ["ideal", "pm_ideal1", "pm_ideal4"], "Textbook", "binary"),
        frame(at, ["kit", "pm_kit1", "pm_kit4"], "From the kit", "optical"),
        note(
            kit["x"] - 30.0,
            kit["y"] + NODE_H + 40.0,
            "0.45 dB excess loss and 0.35 dB imbalance, measured on a wafer.",
            w=150.0,
            h=96.0,
        ),
        arrow(
            kit["x"] + NODE_W / 2.0,
            kit["y"] + NODE_H + 38.0,
            kit["x"] + NODE_W / 2.0,
            kit["y"] + NODE_H + 14.0,
        ),
    ]


# ---------------------------------------------------------------------------
# Phase 6: sensing


def _fbg_strain(at: Layout) -> list[dict[str, Any]]:
    return [
        _source(at),
        frame(at, ["working", "reference"], "One fibre, two gratings", "binary"),
        frame(at, ["osa_work", "osa_ref"], "Each reflection", "metric"),
        below_note(at["reference"], "Loose in its tube: it feels the heat and none of the load."),
        below_arrow(at["reference"]),
    ]


def _fbg_drop(at: Layout) -> list[dict[str, Any]]:
    circ = at["circ"]
    return [
        frame(at, ["ch1550", "ch1552", "mux"], "Two channels", "optical"),
        frame(at, ["circ", "fbg"], "Drop", "binary"),
        frame(at, ["pm_drop", "osa_drop"], "Dropped", "metric"),
        frame(at, ["pm_thru", "osa_thru"], "Express", "metric"),
        *above_note(circ, "Through the circulator twice: 2 × 0.7 dB, plus the grating's 0.30 dB."),
    ]


def _tilted_grating(at: Layout) -> list[dict[str, Any]]:
    return [
        _source(at),
        frame(at, ["tfbg_air", "osa_air"], "In air", "binary"),
        frame(at, ["tfbg_water", "osa_water"], "In water", "symbol"),
        below_note(
            at["tfbg_water"], "The comb moves by up to 88 pm. The Bragg line does not move."
        ),
        below_arrow(at["tfbg_water"]),
    ]


def _lpg_pair(at: Layout) -> list[dict[str, Any]]:
    return [
        _source(at),
        frame(at, ["lpg_one", "osa_one"], "One grating", "binary"),
        frame(at, ["lpg_pair", "osa_pair"], "Two, 200 mm apart", "symbol"),
        below_note(at["lpg_pair"], "Fringes $\\lambda^2/(\\Delta n_g\\,d)$ apart: 2.31 nm here."),
        below_arrow(at["lpg_pair"]),
    ]


def _otdr(at: Layout) -> list[dict[str, Any]]:
    fut = at["fut"]
    return [
        frame(at, ["pulse"], "Probe", "optical"),
        frame(at, ["otdr"], "Echo", "metric"),
        below_note(fut, "25 km on the drawing. The echo stops at 17.30 km: dig there."),
        below_arrow(fut),
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
    "qpsk_b2b": _qpsk_b2b,
    "dualpol": _dualpol,
    "acquisition": _acquisition,
    "zr400": _zr,
    "zr800": _zr,
    "loop": _loop,
    "pcs": _pcs,
    "coupler_length": _coupler_length,
    "mzi": _mzi,
    "ring": _ring,
    "birefringent_ring": _birefringent_ring,
    "chip_couplers": _chip_couplers,
    "pdk_splitter": _pdk_splitter,
    "fbg_strain": _fbg_strain,
    "fbg_drop": _fbg_drop,
    "tilted_grating": _tilted_grating,
    "lpg_pair": _lpg_pair,
    "otdr": _otdr,
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
    "qpsk_b2b": {
        "title": "4.1 QPSK back to back",
        "body": r"""
The simplest coherent link. [[prbs]] feeds two bits per symbol to [[map]], which places each
pair on one of four points. [[drv]] shapes the pulses and drives the two arms of [[mod]], so
the light from [[tx]] carries the symbol in both its amplitude and its phase. At the far end
[[rx]] beats the signal against a local oscillator [[lo]] on the same wavelength and recovers
the in-phase and quadrature parts. [[smp]] filters and samples them once per symbol, and
[[vsa]] and [[const]] show where the symbols landed.

There is no fibre. Between the two ends sits a bench trick: [[voa]] attenuates the signal and
[[ase]] amplifies it back by exactly as much. The power comes back, the amplifier's noise
stays, so the attenuation sets the OSNR without changing anything else. [[osnr]] reads it and
[[osa]] shows the noise floor under the signal.

## From OSNR to SNR

OSNR counts noise in a 12.5 GHz reference bandwidth, both polarisations. The receiver hears
one polarisation through a matched filter as wide as the symbol rate $R_s$:

$$\mathrm{SNR} = \mathrm{OSNR} \cdot \frac{2 B_\text{ref}}{R_s}, \qquad 10\log_{10}\frac{2 \times 12.5}{32} = -1.07\ \text{dB}$$

The transmitter has a floor of its own. With no noise loaded the link reads 21.2 dB, set by
the pulse shaping and the modulator. Noise powers add, so the SNR the receiver sees is

$$\frac{1}{\mathrm{SNR}} = \frac{1}{\mathrm{SNR}_\text{OSNR}} + \frac{1}{\mathrm{SNR}_\text{tx}}$$

## EVM

The error vector is the distance from each received symbol to the point it should have been,
relative to the RMS amplitude. When noise is all there is, $\mathrm{EVM} \approx 1/\sqrt{\mathrm{SNR}}$.

| [[voa]] | OSNR | SNR | EVM |
| :-- | :-- | :-- | :-- |
| 26 dB | 22.39 dB | 18.21 dB | 12.3 % |
| 29 dB | 19.39 dB | 16.47 dB | 15.0 % |
| 32 dB | 16.39 dB | 14.28 dB | 19.3 % |
| 35 dB | 13.39 dB | 11.75 dB | 25.9 % |
| 38 dB | 10.39 dB | 9.00 dB | 35.5 % |

The project opens at 35 dB: a wide cloud on each point, and still not one symbol error.

## Try this

- Set [[voa]] and [[ase]] both to 41 dB and count the errors on [[vsa]].
- Turn the matched filter of [[smp]] off. The noise is now read over the whole simulated
  bandwidth instead of the symbol rate, and the SNR falls by several decibels.
""".strip(),
    },
    "dualpol": {
        "title": "4.4 Two polarisations, one wavelength",
        "body": r"""
Two 16-QAM tributaries share one laser [[tx]]. [[sp]] splits it, [[mod_x]] and [[mod_y]]
modulate each half with its own data, and [[pbc]] combines them on orthogonal polarisations.
That doubles the rate to $2 \times 4 \times 32 = 256$ Gb/s on the same wavelength.

A fibre does not keep the polarisation. [[rot]] turns it by 30° and adds a phase between the
axes, so each branch of the receiver [[rx]] now holds a mixture of both tributaries. The
constellation [[const_raw]] shows it: sixteen points become a smear.

## The butterfly equaliser

The channel is a 2×2 matrix $\mathbf{J}$ acting on the field. [[eq]] learns its inverse with
four filters, one from each input to each output ($*$ is convolution):

$$x' = h_{xx} * x + h_{xy} * y, \qquad y' = h_{yx} * x + h_{yy} * y$$

It has no training sequence. It adapts until each output has the constant-modulus shape a
QAM signal should have, then [[cr_x]] and [[cr_y]] remove the remaining phase.

| | EVM x | EVM y | symbol errors |
| :-- | :-- | :-- | :-- |
| 0°, no equaliser | 2.5 % | 2.5 % | 0 |
| 30°, no equaliser | 218 % | 123 % | 6217 |
| 30°, with [[eq]] | 2.51 % | 2.52 % | 0 |

Without the equaliser the branches are not degraded. They carry no recoverable data at all.

## Try this

- Sweep the *angle* of [[rot]] from 0° to 90°. At 90° the tributaries arrive swapped.
- Watch [[const_x]] and [[const_y]] while you do.
""".strip(),
    },
    "acquisition": {
        "title": "4.5 Acquiring a carrier 20 GHz away",
        "body": r"""
The transmitter [[tx]] sends 16-QAM at 32 GBd. The local oscillator [[lo]] is a free laser
tuned 20 GHz away, as a receiver is before it has locked. [[rx]] mixes them, so the symbols
come out spinning at 20 GHz.

## Why one estimator is not enough

The fine stage [[fo]] raises each symbol to the fourth power. That removes the 16-QAM phase
pattern and leaves $e^{j 4 \cdot 2\pi \Delta f\, t}$, whose frequency gives $\Delta f$. A phase
measured once per symbol can only tell frequencies apart within $\pm R_s/2$, and after the
fourth power that range shrinks to

$$|\Delta f| < \frac{R_s}{2 \times 4} = \frac{32\ \text{GHz}}{8} = 4\ \text{GHz}$$

Beyond it the estimate folds back and reports a wrong offset with full confidence. A rotation
by a quarter turn per symbol is invisible, because the alphabet looks the same after it.

The coarse stage [[acq]] works on the waveform instead, before [[tr]] and the matched filter
in [[smp]]. It finds where the band sits in the spectrum and shifts it to zero. What is left
is small enough for [[fo]].

| LO 20 GHz away | offset found | EVM | symbol errors |
| :-- | :-- | :-- | :-- |
| [[fo]] alone | +2.97 GHz (folded) | 8300 % | 1797 |
| [[acq]] then [[fo]] | 20.07 GHz, then −71 MHz | 6.7 % | 0 |

In the first row nothing is left of the constellation.

## Try this

- Retune [[lo]] to 0.2 GHz away. Both stages work; [[acq]] then has almost nothing to do.
- Watch [[const]] while you move [[lo]] 3, 4 and 5 GHz away.
""".strip(),
    },
    "zr400": {
        "title": "4.6 400ZR over one span",
        "body": r"""
A 400ZR module carries 400 Gb/s on one wavelength as dual-polarisation 16-QAM at 59.84 GBd.
[[tx]] is split by [[pbs]], [[mod_x]] and [[mod_y]] carry one tributary each, and [[pbc]]
combines them. The signal crosses one 80 km span [[fib]] and [[edfa]], then a dual-polarisation
receiver [[rx]], dispersion compensation [[cdc_x]] [[cdc_y]], the butterfly equaliser [[eq]] and
carrier recovery.

## The line rate

$$R = R_s \times \log_2 M \times 2 = 59.84 \times 4 \times 2 = 478.72\ \text{Gb/s}$$

The 78.72 Gb/s above the 400 Gb/s payload pays for forward error correction and framing.

## Required OSNR

The error rate of 16-QAM depends on the SNR, and the SNR follows from the OSNR as in 4.1, now
for two polarisations. For a pre-FEC BER of $2\times10^{-2}$, the level a soft-decision FEC of
this class corrects from:

$$\mathrm{OSNR}_\text{req} = 19.47\ \text{dB}$$

[[osnr]] reads 23.19 dB after the span, 3.7 dB of margin. [[vsa_x]] and [[vsa_y]] measure
EVM 16.1 % and 16.2 % and a BER of $2.3\times10^{-3}$ and $1.8\times10^{-3}$, ten times below the
threshold.

## Try this

- Open 4.6 800ZR. It is the same link at twice the symbol rate.
- Raise the *length* of [[fib]] to 100 km and set [[edfa]]'s gain to 20 dB.
""".strip(),
    },
    "zr800": {
        "title": "4.6 800ZR over one span",
        "body": r"""
The same link as 400ZR, at twice the symbol rate: 119.68 GBd dual-polarisation 16-QAM from
[[tx]] through [[mod_x]], [[mod_y]] and [[pbc]], one 80 km span [[fib]] [[edfa]], and the same
receiver [[rx]] and DSP [[eq]].

$$R = 119.68 \times 4 \times 2 = 957.44\ \text{Gb/s}$$

## What doubling the rate costs

Twice the symbol rate means twice the receiver bandwidth, so twice the noise for the same
OSNR. The required OSNR rises by $10\log_{10}2 = 3.01$ dB:

$$\mathrm{OSNR}_\text{req} = 22.48\ \text{dB}$$

The span gives the same 23.19 dB on [[osnr]] as for 400ZR, so the margin is down to 0.7 dB.
[[vsa_x]] and [[vsa_y]] measure EVM 23.3 % and 22.8 % and a BER of $2.1\times10^{-2}$ and
$2.0\times10^{-2}$: right on the threshold.

The other way to 800 Gb/s is a denser format at a lower symbol rate, such as DP-64QAM at
79.8 GBd. It needs less spectrum and even more OSNR.

## Try this

- Lower [[fib]] to 60 km and [[edfa]]'s gain to 12 dB, and watch the BER fall.
""".strip(),
    },
    "loop": {
        "title": "4.7 A recirculating loop",
        "body": r"""
A laboratory tests a transoceanic link without owning one. It closes a few spans on themselves
and sends the signal round many times. Here the loop is reduced to its bones.

[[pulse]] sends one 10 ps pulse into a 3 dB [[coupler]]. Half leaves straight away, to the
oscilloscope [[scope]]. The other half enters the loop: [[delay]] takes 800 ps, [[loss]] takes
1 dB, and [[loop]] brings it back to the coupler, which again lets half out and keeps half.

## Lap by lap

Each lap arrives 800 ps after the last. Lap 1 has crossed the coupler twice and the loss once;
every lap after has one more half and one more pass:

$$P_0 = \frac{P}{2}, \qquad P_1 = \frac{P}{4} \cdot 10^{-0.1}, \qquad P_{n+1} = \frac{P_n}{2} \cdot 10^{-0.1}$$

| lap | arrives | peak on [[scope]] |
| :-- | :-- | :-- |
| 0 | 0 ps | 0.500 mW |
| 1 | 800 ps | 0.199 mW |
| 2 | 1600 ps | 0.079 mW |
| 3 | 2400 ps | 0.031 mW |

In a real loop an amplifier replaces the loss, and the lap number is the distance: ten laps of
100 km are 1000 km.

The log warns that [[loop]] has not converged. Here that is right: each pass is a new lap, not a
step towards a steady state.

## Try this

- Set the *delay* of [[delay]] to 0. Every lap lands on the first and the train disappears.
- Lower [[loss]] to 0 dB and count how many laps stay visible.
""".strip(),
    },
    "pcs": {
        "title": "4.8 Probabilistic shaping",
        "body": r"""
Two 16-QAM transmitters through the same noise. The top one, [[map_u]], sends all sixteen
points equally often. The bottom one, [[map_s]], sends the inner points more often than the
outer ones, with a Maxwell–Boltzmann distribution:

$$p(x) \propto e^{-\lambda |x|^2}$$

$\lambda$ is set so a symbol carries an entropy of 3.7 bit instead of 4. Both branches are
loaded to the same OSNR by [[voa_u]] [[ase_u]] and [[voa_s]] [[ase_s]].

## Why it helps

At the same mean power, the shaped alphabet's points sit further apart, because the rare outer
points carry most of the energy. The signal looks more like Gaussian noise, which is what a
noisy channel carries best. The measure is the mutual information, the bits per symbol the
receiver can really get through:

$$I(X;Y) = \mathbb{E}\left[\log_2 \frac{q(y \mid x)}{\sum_{x'} p(x')\, q(y \mid x')}\right]$$

[[vsa_u]] and [[vsa_s]] compute it from the received symbols. The uniform branch gets
2.96 bit per symbol at an SNR of 9.1 dB.

| [[map_s]] entropy | back-off at [[tx_s]] | MI |
| :-- | :-- | :-- |
| 3.85 bit | 1.25 dB | 3.08 bit |
| 3.70 bit | 1.87 dB | 3.10 bit |
| 3.50 bit | 2.60 dB | 3.05 bit |
| 3.28 bit | 3.47 dB | 2.94 bit |

Too much shaping costs more than it gives: a symbol can never carry more than its entropy.
FlexO-8's 3.28 bit per symbol is matched to a higher SNR and a code rate of its own.

## The back-off

[[drv_s]] scales the outermost point to the modulator's full swing. A shaped signal's outer
points are further out at the same mean power, so it leaves [[mod_s]] weaker. The project
raises [[tx_s]] by that amount, so both branches launch the same mean power.

## Try this

- Set [[tx_s]] back to the same power as [[tx_u]]. The shaping gain disappears, as it does
  in a transmitter limited by its peak power.
- Compare [[const_u]] and [[const_s]].
""".strip(),
    },
    "coupler_length": {
        "title": "5.1 A directional coupler: split ratio against length",
        "body": r"""
Two waveguides run side by side, close enough for their fields to overlap. [[dc]] is such a
section. White light from [[ase]] enters one guide, and [[pm_bar]] and [[pm_cross]] read what
leaves each. [[osa_in]] shows the source, [[osa_bar]] and [[osa_cross]] the two outputs.

## Two supermodes

Together the two guides carry two modes of their own: an even one, in phase in both guides,
and an odd one, out of phase. Their effective indices differ by $\Delta n$. Light launched in
one guide is the sum of the two, and as they drift out of step the power walks across:

$$\kappa(L) = \sin^2\!\left(\frac{\pi\,\Delta n\,L}{\lambda}\right), \qquad L_c = \frac{\lambda}{2\,\Delta n}$$

With $\Delta n = 0.04$ all of it has crossed after $L_c = 19.375$ µm at 1550 nm, and half of it
after 9.69 µm. The project opens there, a 3 dB coupler.

| *length* of [[dc]] | cross |
| :-- | :-- |
| 0 µm | 0.000 |
| 4.84 µm | 0.146 |
| 9.69 µm | 0.500 |
| 14.53 µm | 0.853 |
| 19.38 µm | 1.000 |
| 29.06 µm | 0.500 |
| 38.75 µm | 0.000 |

Past $L_c$ the power walks back. At $2L_c$ it is home again.

## The ratio depends on the wavelength

$\lambda$ is in the formula, so a coupler cut for 3 dB at 1550 nm is not 3 dB elsewhere. At
1534 nm it crosses 0.508 and at 1566 nm 0.492. On [[osa_bar]] and [[osa_cross]], divided by
[[osa_in]], the two outputs tilt in opposite directions across the band, from −2.94 to
−3.08 dB. A real coupler drifts more, because $\Delta n$ itself changes with wavelength.

## Try this

- Sweep the *length* of [[dc]] from 0 to 40 µm and watch [[pm_cross]].
- Open the S-matrix tab, pick [[dc]], and widen the window to 1300–1700 nm.
""".strip(),
    },
    "mzi": {
        "title": "5.2 An MZI interleaver",
        "body": r"""
A Mach–Zehnder interferometer [[mzi]] splits the light in two, sends it down two arms and
recombines it. Here one arm is 200 µm longer than the other. White light from [[ase]] goes in;
[[osa_bar]] and [[osa_cross]] show the two outputs, and [[osa_in]] the light that went in.

## A delay is a filter

The extra length delays one arm by $\Delta\tau = n_g\,\Delta L / c$. The phase that delay adds
grows with frequency, so the two arms come back in phase, then out of phase, then in phase
again. Each output is a comb with a period of

$$\mathrm{FSR} = \frac{c}{n_g\,\Delta L} = \frac{c}{4.20 \times 200\ \mu\text{m}} = 356.9\ \text{GHz}$$

The two outputs are complementary: where [[osa_bar]] peaks, [[osa_cross]] has a null, so
alternate channels of a 357 GHz grid leave by alternate ports. That is an interleaver.

Measured on the traces divided by [[osa_in]], the peaks are 357.0 GHz apart on [[osa_bar]] and
356.9 GHz apart on [[osa_cross]], and the nulls are 46 dB deep.

## Group index, not effective index

The period is set by $n_g$, how fast the envelope travels. The effective index $n_\text{eff} = 2.44$
would give 614 GHz instead. Using it is a common way to be wrong about an interleaver while the
plot still looks right.

## Try this

- Set the *phase_shift* of [[mzi]] to $\pi/2$. The comb slides sideways and keeps its period.
- Set *length_difference* to 0. The device is now a switch, not a filter.
""".strip(),
    },
    "ring": {
        "title": "5.3 A ring resonator filter",
        "body": r"""
A silicon ring [[ring]], 100 µm round, sits between two bus waveguides. White light from
[[ase]] enters one bus. Light at a resonance couples into the ring and leaves by the other bus,
the *drop* port [[osa_drop]]. Everything else passes by on the *through* port [[osa_through]].
[[osa_in]] shows what went in.

## Free spectral range

A resonance is a wavelength that fits a whole number of times round the ring. They repeat every

$$\mathrm{FSR} = \frac{c}{n_g L} = \frac{c}{4.20 \times 100\ \mu\text{m}} = 713.8\ \text{GHz}$$

The drop peaks on [[osa_drop]] are 714.2 GHz apart, which is within one display bin of that.

## Linewidth and extinction

How sharp a resonance is depends on how fast light leaks out: through the two couplers
($\kappa = 0.05$ each) and through loss in the ring (2 dB/cm). Here the couplers dominate and
the linewidth is 12.2 GHz. On resonance the drop port passes all but 0.40 dB of the light and
the through port falls by 21.7 dB.

## Critical coupling

With only one bus, the through port goes fully dark when the coupling equals the round-trip
loss, $\kappa = 1 - a^2$. Over- or under-coupled, the notch fills in, by the *same* amount on
either side, so a measured depth does not tell you which side a ring is on.

## Try this

- Set *drop_coupling* of [[ring]] to 0 and *coupling* to 0.0046, near critical. The through
  notch on [[osa_through]] deepens.
- Lower *coupling* and *drop_coupling* to 0.01. The peaks narrow and the drop port loses more
  to the ring's own loss.
""".strip(),
    },
    "birefringent_ring": {
        "title": "5.4 A birefringent ring",
        "body": r"""
The same 100 µm ring twice, lit by the same unpolarised white light [[ase]], split by [[sp]].
[[ring_plain]] treats both polarisations alike. [[ring_biref]] has *birefringent* on: its TE and
TM modes have different indices, as in every real silicon strip waveguide.

## Two combs

TE light sees $n_g = 4.20$ and TM light $n_g = 3.80$, so each has its own free spectral range:

$$\mathrm{FSR}_\text{TE} = 713.8\ \text{GHz}, \qquad \mathrm{FSR}_\text{TM} = 788.9\ \text{GHz}$$

[[osa_plain]] shows one comb, peaks 714 GHz apart. [[osa_biref]] shows two combs laid over each
other. In this window the TE peaks sit at −299 and +415 GHz from 1550 nm, and the TM peaks at
−661, +128 and +916 GHz. Each polarisation carries half of the unpolarised light, so each comb on
[[osa_biref]] peaks 3 dB lower than on [[osa_plain]].

A ring like this is a polarisation filter. At a TE resonance the drop port passes TE and not TM.

## Try this

- Open the S-matrix tab on [[ring_biref]] and switch the polarisation between TE and TM.
- Set *n_group_tm* of [[ring_biref]] to 4.20. The two combs land on top of each other.
""".strip(),
    },
    "chip_couplers": {
        "title": "5.5 Getting light on and off a chip",
        "body": r"""
Two ways to get light from a fibre into a silicon waveguide and out again. On top, a laser
[[tx]] goes through an edge coupler [[edge_in]], 5 mm of waveguide [[wg_edge]] and a second
edge coupler [[edge_out]] to [[pm_edge]]. Below, white light [[ase]] goes through a grating
coupler [[gc_in]], the same waveguide [[wg_gc]] and [[gc_out]] to [[pm_gc]] and [[osa_gc]].
[[osa_in]] and [[pm_ase]] show what went in.

## The edge coupler: mode overlap

The fibre's mode is 10.4 µm wide; the chip's is 3 µm. The fraction that couples is the overlap
of two Gaussians, one factor per axis:

$$\eta = \left(\frac{2 w_f w_c}{w_f^2 + w_c^2}\right)^2 = 0.284 \quad (-5.47\ \text{dB})$$

Two facets in air reflect a little more, 0.31 dB, so each coupler costs 5.78 dB. With 1 dB of
waveguide, [[pm_edge]] reads 0 − 2 × 5.78 − 1 = −12.56 dBm. The loss is flat across the band.

## The grating coupler: a passband

A grating couples light up out of the chip only where its period matches the wavelength. At
1550 nm each one costs 3 dB, and the loss rises either side. It also passes TE only: the TM half
of unpolarised light is lost. So on [[osa_gc]], divided by [[osa_in]], the peak is

$$-3 - 3 - 1 - 3 = -10.0\ \text{dB}$$

near 1550 nm, and the passband of the pair is 24.7 nm wide at 1 dB down. [[pm_ase]] reads
2.10 dBm in and [[pm_gc]] −8.44 dBm out: across the whole band the pair costs 10.5 dB, more than
at its peak.

## Try this

- Set *offset_x* of [[edge_in]] to 2 µm and watch [[pm_edge]].
- Set *gap_index* of both edge couplers to 1.45, an index-matching gel. The facets stop
  reflecting.
- Change *fibre_angle* of [[gc_in]] and [[gc_out]] to 12°. The passband moves.
""".strip(),
    },
    "pdk_splitter": {
        "title": "5.6 A splitter from a foundry kit",
        "body": r"""
Two 1×4 splitters side by side, fed the same 0 dBm each by [[tx]] and [[sp]]. [[ideal]] is an
MMI as a textbook draws it. [[kit]] is the same device as a foundry's process design kit
describes it, built from `silicon_220nm.pdk.json`.

## What a kit adds

The engine knows what an MMI *is*: self-imaging divides light evenly, with the phases that keep
it lossless. It cannot know what a given process makes. A kit is that knowledge, measured on a
wafer: here 0.45 dB of excess loss and 0.35 dB of imbalance across the four arms.

| arm | [[ideal]] | [[kit]] |
| :-- | :-- | :-- |
| 1 | −6.03 dBm | −6.31 dBm |
| 2 | −6.03 dBm | −6.42 dBm |
| 3 | −6.03 dBm | −6.54 dBm |
| 4 | −6.03 dBm | −6.66 dBm |

The textbook divides by four exactly, $10\log_{10}4 = 6.02$ dB. The kit's arms are not equal,
which is what a real splitter tree does to a design margin.

## A kit has a window

Each value in a kit may be a fit in wavelength, valid only where it was measured. The kit
refuses to extrapolate: its 3 dB coupler, asked for at 1310 nm, raises an error instead of
returning a negative power fraction. `examples/python/pdk_import.py` shows that, and
`netlist_circuit.py` builds a ring from a layout netlist with the same kit.

## Try this

- Raise *imbalance* of [[kit]] to 1 dB and compare [[pm_kit1]] with [[pm_kit4]].
""".strip(),
    },
    "fbg_strain": {
        "title": "6.1 A Bragg grating as a strain gauge and a thermometer",
        "body": r"""
Two Bragg gratings sit in series on one fibre, lit by white light from [[ase]]. [[working]] is
glued to a part under 400 µε of load. [[reference]] is loose in a tube beside it. Both are
15 K warmer than when they were written. Each reflects a narrow band, read on [[osa_work]] and
[[osa_ref]]. Light neither grating reflects goes on to [[osa_thru]].

## What moves the reflection

A grating reflects where its period fits half a wavelength in the glass:

$$\lambda_B = 2\,n_\text{eff}\,\Lambda$$

Stretching the fibre lengthens $\Lambda$ and, through the photo-elastic effect, lowers
$n_\text{eff}$ a little. Warming it expands the glass and raises its index. Both are a material
number times $\lambda_B$:

$$\frac{\Delta\lambda_B}{\lambda_B} = (1 - p_e)\,\varepsilon + (\alpha + \xi)\,\Delta T$$

with $p_e = 0.22$ and $\alpha + \xi = 7.22 \times 10^{-6}$ /K. At 1545 nm that is 1.21 pm per
microstrain and 11.2 pm per kelvin.

## One wavelength, two unknowns

Divide one sensitivity by the other: 9.26 µε per kelvin. A degree of warming moves the peak as
far as 9.26 µε of load, and nothing in the spectrum tells them apart. Read as strain alone,
[[working]] reports 539 µε, not 400: the extra 139 µε is the 15 K.

## The reference grating

[[reference]] feels only the temperature, so its shift is the temperature. The peaks are read
the way an interrogator reads them, as the centroid of the top half of each reflection:

| | written at | peak at | shift |
| :-- | :-- | :-- | :-- |
| [[working]] | 1545 nm | 1545.6498 nm | 649.8 pm |
| [[reference]] | 1555 nm | 1555.1692 nm | 169.2 pm |

The reference's shift gives 15.07 K. Subtract that much drift from [[working]] and what is left
is 399.7 µε of load.

## Try this

- Set *strain* of [[working]] to 0. Its peak moves back by 0.48 nm and the reference stays put.
- Set *temperature_change* of both to 0. Now the two peaks shift by the load alone.
""".strip(),
    },
    "fbg_drop": {
        "title": "6.2 A grating and a circulator: dropping a channel",
        "body": r"""
Two channels, [[ch1550]] and [[ch1552]], share a fibre through [[mux]]. A Bragg grating [[fbg]]
reflects 1550 nm back the way it came. A circulator [[circ]] sends light from port 1 to port 2
and from port 2 to port 3, so the reflection leaves by a third fibre: the *drop* port, read on
[[pm_drop]] and [[osa_drop]]. Everything else passes the grating to [[pm_thru]] and [[osa_thru]].

## The budget

The grating reflects $R = 0.933$ ($-0.30$ dB) at its centre, over 0.198 nm. Each pass through
the circulator costs 0.7 dB, and the dropped light passes twice:

| | 1550 nm | 1552 nm |
| :-- | :-- | :-- |
| [[pm_drop]] | −1.70 dBm | −38.71 dBm |
| [[pm_thru]] | −12.43 dBm | −0.70 dBm |

−1.70 dBm is $2 \times 0.7 + 0.30$ dB. The express channel pays one pass. What the grating does
not reflect at 1550 nm, $1 - R$ or $-11.73$ dB, carries on to the express port.

## What sets the crosstalk

With an ideal circulator the 1552 nm channel reaches the drop port only through the grating's
sidelobes, at −47.36 dBm. This circulator has 40 dB of isolation, and the light it leaks from
port 1 straight to port 3 is stronger: −38.71 dBm, so the dropped channel is 37.0 dB above its
neighbour. Here the circulator, not the grating, limits the drop.

## Try this

- Set *isolation* of [[circ]] to 0, which is ideal. [[pm_drop]] at 1552 nm falls to −47.4 dBm.
- Set *bragg_wavelength* of [[fbg]] to 1552 nm. The other channel is dropped instead.
""".strip(),
    },
    "tilted_grating": {
        "title": "6.3 A tilted grating reads a liquid",
        "body": r"""
A Bragg grating written square to the fibre only reflects the core mode into itself, so it is
blind to what is outside the glass. Tilt its fringes by 4° and the core light can also couple
into backward *cladding* modes, which run along the outside of the glass and feel what the fibre
is dipped in. Each one cuts a narrow notch below the Bragg line.

The same grating is shown twice: [[tfbg_air]] in air and [[tfbg_water]] in water ($n = 1.333$).
[[osa_air]] and [[osa_water]] show what each transmits.

## Where the notches are

The core mode couples to a cladding mode $m$ where

$$\lambda_m = \left(n_\text{core} + n_{\text{clad},m}\right)\Lambda$$

with $\Lambda = 535$ nm. The Bragg line, where the core couples to itself, is the longest one, at
1551.24 nm. Cladding modes have lower indices, so their notches form a comb reaching 15 nm below
it, each about 10 dB deep.

## What water does

Water raises the index outside the glass. That pulls each cladding mode's field outwards and
raises its effective index, so its notch moves to a longer wavelength. Measured by sliding the
two traces over each other:

| part of the comb | moved by |
| :-- | :-- |
| 1536.5 to 1538.5 nm | 88 pm |
| 1546.5 to 1548.5 nm | 17 pm |
| the Bragg line | 0 pm |

The modes deepest in the comb reach furthest out of the glass and move most. The Bragg line never
leaves the core and does not move at all. Temperature moves the whole comb together, so reading
the comb against the Bragg line gives the outside index with its own temperature reference.

## Try this

- Set *surrounding_index* of [[tfbg_water]] to 1.40. The bottom of the comb moves further.
- Set *tilt* of [[tfbg_air]] to 8°. The fringes along the axis spread out by $1/\cos\theta$,
  so the Bragg line moves to 1562.6 nm, out of this window, and only the comb is left in it.

This one takes about half a minute to run: every notch is a mode solve.
""".strip(),
    },
    "lpg_pair": {
        "title": "6.4 A long-period grating, and a pair of them",
        "body": r"""
A long-period grating has a period of hundreds of microns, not half a wavelength. It couples the
core mode *forwards*, into a cladding mode, at the wavelength where

$$\lambda = \left(n_\text{core} - n_{\text{clad},m}\right)\Lambda$$

With $\Lambda = 485$ µm that is the LP04 cladding mode near 1550 nm. On bare fibre the cladding
light is lost in the coating, so the grating cuts a broad notch in the transmission.

## One grating

[[lpg_one]] is just strong enough to hand all of the core light to the cladding at the centre.
[[osa_one]] shows a notch at 1550.14 nm, more than 60 dB deep, and many nanometres wide: the core
and cladding indices part only slowly with wavelength.

## Two gratings

[[lpg_pair]] is the same grating written twice, 200 mm apart, each half as strong. The first sends
half the light into the cladding. Over the gap, core and cladding light travel at different speeds.
The second grating then puts the cladding light back into the core, where it meets the light that
stayed. That is a Mach–Zehnder interferometer inside one fibre, so the notch fills with fringes:

$$\Delta\lambda = \frac{\lambda^2}{\Delta n_g\, d}$$

[[osa_pair]] shows them 2.31 nm apart near the centre, with the deepest at 1549.93 nm. Double the
gap to 400 mm and they close to 1.24 nm. Since $d$ is a little more than the gap, the spacing does
not halve exactly.

Each fringe is a narrow notch that moves with whatever the cladding light felt in the gap, which
makes a pair a more sensitive sensor than one grating.

## Try this

- Set *separation* of [[lpg_pair]] to 400 mm and watch the fringes close up.
- Set *gap_loss* of [[lpg_pair]] to 6 dB. The cladding light no longer cancels the core light and
  the fringes wash out.
""".strip(),
    },
    "otdr": {
        "title": "6.5 An OTDR finds a break",
        "body": r"""
A 25 km fibre has been cut by a digger. An optical time-domain reflectometer (OTDR) finds where,
from one end, without anyone walking the route.

[[pulse]] sends a 100 ns pulse (FWHM) at 20 dBm peak into the fibre [[fut]]. As it travels, the glass
scatters a little of it in every direction (Rayleigh scattering), and a small part of that is
caught by the core going back. [[otdr]] shows what returns, in dBm, against the time since launch.

## Time is distance

Light scattered at a distance $z$ is back after the round trip, so

$$z = \frac{c\,t}{2\,n_g}$$

With $n_g = 1.468$, each 9.79 µs of trace is a kilometre of fibre.

## The slope is the loss

Light scattered at $z$ has crossed $z$ twice, so the trace falls by twice the fibre's loss: 0.4 dB
per km here, for 0.2 dB/km of fibre. From 1 km to 5 km it falls from −41.81 to −43.41 dBm.

How bright it starts is set by the pulse's energy $E$:

$$P(0) = E\; S\,\alpha_s\,\frac{v_g}{2}$$

with $S = 0.0017$ the share the core catches and $\alpha_s$ the scattering part of the loss. The
trace starts near −41.4 dBm, about 61 dB below the pulse.

## Events

- **A splice at 8 km.** The trace steps down by 1.0 dB: the splice's 0.5 dB, paid twice.
- **The break at 17.30 km.** The broken glass reflects a little of the pulse straight back, a spike
  well above the scatter, and then nothing returns at all. The fibre is 25 km on the drawing. The
  last echo says it now ends at 17.30 km, and that is where to dig.

A real OTDR's trace would sink into detector noise far below these levels. This one has no
detector noise, so it falls forever.

## Try this

- Set *break_at* of [[fut]] to 0. The spike moves to 25 km, the clean far end, and is brighter
  because a cleave reflects −14.7 dB.
- Set *width* of [[pulse]] to 600000 ps. The trace gets brighter but the splice step smears out:
  a long pulse trades resolution for range.
""".strip(),
    },
}
