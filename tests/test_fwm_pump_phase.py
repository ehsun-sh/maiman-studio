"""Four-wave mixing with the pumps' own Kerr phase: within a span and between spans.

Self- and cross-phase modulation turn a mixing product's drive ``A_i A_j A_k*``
and the carrier it lands on at different rates, ``-gamma (P_i + P_j - P_k -
P_f)`` apart, and that shifts the mismatch. With ``pump_phase`` the fibre block
carries it in the mixing integral and, through the signal's Kerr history, from
span to span.

What holds it in place is the split-step solution of the same fibre with every
tone in one band, which has no mixing model at all -- the products simply grow
out of ``|A|^2 A``. One strong pump and one weak signal, so the products stay
small and do not mix again among themselves: then the only thing the linear
model leaves out is the Kerr phase, and the comparison measures exactly that.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from maiman.component import Component, PortType
from maiman.components import EDFA, Combiner, Fiber
from maiman.context import SimulationContext
from maiman.kernels import (
    _mixing_quadrature,
    _mixing_series,
    attenuation_db_per_m_to_alpha,
    effective_length,
    fwm_efficiency,
    fwm_mixing_integral,
    fwm_nonlinear_rate,
    fwm_phase_mismatch,
    fwm_product_power,
    kerr_rate,
    propagate_ssfm,
)
from maiman.registry import lookup, registered_names
from maiman.signals import Band, KerrHistory, OpticalSignal, joined_nonlinear_history

ANCHOR = 193.1e12
SPACING = 100e9
SAMPLES = 256
FS = 160e9
GAMMA = 1.3  # 1/W/km

CTX = SimulationContext(
    bit_rate=10e9, samples_per_symbol=16, sequence_length=16, seed=3, precision="double"
)


# ---------------------------------------------------------------------------
# The rate
# ---------------------------------------------------------------------------


def test_with_cross_phase_the_rate_is_the_textbook_four_powers() -> None:
    gamma = 1.3e-3
    rate = fwm_nonlinear_rate(
        gamma,
        (3e-3, 1e-3),
        (2e-3, 0.0),
        (1e-3, 2e-3),
        (0.5e-3, 1e-3),
        cross_phase=True,
        orthogonal_weight=2.0 / 3.0,
    )
    assert rate[0] == pytest.approx(-gamma * (3e-3 + 2e-3 - 1e-3 - 0.5e-3), rel=1e-12)
    assert rate[1] == pytest.approx(-gamma * (1e-3 + 0.0 - 2e-3 - 1e-3), rel=1e-12)


def test_the_cross_phase_of_everything_else_turns_every_carrier_alike() -> None:
    """Which is why the total drops out of the difference: checked with it left in."""
    gamma, total = 1.3e-3, (9e-3, 4e-3)
    powers = [(3e-3, 1e-3), (2e-3, 0.0), (1e-3, 2e-3), (0.0, 0.0)]
    rates = [
        kerr_rate(gamma, p, total, cross_phase=True, orthogonal_weight=2.0 / 3.0) for p in powers
    ]
    direct = fwm_nonlinear_rate(gamma, *powers, cross_phase=True, orthogonal_weight=2.0 / 3.0)
    for axis in (0, 1):
        summed = rates[0][axis] + rates[1][axis] - rates[2][axis] - rates[3][axis]
        assert summed == pytest.approx(direct[axis], rel=1e-12)


def test_without_cross_phase_each_carrier_turns_by_its_own_power() -> None:
    gamma = 1.3e-3
    rate = fwm_nonlinear_rate(
        gamma, (3e-3, 1e-3), (3e-3, 1e-3), (1e-3, 0.0), cross_phase=False, orthogonal_weight=0.5
    )
    assert rate[0] == pytest.approx(gamma * ((5e-3) + 0.5 * 2e-3), rel=1e-12)
    assert rate[1] == pytest.approx(gamma * ((2e-3) + 0.5 * 5e-3), rel=1e-12)


# ---------------------------------------------------------------------------
# The integral
# ---------------------------------------------------------------------------


def test_with_no_nonlinear_rate_the_integral_is_the_old_one() -> None:
    alpha = attenuation_db_per_m_to_alpha(0.2e-3)
    for mismatch in (0.0, 1e-4, -3e-3):
        closed = complex((np.exp(complex(-alpha, mismatch) * 80e3) - 1) / complex(-alpha, mismatch))
        assert fwm_mixing_integral(mismatch, alpha, 80e3) == pytest.approx(closed, rel=1e-12)
        assert fwm_mixing_integral(mismatch, alpha, 80e3, 0.0) == pytest.approx(closed, rel=1e-12)


@pytest.mark.parametrize(
    ("mismatch", "rate"),
    [(1e-4, -3e-5), (5e-3, 2e-4), (-2e-4, 3.5e-4), (0.0, -1e-4), (8e-3, -3.6e-4)],
)
def test_the_series_and_the_quadrature_agree(mismatch: float, rate: float) -> None:
    alpha = attenuation_db_per_m_to_alpha(0.2e-3)
    series = _mixing_series(mismatch, alpha, 80e3, rate)
    quadrature = _mixing_quadrature(mismatch, alpha, 80e3, rate)
    assert abs(series - quadrature) <= 1e-12 * abs(quadrature)


def test_lossless_the_kerr_phase_only_shifts_the_mismatch() -> None:
    for mismatch, rate in ((2e-4, -5e-5), (-1e-4, 3e-4)):
        shifted = complex(0.0, mismatch + rate)
        closed = (np.exp(shifted * 1e4) - 1.0) / shifted
        assert fwm_mixing_integral(mismatch, 0.0, 1e4, rate) == pytest.approx(closed, rel=1e-12)


def test_the_kerr_phase_can_undo_a_linear_mismatch() -> None:
    """Nonlinear phase matching: lossless, ``delta_beta = -r`` mixes as if matched."""
    mismatch, length = 4e-4, 1e4
    assert fwm_efficiency(mismatch, 0.0, length) < 0.4
    assert fwm_efficiency(mismatch, 0.0, length, -mismatch) == pytest.approx(1.0, rel=1e-12)


# ---------------------------------------------------------------------------
# Against the split-step, one span
# ---------------------------------------------------------------------------


def split_step(
    beta2: float,
    *,
    pump: float,
    signal: float,
    spacing: float,
    alpha: float,
    length: float,
    spans: int = 1,
) -> float:
    """Every tone in one band; the product at ``pump - spacing`` [W]."""
    n, fs = 2048, 1.6e12
    t = np.arange(n) / fs
    field = np.sqrt(pump) + np.sqrt(signal) * np.exp(2j * np.pi * spacing * t)
    mismatch = abs(fwm_phase_mismatch(beta2, 0.0, 0.0, spacing))
    for _ in range(spans):
        field, _ = propagate_ssfm(
            field,
            fs,
            beta2=beta2,
            gamma=GAMMA * 1e-3,
            alpha=alpha,
            distance=length,
            max_nonlinear_phase=1e-3,
            # The split-step samples the mixing integral once a step: it has to
            # resolve the mismatch as well as the Kerr phase, or it aliases.
            max_step=0.05 / max(mismatch, 1e-12),
        )
        field = field * math.exp(alpha * length / 2.0)
    spectrum = np.fft.fft(field) / n
    return float(abs(spectrum[-round(spacing / (fs / n))]) ** 2)


@pytest.mark.parametrize("dispersion", [8.0, -8.0])
def test_one_span_is_the_split_step_s_answer_only_with_the_pump_s_phase(dispersion: float) -> None:
    """20 mW over 10 km: a quarter of a radian of nonlinear phase moves it 12-29 %."""
    pump, signal, spacing, length = 20e-3, 20e-6, 50e9, 10e3
    beta2 = Fiber(dispersion=dispersion).reference_beta2(one_band())
    reference = split_step(
        beta2, pump=pump, signal=signal, spacing=spacing, alpha=0.0, length=length
    )
    mismatch = fwm_phase_mismatch(beta2, 0.0, 0.0, spacing)
    rate = fwm_nonlinear_rate(
        GAMMA * 1e-3, (pump, 0.0), (pump, 0.0), (signal, 0.0), cross_phase=True
    )[0]

    def predicted(nonlinear: float) -> float:
        return fwm_product_power(
            pump,
            pump,
            signal,
            gamma=GAMMA * 1e-3,
            alpha=0.0,
            distance=length,
            phase_mismatch=mismatch,
            degenerate=True,
            nonlinear_rate=nonlinear,
        )

    assert predicted(rate) / reference == pytest.approx(1.0, abs=0.03)
    assert abs(predicted(0.0) / reference - 1.0) > 0.1


# ---------------------------------------------------------------------------
# Against the split-step, through the fibre block and its amplifiers
# ---------------------------------------------------------------------------


def one_band() -> OpticalSignal:
    return OpticalSignal(
        bands=(
            Band(
                Ex=np.ones(SAMPLES, dtype=np.complex128),
                Ey=np.zeros(SAMPLES, dtype=np.complex128),
                f0=ANCHOR,
                fs=FS,
            ),
        )
    )


def pump_and_signal(pump: float, signal: float) -> OpticalSignal:
    return OpticalSignal(
        bands=tuple(
            Band(
                Ex=np.full(SAMPLES, np.sqrt(power), dtype=np.complex128),
                Ey=np.zeros(SAMPLES, dtype=np.complex128),
                f0=ANCHOR + index * SPACING,
                fs=FS,
            )
            for index, power in enumerate((pump, signal))
        )
    )


def link(spans: int, *, dispersion: float, pump_phase: bool, pump: float) -> float:
    """``spans`` amplified 80 km spans; the product at ``pump - SPACING`` [W]."""
    signal = pump_and_signal(pump, pump * 1e-3)
    for index in range(spans):
        signal = Fiber(
            length=80.0,
            attenuation=0.2,
            dispersion=dispersion,
            nonlinearity=GAMMA,
            mixing_floor=250.0,
            pump_phase=pump_phase,
            label=f"span{index}",
        ).run(CTX, {"in": signal})["out"]
        signal = EDFA(gain=16.0, noise_figure=0.0, label=f"amp{index}").run(CTX, {"in": signal})[
            "out"
        ]
    target = ANCHOR - SPACING
    return next(b.average_power() for b in signal.bands if abs(b.f0 - target) < 1e3)


@pytest.mark.parametrize("dispersion", [4.0, -4.0])
def test_four_amplified_spans_add_as_the_split_step_adds_them(dispersion: float) -> None:
    """20 mW: the linear model is off by an order of magnitude and more; this is not.

    Four things have to be right at once: the rate inside each span, the walk the
    drive has made against the product's carrier in the spans before, the frame
    that carrier has been turned into since, and the dispersion's own walk. The
    split-step knows none of them by name.
    """
    pump = 20e-3
    beta2 = Fiber(dispersion=dispersion).reference_beta2(one_band())
    alpha = attenuation_db_per_m_to_alpha(0.2e-3)
    reference = split_step(
        beta2, pump=pump, signal=pump * 1e-3, spacing=SPACING, alpha=alpha, length=80e3, spans=4
    )
    with_phase = link(4, dispersion=dispersion, pump_phase=True, pump=pump)
    linear = link(4, dispersion=dispersion, pump_phase=False, pump=pump)
    assert with_phase / reference == pytest.approx(1.0, abs=0.05)
    assert max(linear / reference, reference / linear) > 10.0


def test_the_flag_is_off_by_default_and_needs_mixing() -> None:
    assert not Fiber().pump_phase
    assert Fiber.param_specs()["pump_phase"].applies_when == "four_wave_mixing"


# ---------------------------------------------------------------------------
# The history the signal carries
# ---------------------------------------------------------------------------


def test_a_span_turns_each_carrier_and_the_weak_one_by_rate_times_l_eff() -> None:
    signal = pump_and_signal(20e-3, 20e-6)
    out = Fiber(length=80.0, attenuation=0.2, nonlinearity=GAMMA, label="f").run(
        CTX, {"in": signal}
    )["out"]
    length = effective_length(attenuation_db_per_m_to_alpha(0.2e-3), 80e3)
    gamma = GAMMA * 1e-3
    total = 20e-3 + 20e-6
    assert out.history_at(ANCHOR)[0] == pytest.approx(gamma * (2 * total - 20e-3) * length)
    assert out.history_at(ANCHOR + SPACING)[0] == pytest.approx(
        gamma * (2 * total - 20e-6) * length
    )
    assert out.nonlinear_history.weak[0] == pytest.approx(gamma * 2 * total * length)
    assert out.history_at(ANCHOR - 7 * SPACING) == out.nonlinear_history.weak
    assert pump_and_signal(1e-3, 1e-3).nonlinear_history == KerrHistory()


def optical_ports(component: Component, ports: dict[str, PortType]) -> list[str]:
    return [name for name, kind in ports.items() if kind is PortType.OPTICAL]


def test_every_optical_block_carries_the_kerr_history_through() -> None:
    """The same sweep that holds the dispersion history, for the Kerr one."""
    marker = KerrHistory(carriers=((ANCHOR, 0.25, -0.5),), weak=(0.125, 0.0625))
    checked: list[str] = []
    for name in registered_names():
        component = lookup(name)()
        if component.feedback_passes() > 0 or isinstance(component, Fiber):
            continue
        inputs = optical_ports(component, component.inputs)
        outputs = optical_ports(component, component.outputs)
        if not inputs or not outputs:
            continue
        if any(kind is not PortType.OPTICAL for kind in component.inputs.values()):
            continue
        feed: dict[str, object] = {
            port: OpticalSignal(
                bands=(pump_and_signal(1e-6, 1e-6).bands[0],)
                if index == 0
                else (
                    Band(
                        Ex=np.full(SAMPLES, 1e-3, dtype=np.complex128),
                        Ey=np.zeros(SAMPLES, dtype=np.complex128),
                        f0=ANCHOR + index * SPACING,
                        fs=FS,
                    ),
                ),
                nonlinear_history=marker,
            )
            for index, port in enumerate(inputs)
        }
        produced = component.run(CTX, feed)
        for port in outputs:
            out = produced[port]
            assert isinstance(out, OpticalSignal)
            assert out.nonlinear_history.weak == marker.weak, f"{name}.{port} dropped it"
            assert out.nonlinear_history.carriers == marker.carriers, f"{name}.{port} dropped it"
        checked.append(name)
    assert {"EDFA", "Attenuator", "OpticalFilter", "Splitter", "Combiner"} <= set(checked)


def test_joining_channels_is_a_union_and_a_disagreement_is_kept_not_raised() -> None:
    first = OpticalSignal(
        bands=pump_and_signal(1e-3, 1e-3).bands[:1],
        nonlinear_history=KerrHistory(carriers=((ANCHOR, 0.1, 0.0),), weak=(0.2, 0.0)),
    )
    second = OpticalSignal(
        bands=pump_and_signal(1e-3, 1e-3).bands[1:],
        nonlinear_history=KerrHistory(carriers=((ANCHOR + SPACING, 0.3, 0.0),), weak=(0.2, 0.0)),
    )
    joined = joined_nonlinear_history((first, second), where="mux")
    assert joined.carriers == ((ANCHOR, 0.1, 0.0), (ANCHOR + SPACING, 0.3, 0.0))
    assert not joined.conflict

    fresh = OpticalSignal(bands=second.bands)
    mixed = Combiner(2, label="mux").run(CTX, {"in0": first, "in1": fresh})["out"]
    assert isinstance(mixed, OpticalSignal)
    assert "different Kerr fibre" in mixed.nonlinear_history.conflict

    # Nothing asks for the history, nothing is refused ...
    Fiber(length=10.0, nonlinearity=GAMMA, label="plain").run(CTX, {"in": mixed})
    # ... until a span needs it.
    with pytest.raises(ValueError, match="pump_phase needs one history"):
        Fiber(length=10.0, nonlinearity=GAMMA, pump_phase=True, label="p").run(CTX, {"in": mixed})
