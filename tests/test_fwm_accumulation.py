"""Four-wave mixing products adding across spans, in field rather than in power.

A link is not one fibre. The product a span generates arrives on top of the one
the span before it generated, and whether those add or cancel is decided by how
far the pumps and the product have drifted apart in phase over the fibre already
behind them. That drift is the phase mismatch integrated over the distance
travelled, and the signal carries exactly enough to compute it.

The claim has a sharp form, and it is the one this file leans on: **cutting a
lossless fibre into N pieces must give back the same product it gave in one**.
Splitting a span is not a physical change, so any answer that moves when you do
it was adding the pieces wrong. Before this, eight pieces gave an eighth of the
right answer.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from maiman.component import Component, PortType
from maiman.components import EDFA, Attenuator, Combiner, Fiber, OpticalFilter
from maiman.context import SimulationContext
from maiman.kernels import (
    attenuation_db_per_m_to_alpha,
    dispersion_to_beta2,
    effective_length,
    fwm_accumulated_phase,
    fwm_efficiency,
    fwm_mixing_integral,
    fwm_phase_mismatch,
)
from maiman.registry import lookup, registered_names
from maiman.signals import Band, OpticalSignal, joined_accumulated_gvd

ANCHOR = 193.1e12
SPACING = 100e9
SAMPLES = 256
FS = 160e9
GAMMA = 1.3

#: Microwatts. The point of these tests is the *bookkeeping* between spans, and
#: it is only visible while the closed form the bookkeeping is built on still
#: holds: at a milliwatt over 320 km the nonlinear phase is hundreds of radians,
#: the pumps are long since depleted, and every number is meaningless.
POWER = 1e-6

CTX = SimulationContext(
    bit_rate=10e9, samples_per_symbol=16, sequence_length=16, seed=3, precision="double"
)


def comb(channels: int = 3, power: float = POWER) -> OpticalSignal:
    """Unmodulated carriers on a uniform grid, so the pump powers are exact."""
    return OpticalSignal(
        bands=tuple(
            Band(
                Ex=np.full(SAMPLES, np.sqrt(power), dtype=np.complex128),
                Ey=np.zeros(SAMPLES, dtype=np.complex128),
                f0=ANCHOR + index * SPACING,
                fs=FS,
            )
            for index in range(channels)
        ),
        noise=(),
    )


def through(
    spans: int,
    *,
    total_km: float,
    dispersion: float = 0.0,
    attenuation: float = 0.0,
    amplify: bool = False,
) -> OpticalSignal:
    """``total_km`` of fibre, cut into ``spans`` equal pieces."""
    signal = comb()
    each = total_km / spans
    for index in range(spans):
        signal = Fiber(
            length=each,
            attenuation=attenuation,
            dispersion=dispersion,
            nonlinearity=GAMMA,
            # Effectively off: the product sits 90 dB down and the default floor
            # would discard the very thing being measured.
            mixing_floor=250.0,
            label=f"fib{index}",
        ).run(CTX, {"in": signal})["out"]
        if amplify:
            signal = EDFA(gain=attenuation * each, noise_figure=0.0, label=f"amp{index}").run(
                CTX, {"in": signal}
            )["out"]
    assert isinstance(signal, OpticalSignal)
    return signal


def product_power(signal: OpticalSignal, offset_ghz: float = -100.0) -> float:
    """Power in the mixing product at ``offset_ghz`` from the first channel."""
    target = ANCHOR + offset_ghz * 1e9
    for band in signal.bands:
        if abs(band.f0 - target) < 1e3:
            return band.average_power()
    raise AssertionError(f"no band at {offset_ghz:+.0f} GHz; have {len(signal.bands)} bands")


# ---------------------------------------------------------------------------
# the sharp claim


@pytest.mark.parametrize("dispersion", [0.0, 1.0, 17.0])
@pytest.mark.parametrize("spans", [2, 4, 8])
def test_cutting_a_span_into_pieces_gives_back_the_same_product(
    spans: int, dispersion: float
) -> None:
    """320 km of lossless fibre, whole and in pieces.

    There is no physics in where a span boundary is drawn, so this is an
    identity, not a trend — and it holds at every dispersion, because what makes
    the pieces add correctly is the accumulated mismatch, which is exactly what
    dispersion sets. Adding the pieces in power instead would put eight of them
    9 dB below one.

    The residue that is left comes from the split-step taking different steps in
    a short span than in a long one, and from second-generation products that the
    open floor here lets through; both are hundredths of a decibel.
    """
    whole = product_power(through(1, total_km=320.0, dispersion=dispersion))
    pieces = product_power(through(spans, total_km=320.0, dispersion=dispersion))
    difference_db = 10.0 * math.log10(pieces / whole)
    assert abs(difference_db) < 0.05, f"{spans} pieces moved the product by {difference_db:+.4f} dB"


def test_the_product_grows_as_the_square_of_the_span_count() -> None:
    """Amplified, phase matched: N spans give N² times one span's product.

    Fields add, powers do not. A link that put each span's contribution in with a
    fresh random phase would grow as N — which is what this did before, measured
    at 2.0, 3.2 and 4.4 for two, three and four spans.
    """
    one = product_power(through(1, total_km=80.0, attenuation=0.2, amplify=True))
    for spans in (2, 3, 4):
        many = product_power(through(spans, total_km=80.0 * spans, attenuation=0.2, amplify=True))
        assert many / one == pytest.approx(float(spans**2), rel=1e-3), spans


def test_with_dispersion_the_spans_interfere_instead_of_stacking() -> None:
    """And the tell is that the answer starts depending on the span length.

    Phase matched, four spans give sixteen times one whatever the spans are: the
    contributions are all in phase, and moving a boundary moves nothing. Give the
    fibre dispersion and each span's contribution arrives rotated by the mismatch
    accumulated behind it, so what four spans give is an interference — measured
    here between 0.3 and 17 times one span as the span length moves by seven
    kilometres.

    Note what is *not* claimed: that dispersion suppresses the build-up. At the
    right span length it does the opposite, because the rotation per span comes
    back round to a multiple of 2*pi and the spans stack again. That periodic
    re-phasing is a real property of a dispersion-managed link and is the reason
    the map is designed rather than chosen, so a test asserting "less than" would
    be asserting something false.
    """
    lengths = (78.0, 79.0, 80.0, 82.0, 85.0)

    def growth(km: float, dispersion: float) -> float:
        one = product_power(
            through(1, total_km=km, dispersion=dispersion, attenuation=0.2, amplify=True)
        )
        four = product_power(
            through(4, total_km=4.0 * km, dispersion=dispersion, attenuation=0.2, amplify=True)
        )
        return four / one

    matched = [growth(km, 0.0) for km in lengths]
    dispersive = [growth(km, 17.0) for km in lengths]

    assert all(value == pytest.approx(16.0, rel=1e-3) for value in matched), matched
    assert max(dispersive) / min(dispersive) > 10.0, dispersive
    assert max(dispersive) > 8.0 and min(dispersive) < 1.0, dispersive


def test_a_compensating_span_takes_the_accumulated_dispersion_back_out() -> None:
    """Negative D subtracts here on its own, which is what a dispersion map is.

    Nothing special is done for it: the path history is a sum of ``beta2 * L``
    and a span with the opposite sign contributes the opposite amount.
    """
    signal = comb()
    forward = Fiber(
        length=80.0, attenuation=0.0, dispersion=17.0, nonlinearity=0.0, label="fib"
    ).run(CTX, {"in": signal})["out"]
    assert isinstance(forward, OpticalSignal)
    assert forward.accumulated_gvd != 0.0

    back = Fiber(length=80.0, attenuation=0.0, dispersion=-17.0, nonlinearity=0.0, label="dcf").run(
        CTX, {"in": forward}
    )["out"]
    assert isinstance(back, OpticalSignal)
    assert back.accumulated_gvd == pytest.approx(0.0, abs=1e-9 * abs(forward.accumulated_gvd))


# ---------------------------------------------------------------------------
# the kernel pieces


def test_the_accumulated_phase_is_the_mismatch_times_the_distance() -> None:
    """Which is why one scalar carries it, and why it is one expression not two."""
    beta2 = dispersion_to_beta2(17e-6, 1550e-9)
    length = 80e3
    offsets = (0.0, 2 * SPACING, SPACING)

    assert fwm_accumulated_phase(beta2 * length, *offsets) == pytest.approx(
        fwm_phase_mismatch(beta2, *offsets) * length, abs=0.0, rel=1e-12
    )
    # Linear in the distance, and zero when nothing has been travelled.
    assert fwm_accumulated_phase(0.0, *offsets) == 0.0
    assert fwm_accumulated_phase(2.0 * beta2 * length, *offsets) == pytest.approx(
        2.0 * fwm_accumulated_phase(beta2 * length, *offsets), abs=0.0, rel=1e-12
    )


def test_the_efficiency_is_the_mixing_integral_squared() -> None:
    """The two must not be able to drift, so one is written in terms of the other."""
    alpha = attenuation_db_per_m_to_alpha(0.2e-3)
    for distance in (10e3, 80e3, 400e3):
        reference = effective_length(alpha, distance)
        for mismatch in (0.0, 1e-4, 1e-2, 1.0):
            integral = fwm_mixing_integral(mismatch, alpha, distance)
            assert fwm_efficiency(mismatch, alpha, distance) == pytest.approx(
                abs(integral) ** 2 / reference**2, abs=0.0, rel=1e-12
            )


def test_the_mixing_integral_hits_both_limits() -> None:
    """Phase matched it is the effective length; lossless it is ``L sinc``."""
    alpha = attenuation_db_per_m_to_alpha(0.2e-3)
    length = 80e3

    assert fwm_mixing_integral(0.0, alpha, length).real == pytest.approx(
        effective_length(alpha, length), rel=1e-12
    )
    assert fwm_mixing_integral(0.0, alpha, length).imag == pytest.approx(0.0, abs=1e-9)

    # One radian of accumulated mismatch, so the sinc is still positive and the
    # argument has not wrapped — the identity is the same either way, but the
    # test is about reading it, not about branch cuts.
    mismatch = 1.0 / length
    lossless = fwm_mixing_integral(mismatch, 0.0, length)
    assert abs(lossless) == pytest.approx(
        length * float(np.sinc(mismatch * length / (2.0 * np.pi))), rel=1e-9
    )
    # Its argument is half the accumulated mismatch, which is exactly the part the
    # efficiency throws away and a multi-span link needs.
    assert float(np.angle(lossless)) == pytest.approx(mismatch * length / 2.0, rel=1e-9)


# ---------------------------------------------------------------------------
# the path history, carried


def test_a_fresh_signal_has_travelled_nothing() -> None:
    assert OpticalSignal().accumulated_gvd == 0.0
    assert comb().accumulated_gvd == 0.0


def test_a_span_adds_beta2_times_its_length() -> None:
    """Signed the way beta2 is, so standard fibre contributes a negative number."""
    signal = comb()
    span = Fiber(length=80.0, attenuation=0.0, dispersion=17.0, nonlinearity=0.0, label="fib")
    out = span.run(CTX, {"in": signal})["out"]
    assert isinstance(out, OpticalSignal)

    expected = dispersion_to_beta2(17e-6, signal.bands[0].wavelength) * 80e3
    assert out.accumulated_gvd == pytest.approx(expected, abs=0.0, rel=1e-12)
    assert out.accumulated_gvd < 0.0

    twice = span.run(CTX, {"in": out})["out"]
    assert isinstance(twice, OpticalSignal)
    assert twice.accumulated_gvd == pytest.approx(2.0 * expected, abs=0.0, rel=1e-12)


def optical_ports(component: Component, ports: dict[str, PortType]) -> list[str]:
    return [name for name, kind in ports.items() if kind is PortType.OPTICAL]


def one_carrier(f0: float) -> Band:
    return Band(
        Ex=np.full(SAMPLES, np.sqrt(POWER), dtype=np.complex128),
        Ey=np.zeros(SAMPLES, dtype=np.complex128),
        f0=f0,
        fs=FS,
    )


def test_every_optical_block_carries_the_path_history_through() -> None:
    """Swept over the registry, because the failure is silent and one-line.

    A block that builds its output signal without passing the field along resets
    the link's history to zero, and everything downstream then behaves as though
    it were the first span. Nothing else in the suite would notice: the powers,
    the spectra and the constellations all come out exactly as before.

    Sources are exempt — light that has just been emitted has travelled nothing —
    and so is anything with no optical input to carry it from.
    """
    marker = -1.234e-21
    checked: list[str] = []
    for name in registered_names():
        component = lookup(name)()
        inputs = optical_ports(component, component.inputs)
        outputs = optical_ports(component, component.outputs)
        if not inputs or not outputs:
            continue

        if any(kind is not PortType.OPTICAL for kind in component.inputs.values()):
            # Modulators and the like need a drive; they are pinned by name below.
            continue
        # A distinct carrier per port: a combiner refuses two inputs on one
        # frequency, and rightly.
        feed: dict[str, object] = {
            port: OpticalSignal(
                bands=(one_carrier(ANCHOR + index * SPACING),),
                noise=(),
                accumulated_gvd=marker,
            )
            for index, port in enumerate(inputs)
        }
        produced = component.run(CTX, feed)
        for port in outputs:
            out = produced[port]
            assert isinstance(out, OpticalSignal)
            assert out.accumulated_gvd == pytest.approx(marker, abs=0.0, rel=1e-12), (
                f"{name}.{port} dropped the accumulated dispersion"
            )
        checked.append(name)

    assert {"EDFA", "Attenuator", "OpticalFilter", "Splitter"} <= set(checked), checked


@pytest.mark.parametrize("block", [EDFA, Attenuator, OpticalFilter])
def test_named_blocks_carry_it_too(block: type[Component]) -> None:
    """The sweep above skips anything it cannot feed; these are pinned by name."""
    marker = 7.5e-22
    signal = OpticalSignal(bands=comb(1).bands, noise=(), accumulated_gvd=marker)
    out = block(label="x").run(CTX, {"in": signal})["out"]
    assert isinstance(out, OpticalSignal)
    assert out.accumulated_gvd == pytest.approx(marker, abs=0.0, rel=1e-12)


# ---------------------------------------------------------------------------
# joining two paths


def test_joining_matching_paths_keeps_the_history() -> None:
    signal = OpticalSignal(bands=comb(1).bands, noise=(), accumulated_gvd=-3e-21)
    other = OpticalSignal(
        bands=(Band(Ex=comb(1).bands[0].Ex, Ey=comb(1).bands[0].Ey, f0=ANCHOR + SPACING, fs=FS),),
        noise=(),
        accumulated_gvd=-3e-21,
    )
    out = Combiner(2, label="mux").run(CTX, {"in0": signal, "in1": other})["out"]
    assert isinstance(out, OpticalSignal)
    assert out.accumulated_gvd == pytest.approx(-3e-21, abs=0.0, rel=1e-12)


def test_joining_different_paths_is_refused() -> None:
    """One number cannot hold two histories, and guessing which would be silent.

    The products of channels that came the long way have rotated away from their
    pumps and a freshly added channel's have not. Rather than pick, the block says
    so and names the two ways out.
    """
    travelled = OpticalSignal(bands=comb(1).bands, noise=(), accumulated_gvd=-3e-21)
    fresh = OpticalSignal(
        bands=(Band(Ex=comb(1).bands[0].Ex, Ey=comb(1).bands[0].Ey, f0=ANCHOR + SPACING, fs=FS),),
        noise=(),
    )
    with pytest.raises(ValueError, match="different amounts of fibre"):
        Combiner(2, label="mux").run(CTX, {"in0": travelled, "in1": fresh})


def test_an_empty_arm_has_no_history_to_disagree_about() -> None:
    """A combiner with nothing on one input must not refuse the other."""
    travelled = OpticalSignal(bands=comb(1).bands, noise=(), accumulated_gvd=-3e-21)
    assert joined_accumulated_gvd((travelled, OpticalSignal()), where="mux") == pytest.approx(
        -3e-21, abs=0.0, rel=1e-12
    )
    assert joined_accumulated_gvd((OpticalSignal(), OpticalSignal()), where="mux") == 0.0
