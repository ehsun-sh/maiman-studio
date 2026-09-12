"""The ITU grids, and the two blocks that sit on them.

The roadmap has promised "DWDM MUX/DEMUX with crosstalk" since Phase 3 and the
pieces were always here — a combiner that says it is a multiplexer, a filter that
says it is a demultiplexer. What was missing was the grid they sit on and the
crosstalk that falls out of it, and the claim worth testing hardest is that a
demultiplexer's port is *the same filter* as the single-filter block rather than
a second implementation of one.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from maiman import Graph, SimulationContext
from maiman.components import (
    CWLaser,
    Demultiplexer,
    Multiplexer,
    OpticalFilter,
    PowerMeter,
)
from maiman.grid import (
    CWDM_CHANNELS,
    DWDM_ANCHOR,
    channel_frequencies,
    cwdm_frequencies,
    cwdm_wavelengths,
    dwdm_frequencies,
    dwdm_wavelengths,
    wavelength_spacing,
)
from maiman.units import C_LIGHT, wavelength_to_frequency


@pytest.fixture
def ctx() -> SimulationContext:
    return SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=64)


# --------------------------------------------------------------------------
# The grids
# --------------------------------------------------------------------------


def test_the_dense_grid_is_anchored_where_the_standard_anchors_it() -> None:
    """193.1 THz, which is 1552.5244 nm — the number G.694.1 prints beside it."""
    assert dwdm_frequencies(1)[0] == DWDM_ANCHOR
    assert dwdm_wavelengths(1)[0] * 1e9 == pytest.approx(1552.5244, abs=1e-4)


@pytest.mark.parametrize("spacing", [12.5e9, 25e9, 50e9, 100e9, 200e9])
def test_the_dense_grid_is_exactly_uniform_in_frequency(spacing: float) -> None:
    """Every channel an exact multiple of the spacing from the anchor, which is
    what makes a 50 GHz plan a superset of a 100 GHz one rather than an offset
    copy of it."""
    frequencies = np.array(dwdm_frequencies(9, spacing=spacing, first=-4))
    assert np.diff(frequencies) == pytest.approx(np.full(8, spacing), rel=1e-12)
    # The anchor is on the grid, not between two of its channels.
    assert DWDM_ANCHOR in frequencies


def test_the_dense_grid_is_not_uniform_in_wavelength() -> None:
    """The thing that catches people out, and the reason this module exists.

    Stepping 0.8 nm to build a "100 GHz grid" drifts off ITU by most of a channel
    across the C band: the real spacing runs 0.781 nm at 1530 to 0.817 at 1565.
    """
    for nanometres, expected in ((1530.0, 0.7808), (1550.0, 0.8014), (1565.0, 0.8170)):
        step = wavelength_spacing(C_LIGHT / (nanometres * 1e-9), 100e9)
        assert step * 1e9 == pytest.approx(expected, abs=1e-4)

    # And measured off the grid itself rather than off the formula.
    wavelengths = np.array(dwdm_wavelengths(45, spacing=100e9, first=-22))
    steps = np.abs(np.diff(wavelengths))
    assert steps.min() * 1e9 == pytest.approx(0.7864, abs=2e-3)
    assert steps.max() * 1e9 == pytest.approx(0.8222, abs=2e-3)
    # Wavelength runs backwards against frequency.
    assert np.all(np.diff(wavelengths) < 0.0)


def test_the_coarse_grid_is_the_other_way_round() -> None:
    """Eighteen channels, 1271 to 1611 nm, exactly 20 nm apart *in wavelength* —
    and therefore not equally spaced in frequency at all."""
    wavelengths = np.array(cwdm_wavelengths())
    assert len(wavelengths) == CWDM_CHANNELS == 18
    assert wavelengths[0] * 1e9 == pytest.approx(1271.0)
    assert wavelengths[-1] * 1e9 == pytest.approx(1611.0)
    assert np.diff(wavelengths) * 1e9 == pytest.approx(np.full(17, 20.0), abs=1e-9)

    steps = np.abs(np.diff(np.array(cwdm_frequencies())))
    # 3.654 THz at the blue end and 2.339 at the red: a 56 % spread, which is
    # what a wavelength grid looks like from the frequency side.
    assert steps.max() / steps.min() == pytest.approx(1.562, abs=0.01)


def test_a_spacing_the_standard_does_not_define_is_refused_by_name() -> None:
    """A 75 GHz grid is a reasonable thing to simulate and is not an ITU grid, so
    the function named after the standard declines and names the one that will."""
    with pytest.raises(ValueError, match=r"not a G.694.1 spacing"):
        dwdm_frequencies(4, spacing=75e9)
    # And the general one builds it without claiming anything.
    assert np.diff(np.array(channel_frequencies(4, spacing=75e9))) == pytest.approx(
        np.full(3, 75e9)
    )


def test_the_coarse_grid_stops_where_the_standard_stops() -> None:
    """1631 nm is not a CWDM channel, it is past the end, and continuing the
    arithmetic would invent one."""
    with pytest.raises(ValueError, match="defines 18 channels"):
        cwdm_wavelengths(19)
    with pytest.raises(ValueError, match="defines 18 channels"):
        cwdm_wavelengths(4, first=16)
    with pytest.raises(ValueError, match="starts at channel 0"):
        cwdm_wavelengths(4, first=-1)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: channel_frequencies(0, spacing=100e9), "at least one channel"),
        (lambda: channel_frequencies(4, spacing=0.0), "spacing must be positive"),
        (lambda: channel_frequencies(4, spacing=100e9, first=-10_000), "not a frequency"),
        (lambda: wavelength_spacing(0.0, 100e9), "frequency must be positive"),
    ],
)
def test_the_grid_refuses_what_it_cannot_describe(call: Callable[[], object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()


# --------------------------------------------------------------------------
# The blocks
# --------------------------------------------------------------------------


def test_a_demultiplexer_port_is_the_single_filter_block(ctx: SimulationContext) -> None:
    """The claim the whole refactor exists for, asserted sample for sample.

    A multi-port device whose per-channel response drifted from the single-filter
    block's would be the worst kind of disagreement: both would look right alone,
    and the crosstalk number would depend on which one happened to be used.
    """
    demux = Demultiplexer(4, first_wavelength=1550.0, spacing=100.0, bandwidth=50.0)
    channel = 2
    centre_nm = demux.channel_wavelengths()[channel] * 1e9
    separate = OpticalFilter(
        center_wavelength=centre_nm,
        bandwidth=50.0,
        order=demux.order,
        insertion_loss=demux.insertion_loss,
        extinction=demux.extinction,
    )

    graph = Graph(ctx)
    laser = graph.add(CWLaser(power=0.0, wavelength=1550.0, label="laser"))
    graph.add(demux)
    graph.add(separate)
    through_demux = graph.add(PowerMeter(label="demux_port"))
    through_filter = graph.add(PowerMeter(label="one_filter"))
    graph.connect(laser, demux["in"])
    graph.connect(laser, separate["in"])
    graph.connect(demux[f"out{channel}"], through_demux["in"])
    graph.connect(separate, through_filter["in"])

    results = graph.run(keep=[demux, separate])
    from_demux = results.port(demux, f"out{channel}")
    from_filter = results.port(separate, "out")

    assert len(from_demux.bands) == len(from_filter.bands) == 1
    assert from_demux.bands[0].Ex == pytest.approx(from_filter.bands[0].Ex, abs=1e-18)
    assert results[through_demux].power_dbm == pytest.approx(
        results[through_filter].power_dbm, abs=1e-12
    )


def build_link(
    ctx: SimulationContext,
    *,
    channels: int = 4,
    spacing: float = 100.0,
    bandwidth: float = 50.0,
    extinction: float = 40.0,
    insertion_loss: float = 4.0,
) -> tuple[Graph, Multiplexer, Demultiplexer, list[PowerMeter]]:
    """One transmitter per channel, a mux, a demux, and a meter on every port."""
    settings = {
        "first_wavelength": 1550.0,
        "spacing": spacing,
        "bandwidth": bandwidth,
        "extinction": extinction,
        "insertion_loss": insertion_loss,
    }
    graph = Graph(ctx)
    mux = Multiplexer(channels, label="mux", **settings)
    demux = Demultiplexer(channels, label="demux", **settings)
    graph.add(mux)
    graph.add(demux)

    for index, wavelength in enumerate(mux.channel_wavelengths()):
        laser = graph.add(CWLaser(power=0.0, wavelength=wavelength * 1e9, label=f"tx{index}"))
        graph.connect(laser, mux[f"in{index}"])
    graph.connect(mux, demux["in"])

    meters = [graph.add(PowerMeter(label=f"rx{index}")) for index in range(channels)]
    for index, meter in enumerate(meters):
        graph.connect(demux[f"out{index}"], meter["in"])
    return graph, mux, demux, meters


def test_every_channel_comes_back_on_its_own_port(ctx: SimulationContext) -> None:
    """The round trip, and the loss is exactly two passes of the declared figure."""
    graph, mux, _demux, meters = build_link(ctx, channels=4)
    results = graph.run()
    wavelengths = [w * 1e9 for w in mux.channel_wavelengths()]

    for index, meter in enumerate(meters):
        reading = results[meter]
        wanted = min(reading.bands, key=lambda b: abs(b.wavelength_nm - wavelengths[index]))
        assert wanted.wavelength_nm == pytest.approx(wavelengths[index], abs=1e-3)
        # 0 dBm launched, 4 dB through the mux and 4 dB through the demux.
        assert wanted.power_dbm == pytest.approx(-8.0, abs=1e-4)


def test_crosstalk_is_produced_rather_than_declared(ctx: SimulationContext) -> None:
    """Narrow the spacing and the neighbours arrive, on their own.

    At 100 and 50 GHz a 50 GHz passband's skirt is already below the extinction
    floor, so the floor is what is measured — which is the physically right
    answer and the reason the floor exists. At 25 GHz the passband is twice the
    spacing and the neighbour comes through at half power, which is a channel
    plan that does not work rather than a model that does not.
    """
    measured = {}
    for spacing in (100.0, 50.0, 25.0):
        graph, mux, _, meters = build_link(ctx, channels=4, spacing=spacing)
        results = graph.run()
        wavelengths = [w * 1e9 for w in mux.channel_wavelengths()]
        reading = results[meters[1]]
        wanted = min(reading.bands, key=lambda b: abs(b.wavelength_nm - wavelengths[1]))
        neighbours = [b for b in reading.bands if b is not wanted]
        measured[spacing] = max(b.power_dbm for b in neighbours) - wanted.power_dbm

    assert measured[100.0] == pytest.approx(-40.0, abs=0.1)
    assert measured[50.0] == pytest.approx(-40.0, abs=0.1)
    assert measured[25.0] == pytest.approx(-3.01, abs=0.1)
    assert measured[25.0] > measured[50.0]


def test_the_skirt_is_what_rejects_a_neighbour_when_the_floor_is_lifted(
    ctx: SimulationContext,
) -> None:
    """With ``extinction = 0`` there is no floor, so what is left is the
    super-Gaussian's own skirt — and it moves with the spacing, which is the
    point of the number being computed rather than declared.

    A 25 % change in spacing is 142 dB of rejection, because a third-order
    super-Gaussian falls as the sixth power of detuning.

    **The regime had to be chosen, and both ways of getting it wrong are worth
    recording.** Too far out and the skirt underflows to exactly zero in double
    precision — ``exp(-2839)`` at a hundred gigahertz from a 50 GHz passband —
    and the rejection reports as infinite, which is the reason ``extinction``
    exists at all: no real filter's skirts fall without limit, and what
    accumulates down a chain is the floor rather than the skirt. Further out
    still and it is the *storage* that runs out first: the fields are complex64,
    and a rejection past about -600 dB puts the sample below the smallest float32
    at 1.2e-38. Neither is a physics limit and neither should be mistaken for one.
    """
    rejections = []
    for spacing in (40.0, 50.0):
        graph, mux, _, meters = build_link(
            ctx, channels=3, spacing=spacing, bandwidth=50.0, extinction=0.0
        )
        results = graph.run()
        wavelengths = [w * 1e9 for w in mux.channel_wavelengths()]
        reading = results[meters[1]]
        wanted = min(reading.bands, key=lambda b: abs(b.wavelength_nm - wavelengths[1]))
        neighbours = [b for b in reading.bands if b is not wanted]
        rejections.append(max(b.power_dbm for b in neighbours) - wanted.power_dbm)

    assert rejections[0] == pytest.approx(-50.5, abs=2.0)
    assert rejections[1] == pytest.approx(-192.7, abs=5.0)
    assert np.isfinite(rejections).all()


def test_the_router_loss_does_not_grow_with_the_channel_count(
    ctx: SimulationContext,
) -> None:
    """An arrayed-waveguide grating routes; it does not divide power N ways.

    A broadcast-and-select demux would cost 10*log10(N) — 6 dB at four channels
    and 12 at sixteen — and that device is a Splitter in front of filters, which
    this deliberately is not.
    """
    kept = []
    for channels in (2, 4, 16):
        graph, mux, _, meters = build_link(ctx, channels=channels)
        results = graph.run()
        wavelengths = [w * 1e9 for w in mux.channel_wavelengths()]
        reading = results[meters[0]]
        wanted = min(reading.bands, key=lambda b: abs(b.wavelength_nm - wavelengths[0]))
        kept.append(wanted.power_dbm)

    assert kept == pytest.approx([-8.0, -8.0, -8.0], abs=1e-4)


def test_a_multiplexer_filters_its_inputs(ctx: SimulationContext) -> None:
    """Which is what makes it a multiplexer rather than a combiner.

    A transmitter tuned to the wrong channel does not simply appear on the line:
    it lands on a port whose passband rejects it, by the same figure the demux at
    the far end would have applied.
    """
    graph = Graph(ctx)
    mux = Multiplexer(2, first_wavelength=1550.0, spacing=100.0, bandwidth=50.0, label="mux")
    graph.add(mux)
    wavelengths = [w * 1e9 for w in mux.channel_wavelengths()]

    right = graph.add(CWLaser(power=0.0, wavelength=wavelengths[0], label="right"))
    # Tuned to channel 0's wavelength but plugged into channel 1's port.
    wrong = graph.add(CWLaser(power=0.0, wavelength=wavelengths[0] + 0.4, label="wrong"))
    meter = graph.add(PowerMeter(label="line"))
    graph.connect(right, mux["in0"])
    graph.connect(wrong, mux["in1"])
    graph.connect(mux, meter["in"])

    results = graph.run()
    by_wavelength = {round(b.wavelength_nm, 3): b.power_dbm for b in results[meter].bands}
    on_channel = by_wavelength[round(wavelengths[0], 3)]
    off_channel = by_wavelength[round(wavelengths[0] + 0.4, 3)]
    assert on_channel == pytest.approx(-4.0, abs=1e-4)
    assert off_channel < on_channel - 30.0


def test_a_multiplexer_refuses_two_transmitters_on_one_channel(
    ctx: SimulationContext,
) -> None:
    """For the reason Combiner refuses them: co-located carriers interfere and
    have to be added as fields on a common grid."""
    graph = Graph(ctx)
    mux = Multiplexer(2, first_wavelength=1550.0, spacing=100.0, label="mux")
    graph.add(mux)
    for port in ("in0", "in1"):
        laser = graph.add(CWLaser(power=0.0, wavelength=1550.0, label=f"tx_{port}"))
        graph.connect(laser, mux[port])
    meter = graph.add(PowerMeter(label="line"))
    graph.connect(mux, meter["in"])

    with pytest.raises(ValueError, match="Co-located carriers"):
        graph.run()


@pytest.mark.parametrize("block", [Multiplexer, Demultiplexer])
def test_a_grid_with_no_channels_is_refused(block: type) -> None:
    with pytest.raises(ValueError, match="at least one channel"):
        block(0)


def test_the_blocks_report_the_grid_they_are_on() -> None:
    """So a project can put its lasers where its mux expects them, rather than
    the two being set from arithmetic done twice."""
    mux = Multiplexer(4, first_wavelength=1552.5244, spacing=100.0)
    frequencies = mux.channel_frequencies()
    assert frequencies[0] == pytest.approx(wavelength_to_frequency(1552.5244e-9), rel=1e-12)
    assert np.diff(np.array(frequencies)) == pytest.approx(np.full(3, 100e9), rel=1e-9)
    # Descending in wavelength, as a frequency grid is.
    assert np.all(np.diff(np.array(mux.channel_wavelengths())) < 0.0)
