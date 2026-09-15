"""The DWDM template: eight channels on the ITU grid, one span, one channel received.

It is the link the studio's File menu opens, so what it claims on screen has to
be what it does. These pin the grid it sits on and the numbers it produces, from
the same script that writes the project file.
"""

from __future__ import annotations

import dwdm_link
import pytest

from maiman import Graph
from maiman.components import CWLaser, Demultiplexer, Fiber, Multiplexer
from maiman.grid import dwdm_frequencies
from maiman.project import graph_to_dict
from maiman.signals import EyeMeasurement, PowerReading
from maiman.units import wavelength_to_frequency


@pytest.fixture(scope="module")
def graph() -> Graph:
    return dwdm_link.build()


@pytest.fixture(scope="module")
def results(graph: Graph) -> dict[str, object]:
    return {label: value for (label, _), value in graph.run().items()}


def blocks(graph: Graph, kind: type) -> list:
    return [c for c in graph.components if isinstance(c, kind)]


def test_the_channels_sit_on_the_itu_grid(graph: Graph) -> None:
    """Every laser on its multiplexer port, and every port a G.694.1 channel."""
    (mux,) = blocks(graph, Multiplexer)
    (demux,) = blocks(graph, Demultiplexer)
    assert mux.channel_frequencies() == demux.channel_frequencies()

    itu = set(round(f / 1e9, 3) for f in dwdm_frequencies(64, spacing=100e9))
    assert {round(f / 1e9, 3) for f in mux.channel_frequencies()} <= itu

    lasers = sorted(blocks(graph, CWLaser), key=lambda laser: laser.label)
    tuned = [wavelength_to_frequency(laser.si("wavelength")) for laser in lasers]
    assert tuned == pytest.approx(list(mux.channel_frequencies()), rel=0.0, abs=1.0)


def test_the_spool_takes_the_span_s_dispersion_back_out(graph: Graph) -> None:
    fibres = {fibre.label: fibre for fibre in blocks(graph, Fiber)}
    accumulated = sum(f.dispersion * f.length for f in fibres.values())
    assert accumulated == pytest.approx(0.0, abs=1e-9)
    assert fibres["span"].nonlinearity > 0.0, "the Kerr effect is on, as a DWDM link needs"
    assert fibres["span"].cross_phase_modulation


def test_the_link_measures_what_the_template_shows(results: dict[str, object]) -> None:
    """OSNR, crosstalk and errors, as ``python examples/dwdm_link.py`` prints them."""
    assert results["osnr"] == pytest.approx(33.04, abs=0.05)

    drop = results["drop_power"]
    assert isinstance(drop, PowerReading)
    levels = sorted(band.power_dbm for band in drop.bands)
    assert levels[-1] == pytest.approx(-0.71, abs=0.05)
    assert levels[-1] - levels[-2] == pytest.approx(40.0, abs=0.1), (
        "the neighbour leaks at the demultiplexer's extinction floor"
    )

    eye = results["ber"]
    assert isinstance(eye, EyeMeasurement)
    assert eye.errors == 0
    assert eye.bits_evaluated == 504
    assert eye.q_factor > 20.0


def test_the_saved_project_is_the_graph_the_script_builds(graph: Graph) -> None:
    import json

    on_disk = json.loads(dwdm_link.PROJECT.read_text(encoding="utf-8"))
    built = graph_to_dict(graph, ui=dwdm_link.LAYOUT)
    assert on_disk["nodes"] == built["nodes"]
    assert sorted(map(str, on_disk["edges"])) == sorted(map(str, built["edges"]))
    assert on_disk["context"] == built["context"]
