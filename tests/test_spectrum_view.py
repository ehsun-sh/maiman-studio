"""A block's scattering matrix across a window: the studio's S-matrix pane, and what feeds it.

The pane draws what :meth:`ScatteringDevice.spectrum` returns and nothing else,
so the physics is checked there, where a script would call it: a straight
waveguide's group delay is its transit time ``n_g L / c`` at every wavelength, a
Bragg grating's transmission far from its line is delayed by one pass through
it and its reflection peaks on the line. Where an entry is too small to have a
phase, its group delay is left undefined rather than drawn as a spike -- the
first version of the pane drew those spikes, and they flattened everything else.

Then the endpoint: that it is a thin wrapper, answering 400 for a block with no
spectrum and a bad window, and 422 for one that will not build.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from collections.abc import Iterator
from http import HTTPStatus
from typing import Any

import numpy as np
import pytest

from maiman import manifests
from maiman.components import FiberBraggGrating, RingResonator, Waveguide
from maiman.components.photonic import ScatteringDevice
from maiman.project import component_from_dict
from maiman.registry import _REGISTRY
from maiman.server import STUDIO, serve
from maiman.units import C_LIGHT

# ---------------------------------------------------------------------------
# The engine side
# ---------------------------------------------------------------------------


def test_the_manifest_says_which_blocks_have_a_spectrum() -> None:
    catalogue = manifests()
    for name, cls in _REGISTRY.items():
        assert catalogue[name]["spectral"] is issubclass(cls, ScatteringDevice), name
    assert catalogue["FiberBraggGrating"]["spectral"]
    assert not catalogue["Fiber"]["spectral"]


def test_a_straight_waveguide_is_delayed_by_its_transit_time_at_every_wavelength() -> None:
    guide = Waveguide(label="wg")
    spectrum = guide.spectrum(1.5e-6, 1.6e-6, points=201)
    assert spectrum.pairs() == [("out", "in"), ("in", "out")]
    delay = spectrum.group_delay("out", "in")
    assert np.all(np.isfinite(delay))
    assert delay == pytest.approx(np.full(delay.size, guide.group_delay()), rel=1e-3)


def test_a_grating_reflects_on_its_line_and_delays_what_it_passes_by_one_transit() -> None:
    grating = FiberBraggGrating(label="fbg")
    spectrum = grating.spectrum(points=801)
    reflected = spectrum.power_db("in", "in")
    step = float(spectrum.wavelengths[1] - spectrum.wavelengths[0])
    peak = float(spectrum.wavelengths[int(np.argmax(reflected))])
    assert peak == pytest.approx(grating.sensed_bragg_wavelength(), abs=step)
    assert float(np.max(reflected)) == pytest.approx(
        10 * math.log10(grating.peak_reflectivity()), abs=0.05
    )

    through = spectrum.group_delay("out", "in")
    transit = grating.n_eff * grating.si("length") / C_LIGHT
    assert through[0] == pytest.approx(transit, rel=0.01), "far from the line: one pass"
    assert float(np.nanmax(through)) > 1.3 * transit, "slow light at the band edge"


def test_a_delay_is_left_undefined_where_there_is_nothing_to_have_a_phase() -> None:
    spectrum = FiberBraggGrating(label="fbg").spectrum(points=801)
    power = spectrum.power_db("in", "in")
    delay = spectrum.group_delay("in", "in")
    deep = power < float(power.max()) - 40.0
    assert deep.any()
    assert np.all(np.isnan(delay[deep]))
    assert np.all(np.isfinite(delay[~deep]))


def test_each_device_offers_the_window_it_is_worth_looking_at() -> None:
    grating = FiberBraggGrating(label="fbg")
    low, high = grating.spectral_window()
    assert low < grating.sensed_bragg_wavelength() < high
    assert high - low == pytest.approx(4.0 * grating.bandwidth(), rel=1e-12)

    ring = RingResonator(label="ring")
    low, high = ring.spectral_window()
    free = ring.si("reference_wavelength") ** 2 * ring.free_spectral_range() / C_LIGHT
    assert (high - low) / free == pytest.approx(4.0, rel=1e-12)


def test_what_is_not_a_window_is_refused() -> None:
    guide = Waveguide(label="wg")
    with pytest.raises(ValueError, match="shorter wavelength"):
        guide.spectrum(1.6e-6, 1.5e-6)
    with pytest.raises(ValueError, match="points"):
        guide.spectrum(points=1)
    with pytest.raises(ValueError, match="polarization"):
        guide.spectrum(polarization="x")


def test_a_node_is_built_the_way_a_project_builds_it() -> None:
    component = component_from_dict(
        {"id": "fbg", "type": "FiberBraggGrating", "params": {"length": 20.0}}
    )
    assert isinstance(component, FiberBraggGrating)
    assert component.label == "fbg"
    assert component.si("length") == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# The endpoint
# ---------------------------------------------------------------------------


@pytest.fixture
def session() -> Iterator[str]:
    httpd = serve("127.0.0.1", 0)
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def post(url: str, payload: Any) -> tuple[int, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_the_endpoint_returns_every_entry_as_power_phase_and_delay(session: str) -> None:
    status, body = post(
        f"{session}/api/spectrum",
        {"node": {"id": "fbg", "type": "FiberBraggGrating"}, "points": 201},
    )
    assert status == HTTPStatus.OK
    assert body["ports"] == ["in", "out"]
    assert len(body["wavelength_nm"]) == 201
    assert {(t["to"], t["from"]) for t in body["traces"]} == {
        ("in", "in"),
        ("out", "in"),
        ("in", "out"),
        ("out", "out"),
    }
    reflection = next(t for t in body["traces"] if t["to"] == "in" and t["from"] == "in")
    assert None in reflection["group_delay_ps"], "undefined points travel as null"
    direct = FiberBraggGrating(label="fbg").spectrum(points=201)
    assert reflection["power_db"] == pytest.approx(direct.power_db("in", "in").tolist(), abs=1e-9)


def test_the_window_can_be_moved(session: str) -> None:
    status, body = post(
        f"{session}/api/spectrum",
        {"node": {"id": "wg", "type": "Waveguide"}, "from_nm": 1300, "to_nm": 1310, "points": 11},
    )
    assert status == HTTPStatus.OK
    assert body["wavelength_nm"][0] == pytest.approx(1300.0)
    assert body["wavelength_nm"][-1] == pytest.approx(1310.0)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"node": {"id": "f", "type": "Fiber"}}, HTTPStatus.BAD_REQUEST),
        (
            {"node": {"id": "wg", "type": "Waveguide"}, "from_nm": 1600, "to_nm": 1500},
            HTTPStatus.BAD_REQUEST,
        ),
        ({"node": {"id": "wg", "type": "Waveguide"}, "points": 10**6}, HTTPStatus.BAD_REQUEST),
        ({"node": {"id": "x", "type": "NoSuchBlock"}}, HTTPStatus.UNPROCESSABLE_ENTITY),
        (
            {"node": {"id": "wg", "type": "Waveguide", "params": {"length": -5.0}}},
            HTTPStatus.UNPROCESSABLE_ENTITY,
        ),
        ({"nodes": []}, HTTPStatus.BAD_REQUEST),
    ],
)
def test_a_request_that_cannot_be_answered_says_whose_fault_it_is(
    session: str, payload: dict[str, Any], expected: HTTPStatus
) -> None:
    status, body = post(f"{session}/api/spectrum", payload)
    assert status == expected, body
    assert body.get("error")


def test_the_studio_has_the_pane_and_calls_the_endpoint() -> None:
    page = STUDIO.read_text(encoding="utf-8")
    assert 'data-pane="smx"' in page
    assert '"/api/spectrum"' in page
    assert "Show its S-matrix spectrum" in page
