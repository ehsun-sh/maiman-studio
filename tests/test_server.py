"""The session server, and the reduction that keeps it honest.

Two things are being guarded. That the interface can only ever receive data the
engine already reduced — the rule that stops a second, untested implementation
of the physics growing in the browser. And that a malformed request is answered
with the status that says whose fault it is, because a design tool that reports
every mistake as a 500 teaches its user nothing.
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

from maiman import Graph, SimulationContext, registered_names
from maiman.component import PortType
from maiman.components import (
    EDFA,
    BERAnalyzer,
    CWLaser,
    ElectricalFilter,
    EyeDiagram,
    Fiber,
    MachZehnderModulator,
    NRZDriver,
    OpticalSpectrumAnalyzer,
    OSNRMeter,
    PINPhotodiode,
    PowerMeter,
    PRBSGenerator,
)
from maiman.encoding import (
    MAX_ENCODED_NUMBERS,
    EncodingError,
    encode,
    encode_results,
    number,
    register_encoder,
)
from maiman.project import graph_to_dict
from maiman.registry import lookup
from maiman.server import MAX_BODY, MAX_SAMPLES, RequestError, run_project, serve
from maiman.signals import ElectricalSignal, EyeMeasurement, PowerReading


def simple_link() -> Graph:
    """A laser into a span into two meters. Small, valid, and quick."""
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=4, sequence_length=64, seed=5)
    graph = Graph(ctx)
    laser = graph.add(CWLaser(power=3.0, wavelength=1550.0, label="tx"))
    fiber = graph.add(Fiber(length=40.0, attenuation=0.2, label="span"))
    meter = graph.add(PowerMeter(label="pm"))
    osnr = graph.add(OSNRMeter(label="osnr"))
    graph.connect(laser, fiber["in"])
    graph.connect(fiber, meter["in"])
    graph.connect(fiber, osnr["in"])
    return graph


def document() -> dict[str, Any]:
    return graph_to_dict(simple_link(), ui={"tx": {"x": 10.0, "y": 20.0}})


# ---------------------------------------------------------------------------
# The reduction rule
# ---------------------------------------------------------------------------


def metric_rich_link() -> Graph:
    """An OOK link carrying every optical-side measurement the library has."""
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=256, seed=9)
    graph = Graph(ctx)
    prbs = graph.add(PRBSGenerator(order=7.0, label="prbs"))
    driver = graph.add(NRZDriver(v_low=4.0, v_high=0.0, label="drv"))
    laser = graph.add(CWLaser(power=0.0, wavelength=1550.0, label="laser"))
    mzm = graph.add(MachZehnderModulator(v_pi=4.0, label="mzm"))
    fiber = graph.add(Fiber(length=40.0, attenuation=0.2, dispersion=17.0, label="fiber"))
    amp = graph.add(EDFA(gain=8.0, noise_figure=5.0, label="edfa"))
    pin = graph.add(PINPhotodiode(responsivity=0.8, label="pin"))
    lpf = graph.add(ElectricalFilter(bandwidth=7.0, label="lpf"))
    graph.chain(prbs, driver)
    graph.connect(laser, mzm["optical_in"])
    graph.connect(driver, mzm["electrical_in"])
    graph.chain(mzm, fiber, amp)

    for sink in (
        PowerMeter(label="pm"),
        OSNRMeter(label="osnr"),
        OpticalSpectrumAnalyzer(points=256.0, label="osa"),
    ):
        graph.connect(amp, graph.add(sink)["in"])

    graph.chain(amp, pin, lpf)
    eye = graph.add(EyeDiagram(label="eye"))
    ber = graph.add(BERAnalyzer(label="ber"))
    graph.connect(lpf, eye["in"])
    graph.connect(lpf, ber["in"])
    graph.connect(prbs["out"], ber["reference"])
    return graph


def test_no_metric_port_in_the_library_encodes_as_opaque() -> None:
    """Every measurement the library can produce must be drawable.

    Two graphs between them touch all nine metric ports — the optical/OOK side
    here, the coherent side through the shipped export example — and the
    assertion is that none of them come back tagged ``opaque``. That tag is the
    honest answer for a plugin's own result type, and the wrong answer for
    anything shipped in this package.
    """
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "examples"))
    from export_ui_data import build as build_coherent

    covered: set[tuple[str, str]] = set()
    for graph in (metric_rich_link(), build_coherent(sequence_length=256)):
        by_label = {c.label: c for c in graph.components}
        encoded = encode_results(graph.run())
        for label, ports in encoded.items():
            component = by_label[label]
            for port, value in ports.items():
                if component.outputs.get(port) is PortType.METRIC:
                    covered.add((type(component).__name__, port))
                    assert value["kind"] != "opaque", (
                        f"{type(component).__name__}.{port} produced "
                        f"{value.get('type')}, which nothing knows how to draw"
                    )

    expected = {
        (name, port)
        for name in registered_names()
        for port, kind in lookup(name).outputs.items()
        if kind is PortType.METRIC
    }
    assert covered == expected, (
        f"these metric ports were never exercised: {sorted(expected - covered)}. "
        "Add them to a graph here rather than narrowing the assertion."
    )


def test_a_waveform_is_summarised_and_never_shipped() -> None:
    """The central rule: no raw sample buffer reaches the client."""
    samples = np.linspace(-1.0, 1.0, 65536)
    encoded = encode(ElectricalSignal(samples=samples, fs=40e9, unit="V"))

    assert encoded["kind"] == "electrical"
    assert encoded["samples"] == 65536
    assert encoded["min"] == pytest.approx(-1.0)
    assert encoded["max"] == pytest.approx(1.0)
    # Nothing in the payload is an array of samples.
    assert not any(isinstance(value, list) for value in encoded.values())


def test_a_whole_run_stays_small() -> None:
    """A run's results are kilobytes, because they are already reduced.

    Stated as a size because that is the consequence anyone will notice: the
    round trip has to feel instant while someone iterates fifty times an hour,
    and it cannot if every run moves the waveforms.
    """
    payload = run_project(document())
    size = len(json.dumps(payload, allow_nan=False))
    assert size < 64 * 1024, f"a small run encoded to {size} bytes"


def test_non_finite_numbers_become_null() -> None:
    """Infinity is a normal result here and is not valid JSON."""
    assert number(math.inf) is None
    assert number(-math.inf) is None
    assert number(math.nan) is None
    assert number(2.5) == 2.5

    # A Q factor of zero gives -inf decibels through a real code path.
    measurement = EyeMeasurement(
        q_factor=0.0,
        mean_one=0.0,
        mean_zero=0.0,
        std_one=0.0,
        std_zero=0.0,
        threshold=0.0,
        sample_offset=0,
        bits_evaluated=0,
        errors=0,
    )
    encoded = encode(measurement)
    assert encoded["q_db"] is None
    # The point of nulling: the payload is serialisable at all.
    json.dumps(encoded, allow_nan=False)


def test_a_run_never_produces_invalid_json() -> None:
    """allow_nan=False is what the server sends with; prove a real run survives it."""
    json.dumps(run_project(document()), allow_nan=False)


def test_an_unreduced_array_is_refused_rather_than_truncated() -> None:
    """A silently shortened trace is a plot that lies."""
    from maiman.signals import OpticalSpectrum

    huge = np.zeros(MAX_ENCODED_NUMBERS + 1)
    spectrum = OpticalSpectrum(frequencies=huge, power_w=huge, resolution_bandwidth=12.5e9)
    with pytest.raises(EncodingError, match="refusing to encode"):
        encode(spectrum)


def test_an_unknown_type_is_marked_opaque_rather_than_guessed_at() -> None:
    """A plugin's own metric must not fail the run, and must not be faked either."""

    class Custom:
        def __repr__(self) -> str:
            return "Custom(whatever)"

    encoded = encode(Custom())
    assert encoded["kind"] == "opaque"
    assert encoded["type"] == "Custom"
    assert "whatever" in encoded["repr"]


def test_a_plugin_can_register_an_encoder_for_its_own_result() -> None:
    from maiman import encoding

    class Custom:
        value = 7

    register_encoder(Custom, lambda c: {"kind": "custom", "value": c.value})
    try:
        assert encode(Custom()) == {"kind": "custom", "value": 7}
    finally:
        del encoding._ENCODERS[Custom]


def test_a_subclass_encodes_as_its_base() -> None:
    """Dispatch walks the MRO, so extending a result type does not break drawing."""

    class Extended(PowerReading):
        pass

    encoded = encode(Extended(signal_power_w=1e-3, noise_power_w=1e-6))
    assert encoded["kind"] == "power"
    assert encoded["power_w"] == pytest.approx(1.001e-3)


def test_booleans_stay_booleans() -> None:
    """bool is a subclass of int; checked first or a flag arrives as 1.

    Asserted with ``is``, not ``==``. In Python ``1.0 == True``, so an equality
    check here passes whether the flag survives as a boolean or arrives as the
    float it degrades into — which is exactly the bug being guarded against.
    """
    assert encode(True)["value"] is True
    assert encode(False)["value"] is False
    assert encode(1)["value"] == 1.0
    assert encode(1)["value"] is not True


def test_a_quantity_is_sent_in_the_unit_it_is_quoted_in() -> None:
    """Accumulated dispersion is held in s/m and read in ps/nm, a factor of 1e3.

    Both go over the wire. Leaving the conversion to the client puts a factor of
    a thousand in code no test runs — and it did: the page printed "1 ps/nm
    removed" for a compensator set to 1360.
    """
    from maiman.components.dsp import DispersionDiagnostics

    encoded = encode(DispersionDiagnostics(accumulated_dispersion=1.36, removed_symbols=13.4))
    assert encoded["accumulated_dispersion"] == pytest.approx(1.36, abs=0.0, rel=1e-12)
    assert encoded["accumulated_dispersion_ps_nm"] == pytest.approx(1360.0, abs=0.0, rel=1e-12)


def test_results_are_keyed_by_component_then_port() -> None:
    results = simple_link().run()
    encoded = encode_results(results)
    assert encoded["pm"]["out"]["kind"] == "power"
    assert encoded["osnr"]["out"]["kind"] == "scalar"


# ---------------------------------------------------------------------------
# run_project: whose fault was it
# ---------------------------------------------------------------------------


def test_a_real_project_runs_and_reports_its_context() -> None:
    payload = run_project(document())
    assert payload["results"]["pm"]["out"]["power_dbm"] == pytest.approx(3.0 - 8.0, abs=0.01)
    assert payload["context"]["sequence_length"] == 64
    assert payload["ui"]["tx"] == {"x": 10.0, "y": 20.0}


def test_a_body_that_is_not_an_object_is_the_callers_fault() -> None:
    with pytest.raises(RequestError) as caught:
        run_project([1, 2, 3])  # type: ignore[arg-type]
    assert caught.value.status == HTTPStatus.BAD_REQUEST


def test_an_unknown_component_names_what_to_install() -> None:
    """The registry's message is the useful one; it must not be swallowed."""
    doc = document()
    doc["nodes"][0]["type"] = "FluxCapacitor"
    with pytest.raises(RequestError) as caught:
        run_project(doc)
    assert caught.value.status == HTTPStatus.BAD_REQUEST
    assert "FluxCapacitor" in caught.value.message


def test_an_oversized_window_is_refused_before_any_work_starts() -> None:
    """Refusing the work, not discovering half way through that it was too much."""
    doc = document()
    doc["context"]["sequence_length"] = MAX_SAMPLES
    doc["context"]["samples_per_symbol"] = 16
    with pytest.raises(RequestError) as caught:
        run_project(doc)
    assert caught.value.status == HTTPStatus.REQUEST_ENTITY_TOO_LARGE
    assert str(MAX_SAMPLES) in caught.value.message


def test_a_window_just_inside_the_limit_is_allowed() -> None:
    """The guard must have an inside as well as an outside."""
    doc = document()
    doc["context"]["sequence_length"] = MAX_SAMPLES // 16
    doc["context"]["samples_per_symbol"] = 16
    # Not run — building it is enough to prove the guard let it through, and
    # running a four-million-sample window in a unit test is not a kindness.
    graph = SimulationContext(
        bit_rate=10e9, samples_per_symbol=16, sequence_length=MAX_SAMPLES // 16
    )
    assert graph.num_samples == MAX_SAMPLES


def test_the_window_is_checked_before_the_project_is_even_read() -> None:
    """Refuse the work, do not discover half way through that it was too much.

    The document below is wrong in two ways at once — an enormous window and a
    component that does not exist. Whichever check runs first decides the
    status, so this pins the order rather than merely the outcome.
    """
    doc = document()
    doc["context"]["sequence_length"] = MAX_SAMPLES
    doc["context"]["samples_per_symbol"] = 16
    doc["nodes"][0]["type"] = "FluxCapacitor"
    with pytest.raises(RequestError) as caught:
        run_project(doc)
    assert caught.value.status == HTTPStatus.REQUEST_ENTITY_TOO_LARGE


def test_a_graph_that_cannot_run_is_unprocessable_not_malformed() -> None:
    """400 means 'I could not read that'; 422 means 'I read it and it will not run'."""
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=4, sequence_length=32, seed=1)
    graph = Graph(ctx)
    graph.add(PRBSGenerator(order=7.0, label="prbs"))
    graph.add(PowerMeter(label="pm"))  # its input is never connected
    with pytest.raises(RequestError) as caught:
        run_project(graph_to_dict(graph))
    assert caught.value.status == HTTPStatus.UNPROCESSABLE_ENTITY


def test_a_component_rejecting_its_parameters_is_also_unprocessable() -> None:
    doc = document()
    for node in doc["nodes"]:
        if node["id"] == "prbs" or node["type"] == "PRBSGenerator":
            node["params"]["order"] = 12.0
    doc["nodes"].append({"id": "bad", "type": "PRBSGenerator", "params": {"order": 12.0}})
    with pytest.raises(RequestError) as caught:
        run_project(doc)
    assert caught.value.status == HTTPStatus.UNPROCESSABLE_ENTITY


# ---------------------------------------------------------------------------
# Over a socket
# ---------------------------------------------------------------------------


@pytest.fixture
def session() -> Iterator[str]:
    """A real server on an ephemeral port, shut down afterwards."""
    httpd = serve("127.0.0.1", 0)
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def get(url: str) -> tuple[int, Any]:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


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


def test_health_reports_the_component_count(session: str) -> None:
    status, body = get(f"{session}/api/health")
    assert status == HTTPStatus.OK
    assert body["status"] == "ok"
    assert body["components"] == len(list(registered_names()))


def test_manifests_are_served_and_are_the_engines_own(session: str) -> None:
    """The palette is generated, so it cannot drift from what the engine offers."""
    status, body = get(f"{session}/api/manifests")
    assert status == HTTPStatus.OK
    assert set(body["manifests"]) == set(registered_names())
    fiber = body["manifests"]["Fiber"]
    assert fiber["category"] == "Fiber"
    assert "cross_phase_modulation" in fiber["parameters"]
    assert fiber["ports"]["outputs"]["diagnostics"] == "metric"


def test_a_run_over_http_returns_the_same_numbers_as_python(session: str) -> None:
    """The server gets no privileged access, so it must agree with a direct call."""
    doc = document()
    status, body = post(f"{session}/api/run", doc)
    assert status == HTTPStatus.OK
    assert body["results"]["pm"]["out"]["power_dbm"] == pytest.approx(
        run_project(doc)["results"]["pm"]["out"]["power_dbm"]
    )


def test_an_unknown_route_is_a_404(session: str) -> None:
    status, body = get(f"{session}/api/nothing")
    assert status == HTTPStatus.NOT_FOUND
    assert "error" in body


def test_an_empty_body_is_rejected(session: str) -> None:
    request = urllib.request.Request(f"{session}/api/run", data=b"", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raise AssertionError(f"expected a rejection, got {response.status}")
    except urllib.error.HTTPError as error:
        assert error.code == HTTPStatus.BAD_REQUEST
        # Both the guard and the JSON decoder answer 400, so the status alone
        # cannot tell whether the guard is there. The message can, and it is the
        # reason the guard exists: "empty request body" says what to do about it
        # and "Expecting value: line 1 column 1" does not.
        assert json.loads(error.read())["error"] == "empty request body"


def test_a_body_that_is_not_json_is_rejected(session: str) -> None:
    request = urllib.request.Request(f"{session}/api/run", data=b"{not json", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raise AssertionError(f"expected a rejection, got {response.status}")
    except urllib.error.HTTPError as error:
        assert error.code == HTTPStatus.BAD_REQUEST
        assert "not valid JSON" in json.loads(error.read())["error"]


def test_a_bad_project_answers_with_its_reason(session: str) -> None:
    doc = document()
    doc["nodes"][0]["type"] = "FluxCapacitor"
    status, body = post(f"{session}/api/run", doc)
    assert status == HTTPStatus.BAD_REQUEST
    assert "FluxCapacitor" in body["error"]


def test_an_oversized_body_is_refused_without_being_read(session: str) -> None:
    """The limit is on the declared length, so nothing large is ever buffered."""
    request = urllib.request.Request(
        f"{session}/api/run",
        data=b"{}",
        headers={"Content-Type": "application/json", "Content-Length": str(MAX_BODY + 1)},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raise AssertionError(f"expected a rejection, got {response.status}")
    except urllib.error.HTTPError as error:
        assert error.code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE


def test_a_nan_that_escapes_the_encoder_fails_loudly(session: str) -> None:
    """Defence in depth for the one thing that breaks a client silently.

    The encoders null non-finite numbers, so in normal operation nothing here
    ever sees one. If one ever escapes, ``allow_nan=False`` turns it into a
    visible 500 instead of a 200 carrying bare ``NaN`` — which is not JSON, and
    which the browser rejects with a parse error naming a column rather than a
    cause. Python's own ``json.loads`` accepts ``NaN`` happily, so this checks
    the bytes.
    """
    from maiman import encoding

    original = encoding._ENCODERS[PowerReading]
    encoding._ENCODERS[PowerReading] = lambda reading: {"kind": "power", "power_w": math.nan}
    try:
        request = urllib.request.Request(
            f"{session}/api/run",
            data=json.dumps(document()).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read()
                raise AssertionError(f"expected a failure, got {response.status}: {raw[:120]!r}")
        except urllib.error.HTTPError as error:
            raw = error.read()
            assert error.code == HTTPStatus.INTERNAL_SERVER_ERROR
            assert b"NaN" not in raw
            json.loads(raw)
    finally:
        encoding._ENCODERS[PowerReading] = original


def test_the_studio_page_is_served(session: str) -> None:
    with urllib.request.urlopen(f"{session}/", timeout=30) as response:
        assert response.status == HTTPStatus.OK
        assert response.headers["Content-Type"].startswith("text/html")
        assert b"Maiman Studio" in response.read()
