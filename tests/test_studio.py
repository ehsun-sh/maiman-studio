"""The studio page, where it makes claims the engine has to back.

The page is a single file with its data baked in, which is what lets it be
opened from a checkout with nothing running. The cost of that is a copy, and a
copy can go stale — twice in one afternoon it did: once carrying a palette two
components short, once missing the project document entirely. Nothing noticed
either time, because nothing was looking.

These are the things that must stay true of it, checked from the file rather
than from a browser: what it draws is a real graph, that graph runs, and every
block on it has somewhere to be drawn.
"""

from __future__ import annotations

import base64
import json
import math
import re
from pathlib import Path
from typing import Any

import pytest

from maiman import registered_names
from maiman.server import run_project

ROOT = Path(__file__).resolve().parent.parent
STUDIO = ROOT / "src" / "maiman" / "studio" / "index.html"
EXPORT = ROOT / "examples" / "ui_data.json"
DATA_TAG = '<script id="maiman-data" type="application/json">'


def embedded() -> Any:
    """The JSON the page carries, parsed the way the page parses it."""
    text = STUDIO.read_text(encoding="utf-8")
    start = text.index(DATA_TAG) + len(DATA_TAG)
    end = text.index("</script>", start)
    return json.loads(text[start:end])


def test_the_page_carries_the_current_export() -> None:
    """The baked-in copy and the file it was baked from must agree.

    They are the same bytes on purpose: `examples/export_ui_data.py` writes the
    JSON and it is spliced in verbatim, so any difference means the splice was
    not done after the export was regenerated.
    """
    assert embedded() == json.loads(EXPORT.read_text(encoding="utf-8")), (
        "the studio page carries a stale copy of examples/ui_data.json. "
        "Re-run the export and splice the result back into the page."
    )


def test_the_palette_is_the_whole_library() -> None:
    """It is generated from the registry, so it cannot quietly lose a component."""
    assert set(embedded()["manifests"]) == set(registered_names())


def js_table(name: str, pattern: str) -> set[str]:
    """The strings in one of the page's small lookup tables."""
    text = STUDIO.read_text(encoding="utf-8")
    start = text.index(f"const {name} = ")
    end = text.index("];" if "[" in text[start : start + 40] else "};", start)
    return set(re.findall(pattern, text[start:end]))


def test_every_category_has_a_place_in_the_palette_and_a_colour() -> None:
    """A category the page has never heard of does not error — it misfiles itself.

    ``indexOf`` returns -1 for an unknown category, which sorts it above every
    known one, and the colour lookup returns undefined and falls back to the
    binary swatch. So a new family of components appears at the top of the
    palette in the wrong colour and nothing says why. It did, the first time a
    photonic block was registered.
    """
    categories = {manifest["category"] for manifest in embedded()["manifests"].values()}
    ordered = js_table("CATEGORY_ORDER", r'"([^"]+)"')
    coloured = js_table("CATEGORY_PORT", r'"([^"]+)":')

    assert categories <= ordered, f"no position in the palette: {sorted(categories - ordered)}"
    assert categories <= coloured, f"no swatch colour: {sorted(categories - coloured)}"


def test_the_schematic_on_the_canvas_is_a_graph_that_runs() -> None:
    """The strongest claim the page makes, and the cheapest one to break.

    The canvas draws `DATA.project` and Run posts it back unchanged. So if this
    document does not run, the page shows a link that cannot exist — which is
    the one thing PRODUCT.md says must never happen.
    """
    payload = run_project(embedded()["project"])
    assert payload["results"], "the schematic produced no results at all"


def test_the_numbers_printed_beside_the_plots_come_from_that_graph() -> None:
    """The dock's reference figures must be this graph's, not an older one's.

    The page shows EVM, SNR and BER before anything has been run, from the
    bundled reference. Running the bundled project has to reproduce them, or
    the page is quoting a number no graph on screen produces.
    """
    data = embedded()
    results = run_project(data["project"])["results"]
    measured = results["vsa"]["out"]
    stated = data["measurement"]

    assert measured["evm"] == pytest.approx(stated["evm"], rel=1e-9, abs=0.0)
    assert measured["snr_db"] == pytest.approx(stated["snr_db"], rel=1e-9, abs=0.0)
    assert measured["symbols_evaluated"] == stated["symbols"]
    assert results["ber"]["out"]["symbol_errors"] == stated["symbol_errors"]


def layout_ids() -> set[str]:
    """Block ids the page has a position for, read out of its LAYOUT table."""
    text = STUDIO.read_text(encoding="utf-8")
    start = text.index("const LAYOUT = {")
    end = text.index("};", start)
    return set(re.findall(r"(\w+):\s*\{\s*x:", text[start:end]))


def test_every_block_in_the_project_has_somewhere_to_be_drawn() -> None:
    """A block with no entry falls back to the top-left corner, silently.

    Layout is the one thing the page still owns — the engine has no opinion
    about where a block sits — so it is also the one thing that can fall out of
    step with the project without anything else noticing.
    """
    project_ids = {node["id"] for node in embedded()["project"]["nodes"]}
    assert layout_ids() == project_ids, (
        f"positions without a block: {sorted(layout_ids() - project_ids)}; "
        f"blocks with no position: {sorted(project_ids - layout_ids())}"
    )


def test_the_page_still_works_without_a_server() -> None:
    """Opened from disk it must fall back, not fail.

    The README tells people they can open the file directly, so the reference
    data has to be sufficient on its own: everything the page draws before a run
    comes from the embedded payload, and the API is only consulted when the page
    was served over http.
    """
    data = embedded()
    for key in ("manifests", "project", "constellation", "eye", "sensitivity", "measurement"):
        assert key in data, f"the page cannot render offline without {key!r}"

    text = STUDIO.read_text(encoding="utf-8")
    assert 'location.protocol === "http:"' in text, (
        "the page must decide whether it has a server from the protocol it was "
        "loaded over, not assume one"
    )


def test_the_canvas_marks_up_what_can_be_edited() -> None:
    """The editor hangs one handler on the SVG and reads the target's markup.

    That is what lets it keep working across the redraw that every edit causes —
    per-element listeners would have to be rebuilt each time. It also means a
    redraw that stops emitting these attributes breaks dragging, wiring and
    selection *silently*: nothing errors, the canvas simply stops responding.
    """
    text = STUDIO.read_text(encoding="utf-8")
    # Both halves of each pair: what the canvas writes, and what the handler
    # reads. Checking only that an attribute appears somewhere is satisfied by
    # any one of its uses, which is no guard at all — dropping it from just the
    # inputs would leave every "data-dir" in the file and break every drop.
    for emitted, read in (
        ('"data-dir": "in"', 'dataset.dir === "in"'),
        ('"data-dir": "out"', 'dataset.dir === "out"'),
        # Tied to the block group specifically. The same literal also appears on
        # every port, so a bare presence check passes while dragging a block by
        # its body is broken — which is exactly what happened when this was
        # written loosely.
        ('role: "button", "data-node": n.id', "[data-node]"),
        ('"data-port": name', "[data-port]"),
        ('"data-edge": index', "[data-edge]"),
    ):
        assert emitted in text, f"the canvas no longer emits {emitted}"
        assert read in text, f"nothing reads {read} any more"


def test_an_input_can_only_have_one_source() -> None:
    """The rule the editor enforces while wiring, pinned against the engine.

    The editor refuses a second wire into an input and says which port already
    has a source. It is not inventing that rule — the engine holds it too, and
    raises. What the editor adds is the timing: the same refusal arrives while
    the wire is being dragged rather than when Run is pressed, which is the
    difference between a rule you can feel and one you discover.
    """
    from maiman import Graph, GraphError, SimulationContext
    from maiman.components import CWLaser, Fiber

    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=4, sequence_length=16, seed=1)
    graph = Graph(ctx)
    first = graph.add(CWLaser(label="a"))
    second = graph.add(CWLaser(label="b"))
    fiber = graph.add(Fiber(label="span"))
    graph.connect(first, fiber["in"])
    with pytest.raises(GraphError, match="exactly one connection"):
        graph.connect(second, fiber["in"])
    assert len(graph.edges) == 1


def test_the_sweep_form_offers_the_axis_the_server_expects() -> None:
    """The form's fields and the endpoint's request must be the same shape.

    The page builds ``{"node", "parameter", "values"}`` and the server reads
    exactly those three. They are in different languages in different files, so
    nothing but a test connects them, and a rename on either side would be
    answered as a 400 that looks like the user's fault.
    """
    text = STUDIO.read_text(encoding="utf-8")
    assert "axis: { node, parameter, values }" in text
    for field in ("sw-node", "sw-param", "sw-from", "sw-to", "sw-steps", "sw-runs", "sw-run"):
        assert f'id="{field}"' in text, f"the sweep form lost {field}"


def test_a_saved_project_is_one_the_engine_can_open() -> None:
    """What Save writes has to be what load() reads — it is one format, not two.

    The page adds ``ui`` positions to each node, which is what the format
    provides for and what ``ui_from_dict`` reads back. This checks the engine
    accepts a document shaped that way rather than merely tolerating it.
    """
    from maiman.project import graph_to_dict, ui_from_dict

    positions = {"tx": {"x": 10.0, "y": 20.0}, "span": {"x": 30.0, "y": 40.0}}
    saved = graph_to_dict(_simple_graph(), ui=positions)
    assert ui_from_dict(saved) == positions
    payload = run_project(saved)
    assert payload["ui"] == positions
    assert payload["results"]


def _simple_graph():  # type: ignore[no-untyped-def]
    from maiman import Graph, SimulationContext
    from maiman.components import CWLaser, Fiber, PowerMeter

    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=4, sequence_length=32, seed=3)
    graph = Graph(ctx)
    laser = graph.add(CWLaser(power=5.0, label="tx"))
    fiber = graph.add(Fiber(length=100.0, attenuation=0.25, label="span"))
    meter = graph.add(PowerMeter(label="pm"))
    graph.connect(laser, fiber["in"])
    graph.connect(fiber, meter["in"])
    return graph


def test_opening_a_project_does_not_reach_the_server() -> None:
    """A file is chosen in the operating system's picker and read locally.

    A server that opened or wrote a path it was handed would be a different and
    much worse program. Nothing in the page posts a project to be read, and
    there is no route that would accept one — both halves are pinned here,
    because either alone would leave the other free to drift.
    """
    from maiman import server

    text = STUDIO.read_text(encoding="utf-8")
    assert 'type="file"' in text, "opening must go through the file picker"
    assert "readAsText" in text, "the file is read in the browser"

    source = Path(server.__file__).read_text(encoding="utf-8")
    for route in ("/api/open", "/api/load", "/api/save"):
        assert f'route == "{route}"' not in source, f"the server should have no {route} route"


def test_hidden_actually_hides() -> None:
    """The page must declare it, because the browser's own rule is not enough.

    A browser hides ``[hidden]`` through its user-agent stylesheet, and *any*
    author declaration of ``display`` beats that entire sheet regardless of
    specificity. So one rule like ``.menu-pop { display: flex }`` un-hides
    every element of that class that carries the attribute — silently, with
    ``element.hidden`` still reading true.

    That is exactly what happened to the File menu: the logic was right, every
    test that asked the DOM said hidden, and the menu sat open on screen. It
    shipped in the README screenshot that way. A test that reads the flag
    cannot see it; this one checks that the page declares the rule that makes
    the flag mean something.
    """
    text = STUDIO.read_text(encoding="utf-8")
    assert "[hidden] { display: none !important; }" in text, (
        "the page needs a global [hidden] rule that outranks its own display "
        "declarations, or any hidden element with a display rule stays visible"
    )


def test_the_format_is_edited_link_wide() -> None:
    """Bits per symbol is a property of the signal, not of a block.

    Three blocks in the shipped graph each carry it — the mapper, the reference
    mapper, and the generator whose only use for it is knowing how many bits to
    emit — and no unequal setting of them runs. The page edits them together;
    this pins the engine behaviour that makes that the only sensible choice.
    """
    text = STUDIO.read_text(encoding="utf-8")
    assert 'LINK_WIDE = new Set(["bits_per_symbol"])' in text

    carriers = {
        name
        for name, manifest in __import__("maiman").manifests().items()
        if "bits_per_symbol" in manifest["parameters"]
    }
    assert carriers == {"PRBSGenerator", "QAMMapper"}, (
        f"a new component carries bits_per_symbol: {sorted(carriers)}. The editor "
        "spreads the value to every block that has it, so this is a note that the "
        "set grew, not necessarily a fault."
    )

    document_ = embedded()["project"]
    for node in document_["nodes"]:
        if node["type"] in carriers:
            node["params"]["bits_per_symbol"] = 2.0
    assert run_project(document_)["results"], "a link-wide format change must run"


def test_the_constellation_in_the_dock_is_produced_by_a_block_on_the_canvas() -> None:
    """What the interface draws must be something the schematic contains."""
    project = embedded()["project"]
    assert "ConstellationDiagram" in {node["type"] for node in project["nodes"]}
    results = run_project(project)["results"]
    kinds = {value["kind"] for ports in results.values() for value in ports.values()}
    assert "constellation" in kinds


def test_the_coherent_link_has_no_eye_and_does_not_pretend_to() -> None:
    """A coherent receiver has no eye, so the shipped graph has no Eye Diagram.

    The rails only mean something relative to a recovered carrier phase, and the
    constellation arrives rotated by whatever two free-running lasers happen to
    differ by — measured at 35 to 45 degrees, and drifting. Folding either rail
    gives a smear. An EyeDiagram wired there would draw one, and it would be
    meaningless, which is worse than an empty panel.
    """
    project = embedded()["project"]
    assert "EyeDiagram" not in {node["type"] for node in project["nodes"]}

    text = STUDIO.read_text(encoding="utf-8")
    assert 'const canProduce = PROJECT.nodes.some((n) => n.type === "EyeDiagram");' in text, (
        "the eye pane must show the reference only while the graph on canvas could produce one"
    )


def test_the_dock_draws_the_run_s_plots_and_not_the_bundled_ones() -> None:
    """Both halves: the page takes a plot off the run, and the plot reads it.

    Same shape of guard as the canvas markup. A run's results reaching
    ``SESSION.plots`` and the drawing code reading them are in different parts
    of one file, and severing either leaves the dock showing the reference run
    for ever, silently, under a badge that says live. That is what the eye did
    before there was a block behind it.
    """
    text = STUDIO.read_text(encoding="utf-8")
    for stored, read in (
        ('SESSION.plots.eye = firstOfKind("eye")', "const live = SESSION.plots.eye;"),
        (
            'SESSION.plots.constellation = firstOfKind("constellation")',
            "SESSION.plots.constellation || DATA.constellation",
        ),
    ):
        assert stored in text, f"a run's plots no longer reach: {stored}"
        assert read in text, f"nothing draws from it: {read}"

    # And the pane says so rather than drawing the reference, when a run
    # produced no plot of that kind at all.
    assert "SESSION.hasRun && !SESSION.plots.eye" in text
    assert "No eye in this run" in text


def eye_openness(counts: list[list[int]]) -> float:
    """Fraction of the histogram's central column that no trace crosses.

    What "is the eye open" means, measured without assuming how many levels the
    format has. A first attempt split the samples into equal groups by value and
    compared their spreads; it scored a *noiseless* four-level signal worse than
    a noisy one, and everything it had measured was thrown away.
    """
    column = [row[len(row) // 2] for row in counts]
    return sum(1 for value in column if value == 0) / len(column)


def test_the_shipped_eye_is_an_open_one() -> None:
    """The panel exists to show what an eye looks like, so it must show one.

    Around 83 per cent of the central column crossed by nothing. The coherent
    rails that used to feed this panel scored 24 per cent, which is what a
    closed eye looks like.
    """
    assert eye_openness(embedded()["eye"]["counts"]) > 0.7


def test_the_shipped_eye_comes_from_the_project_beside_it() -> None:
    """And that project opens, runs, and produces exactly that eye.

    The reference is no longer the graph on the canvas — it cannot be, since
    that link has no eye — so it ships with the project it did come from, which
    anyone can open from the File menu and take further.
    """
    project = json.loads((ROOT / "examples" / "ook_eye.maiman").read_text(encoding="utf-8"))
    assert "EyeDiagram" in {node["type"] for node in project["nodes"]}
    assert all("ui" in node for node in project["nodes"]), "it should open laid out"

    results = run_project(project)["results"]
    live = next(v for ports in results.values() for v in ports.values() if v["kind"] == "eye")
    assert live["counts"] == embedded()["eye"]["counts"]
    assert eye_openness(live["counts"]) > 0.7


def test_a_two_point_alphabet_passes_through_the_differential_decoder() -> None:
    """BPSK has no quadrants, so there is nothing to undo, so it is the identity.

    Not the block declining to work: QAMMapper refuses to differentially encode
    BPSK at all, so nothing was encoded, and undoing nothing is a pass-through.
    A link built for a higher format keeps running when someone drops it to
    BPSK to see what happens.
    """
    import numpy as np

    from maiman import SimulationContext
    from maiman.components import DifferentialDecoder
    from maiman.modulation import qam_constellation
    from maiman.signals import SymbolSignal

    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=4, sequence_length=8, seed=1)
    points = qam_constellation(1)
    symbols = points[np.array([0, 1, 1, 0, 1, 0, 0, 1])]
    signal = SymbolSignal(symbols=symbols, symbol_rate=10e9, constellation=points)

    out = DifferentialDecoder(label="dec").run(ctx, {"in": signal})["out"]
    assert np.array_equal(np.asarray(out.symbols), symbols)
    assert np.array_equal(np.asarray(out.constellation), points)


def test_the_dock_reads_the_analyser_that_is_measuring_the_signal() -> None:
    """A link may hold two constellation analysers and they are not alike.

    One on the recovered symbols measures modulation quality. One after a
    decision device sees points sitting exactly on the constellation and reports
    an EVM of zero however bad the link is — DifferentialDecoder's own docstring
    says not to measure EVM after it. Taking whichever came first put that zero
    in the dock under a badge reading live.
    """
    text = STUDIO.read_text(encoding="utf-8")
    assert 'allOfKind("constellation_measurement")' in text
    assert "m.evm > best.evm" in text, "the dock must pick the analyser with a real EVM"

    results = run_project(embedded()["project"])["results"]
    evms = sorted(
        value["evm"]
        for ports in results.values()
        for value in ports.values()
        if value["kind"] == "constellation_measurement"
    )
    assert len(evms) >= 2, "the shipped link should still have both analysers"
    assert evms[0] < 1e-6 < evms[-1], f"one analyser should read ~0 and one a real EVM, got {evms}"


def test_the_dock_is_cleared_before_a_run_writes_to_it() -> None:
    """A field the new run says nothing about must not keep the last one's answer.

    Opening a direct-detection link over a coherent one left EVM, SNR and BER
    reading the coherent link's values, and the error count reading
    "undefined / undefined" — that one at least was visible.
    """
    text = STUDIO.read_text(encoding="utf-8")
    assert 'for (const id of ["m-evm", "m-snr", "m-ber", "m-err"' in text
    # And the headings move with the measurement, so a Q never sits under "SNR".
    assert 'label("k-evm", "Q")' in text


def test_the_page_only_reads_dispersion_fields_the_engine_sends() -> None:
    """Every ``value.x`` in the compensator's log line has to be a key of the encoding.

    The compensator learned to search for its own value, which added three fields
    to what it reports and a branch in the page that reads two of them. A field
    renamed on one side of that and not the other prints "undefined" in a log
    whose whole claim is that every line is a fact from the response — and it
    would print it only on runs with the search switched on, which is exactly the
    run nobody re-tests by hand.
    """
    from maiman.components.dsp import DispersionDiagnostics
    from maiman.encoding import encode

    text = STUDIO.read_text(encoding="utf-8")
    start = text.index('case "dispersion":')
    # To the next case at the same indentation, whichever that turns out to be.
    # An earlier version ended the slice at `case "opaque"` because that was the
    # one which happened to follow; four cases were later inserted between them
    # and the slice swallowed all four.
    end = re.search('\n      case "', text[start + 1 :])
    assert end, "the dispersion case is the last one; the slice has nothing to stop at"
    body = text[start : start + 1 + end.start()]
    read = set(re.findall(r"value\.([A-Za-z_][A-Za-z0-9_]*)", body))

    sent = set(
        encode(
            DispersionDiagnostics(
                accumulated_dispersion=1.36,
                removed_symbols=13.4,
                estimated=True,
                declared=1.30,
                contrast=31.8,
            )
        )
    )
    assert read, "the dispersion branch reads nothing; the parse is wrong, not the page"
    assert read <= sent, f"the page reads {sorted(read - sent)}, which the engine never sends"
    # And the branch that only runs with the search on is present at all.
    assert "estimated" in read


def test_the_coherent_link_has_no_spectrum_and_does_not_pretend_to() -> None:
    """One channel and nothing to look at it with, so the pane says so.

    Same guard as the eye, for a plainer reason: the shipped graph has no
    Optical Spectrum Analyzer on it, so there is nothing for the pane to draw
    and the bundled trace belongs to a different link. Drawing it anyway under
    a badge reading live is the mistake the dock already made once.
    """
    project = embedded()["project"]
    assert "OpticalSpectrumAnalyzer" not in {node["type"] for node in project["nodes"]}

    text = STUDIO.read_text(encoding="utf-8")
    assert (
        'const canProduce = PROJECT.nodes.some((n) => n.type === "OpticalSpectrumAnalyzer");'
        in text
    ), "the spectrum pane must show the reference only while the graph could produce one"
    assert "SESSION.hasRun && !SESSION.plots.spectrum" in text
    assert "No spectrum in this run" in text


def test_the_dock_draws_the_run_s_spectrum_and_not_the_bundled_one() -> None:
    """The run's trace has to reach the pane, and the pane has to read it."""
    text = STUDIO.read_text(encoding="utf-8")
    assert 'SESSION.plots.spectrum = firstOfKind("spectrum")' in text, (
        "a run's spectrum no longer reaches the dock"
    )
    assert "const live = SESSION.plots.spectrum;" in text, "nothing draws from it"


def test_the_readings_are_cleared_with_the_trace_they_describe() -> None:
    """A pane that has just said there is no spectrum must not list one's numbers.

    The plot and the panel beside it are written by different code, and leaving
    the bundled readings up next to an empty plot tells exactly the lie the
    empty plot was careful not to.
    """
    text = STUDIO.read_text(encoding="utf-8")
    assert text.count("clearSpectrumPanel(); drawNoSpectrum(ctx, w, h); return;") == 2, (
        "both refusals must clear the readings before drawing the empty state"
    )


def test_the_shipped_spectrum_comes_from_the_project_beside_it() -> None:
    """And that project opens, runs, and produces exactly that trace.

    Bundled reduced — wavelength and displayed level — because those are the two
    arrays the plot reads. A live run sends four, so the reduction is redone here
    from the run's own numbers and has to land on the shipped ones.
    """
    project = json.loads((ROOT / "examples" / "wdm_osa.maiman").read_text(encoding="utf-8"))
    assert "OpticalSpectrumAnalyzer" in {node["type"] for node in project["nodes"]}
    assert all("ui" in node for node in project["nodes"]), "it should open laid out"

    results = run_project(project)["results"]
    live = next(v for ports in results.values() for v in ports.values() if v["kind"] == "spectrum")
    shipped = embedded()["spectrum"]

    assert live["wavelengths_nm"] == pytest.approx(shipped["wavelengths_nm"], abs=1e-4)
    assert live["resolution_bandwidth"] / 1e9 == pytest.approx(shipped["resolution_bandwidth_ghz"])
    displayed = [
        max(10.0 * math.log10(max(w, 1e-18) * 1e3), -90.0) for w in live["power_per_resolution_w"]
    ]
    assert displayed == pytest.approx(shipped["dbm"], abs=1e-3)


def test_the_shipped_spectrum_shows_a_comb_standing_over_a_floor() -> None:
    """The pane exists to show what a spectrum looks like, so it must show one.

    Four channels on the 100 GHz grid and an ASE floor beneath them, which is
    the whole reason the reference link is amplified. A trace with no floor
    teaches nothing about resolution bandwidth, and one with no channels is not
    a spectrum anybody would recognise.
    """
    shipped = embedded()["spectrum"]
    dbm = shipped["dbm"]
    peak = max(dbm)
    floor = sorted(dbm)[len(dbm) // 2]
    assert peak - floor > 15.0, "the comb has to stand clear of the noise it sits on"

    # Four channels, counted as maxima within 3 dB of the peak rather than
    # assumed from the project: if the grid or the span changes, this changes
    # with it instead of going quietly wrong.
    tops = [
        i
        for i in range(1, len(dbm) - 1)
        if dbm[i] >= dbm[i - 1] and dbm[i] > dbm[i + 1] and dbm[i] > peak - 3.0
    ]
    assert len(tops) == 4, f"expected four channels, found {len(tops)}"


def test_the_spectrum_is_read_in_the_bandwidth_it_was_measured_in() -> None:
    """The axis is labelled per resolution bandwidth, because the trace is.

    An OSA reports power within its own resolution, which is why a carrier and a
    noise floor answer the setting differently. A trace labelled in bare dBm
    would be a number nobody could act on.
    """
    text = STUDIO.read_text(encoding="utf-8")
    assert "`dBm / ${fmt(source.resolution_bandwidth_ghz)} GHz`" in text
    assert "power_per_resolution_w" in text, "the plot must draw what an instrument displays"


def test_the_page_carries_the_current_mark() -> None:
    """The tab icon is the artwork in assets/, byte for byte.

    Inlined because the page has to open straight off disk with nothing running
    — the server hands out `/` and nothing else — and the cost of that is the
    same copy the bundled JSON already pays for. Re-export the mark and the page
    keeps the old one, silently, which is exactly how the palette in this file
    once went two components short.
    """
    text = STUDIO.read_text(encoding="utf-8")
    for name, size in (("icon-32.png", "32x32"), ("icon-192.png", "192x192")):
        expected = base64.b64encode((ROOT / "assets" / name).read_bytes()).decode("ascii")
        link = (
            f'<link rel="icon" type="image/png" sizes="{size}" '
            f'href="data:image/png;base64,{expected}">'
        )
        assert link in text, (
            f"the studio page carries a stale copy of assets/{name}. "
            f'Re-inline it as a data URI on the <link rel="icon"> for {size}.'
        )


def test_the_menubar_wears_the_mark() -> None:
    """And it is the tight crop, not the padded app icon.

    Two different pieces of artwork for two different jobs. The icon is squared
    and padded because an operating system draws it in a tile of its own; the
    menubar sets it beside type, where that padding reads as the mark having
    drifted away from the words. Picking either one for both is the mistake this
    guards.
    """
    text = STUDIO.read_text(encoding="utf-8")
    expected = base64.b64encode((ROOT / "assets" / "logo-mark-tight.png").read_bytes()).decode(
        "ascii"
    )
    assert (
        f'class="brand-mark" alt="" aria-hidden="true" src="data:image/png;base64,{expected}"'
        in text
    ), (
        "the menubar carries a stale copy of assets/logo-mark-tight.png. "
        "Re-inline it as the .brand-mark data URI."
    )
    # Decorative: the name is right beside it in text, so announcing the mark
    # too would have a screen reader say the product twice.
    assert 'class="brand-mark" alt="" aria-hidden="true"' in text


def test_every_kind_the_engine_can_send_has_a_line_to_print() -> None:
    """No encoded result may reach the log as the bare word for its own type.

    ``describe()`` ends in a ``default`` that returns ``value.kind``, so a kind
    with no case of its own logs as "electrical" or "optical" — true, useless,
    and silent about it. Four did: the raw signals, whose payloads carried rms,
    sample rate, band powers and noise bins the whole time.

    Enumerated from the encoder rather than listed here, so adding a kind to
    ``encoding.py`` and forgetting the page fails on the next run instead of
    shipping a log line nobody can act on.
    """
    encoder = (ROOT / "src" / "maiman" / "encoding.py").read_text(encoding="utf-8")
    kinds = set(re.findall(r'"kind": "([a-z_]+)"', encoder))
    assert kinds, "no kinds found — has the encoder's shape changed?"

    text = STUDIO.read_text(encoding="utf-8")
    handled = set(re.findall(r'case "([a-z_]+)":', text))
    missing = sorted(kinds - handled)
    assert not missing, (
        f"the log has no case for {missing}, so each falls through to `default` "
        f"and prints its own kind back. Give it the fact the payload carries."
    )


def test_the_dock_shows_a_run_it_cannot_see_the_end_of() -> None:
    """The bar is drawn from the engine's fraction, never from a timer.

    A bar that fills on a clock is a lie with a progress indicator's authority:
    it says 90 % at a moment nothing knows anything about, and it has to be
    walked back or left stuck. What makes this one honest is that every width it
    takes is a fraction the engine has already reported.
    """
    text = STUDIO.read_text(encoding="utf-8")
    assert "/api/run/stream" in text, "the page no longer asks for progress"
    assert "function showProgress" in text
    assert "report.fraction" in text, "the bar must be drawn from the engine's own fraction"
    # And it can still run against a server that has no such route.
    assert "streamError.status !== 404" in text, "no fallback to the unstreamed route"


def test_stop_stops_something() -> None:
    """The button was enabled for the whole of every run and wired to nothing.

    Which is the same class of claim as a plot drawn from the wrong run: the
    interface offering an action it does not perform. It now aborts the
    response, and the log says exactly that much — the engine is left to finish,
    because block-mode execution has no safe place to stop halfway.
    """
    text = STUDIO.read_text(encoding="utf-8")
    assert 'stopBtn.addEventListener("click"' in text, "Stop is still wired to nothing"
    assert "RUN.controller.abort()" in text
    assert 'error.name === "AbortError"' in text, "an abort would be reported as a failure"
