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

import json
import re
from pathlib import Path
from typing import Any

import pytest

from maiman import registered_names
from maiman.server import run_project

ROOT = Path(__file__).resolve().parent.parent
STUDIO = ROOT / "docs" / "ui-mockup.html"
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
        "docs/ui-mockup.html carries a stale copy of examples/ui_data.json. "
        "Re-run the export and splice the result back into the page."
    )


def test_the_palette_is_the_whole_library() -> None:
    """It is generated from the registry, so it cannot quietly lose a component."""
    assert set(embedded()["manifests"]) == set(registered_names())


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
