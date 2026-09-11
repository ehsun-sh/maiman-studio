"""Reading a layout netlist and solving the circuit it describes.

The load-bearing test is the first one: a file gdsfactory actually wrote, parsed
by the shipped reader, solved against the shipped kit, and checked against a loss
anyone can do by hand. Everything else here is about what happens when the
netlist and the kit disagree, which is the normal case while a kit is being
written and the only case where being told beats being given a number.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from maiman.components.photonic import ScatteringDevice
from maiman.netlist import (
    NetlistError,
    circuit_from_netlist,
    load_netlist,
    missing_cells,
    parse_netlist,
)
from maiman.pdk import PDK, PDKError, load_pdk, pdk_from_dict
from maiman.units import C_LIGHT

ROOT = Path(__file__).resolve().parent.parent
KIT_PATH = ROOT / "examples" / "silicon_220nm.pdk.json"
FROM_GDSFACTORY = ROOT / "tests" / "data" / "gdsfactory_straight_with_bend.yml"
RACETRACK = ROOT / "examples" / "ring_racetrack.netlist.yml"

REFERENCE = C_LIGHT / 1550e-9

yaml = pytest.importorskip("yaml", reason="the shipped netlists are YAML, as a layout tool writes")


@pytest.fixture
def kit() -> PDK:
    return load_pdk(KIT_PATH)


def grid(points: int = 2001, span: float = 1.2e12) -> np.ndarray:
    return np.linspace(REFERENCE - span, REFERENCE + span, points)


# ---------------------------------------------------------------------------
# a file the layout tool actually wrote


def test_a_netlist_gdsfactory_wrote_solves_to_the_loss_it_should(kit: PDK) -> None:
    """The whole claim, on a real file and against arithmetic done by hand.

    Two instances, 26.637 um of waveguide between them at 2 dB/cm, so 0.0053 dB.
    The bend's 16.637 um is its *arc* length, which the netlist carries in
    ``info`` and which the radius alone does not give — a kit that used a nominal
    length here would be describing a different circuit from the one drawn.
    """
    netlist = load_netlist(FROM_GDSFACTORY)
    assert netlist.cells() == ["bend_euler", "straight"]
    assert missing_cells(netlist, kit) == []
    assert len(netlist.instances) == 2
    assert len(netlist.nets) == 1
    assert set(netlist.ports) == {"o1", "o2"}

    solved = circuit_from_netlist(netlist, kit, grid()).solve()
    assert set(solved.ports) == {"o1", "o2"}

    total_um = sum(float(i.info["length"]) for i in netlist.instances.values())
    assert total_um == pytest.approx(26.637)
    measured = 10.0 * np.log10(solved.power("o2", "o1").mean())
    assert measured == pytest.approx(-2.0 * total_um * 1e-4, abs=1e-5)


def test_the_length_really_comes_from_the_netlist_and_not_from_the_kit(kit: PDK) -> None:
    """``from_netlist`` is the mechanism; this is what it would mean to lose it.

    The kit's nominal length for a bend is 16.637 um and for a straight 1000. If
    the override silently stopped being applied, the straight in this netlist —
    10 um — would be built at the nominal thousand and the loss would be nearly a
    hundred times larger. Asserting the loss alone would catch that; asserting
    the built length says *why*.
    """
    netlist = load_netlist(FROM_GDSFACTORY)
    straight = next(i for i in netlist.instances.values() if i.cell == "straight")
    assert float(straight.info["length"]) == 10.0

    entry = kit.device_for_cell("straight")
    assert entry.from_netlist == {"length": "info.length"}
    assert kit.parameters(entry.name)["length"] == 1000.0, "the kit's own nominal"

    built = kit.make(entry.name, length=float(straight.lookup("info.length")))
    assert cast(Any, built).length == 10.0


def test_reading_the_same_document_as_json_gives_the_same_netlist(kit: PDK) -> None:
    """YAML is what the tool writes; JSON is what this package can read unaided.

    A netlist is a document either way, and the reader that matters is the one
    over the parsed mapping. This pins that the two routes agree, so the JSON
    path is not a second implementation that drifts.
    """
    from_yaml = load_netlist(FROM_GDSFACTORY)
    as_json = parse_netlist(
        json.loads(json.dumps(yaml.safe_load(FROM_GDSFACTORY.read_text(encoding="utf-8"))))
    )
    assert as_json.instances.keys() == from_yaml.instances.keys()
    assert as_json.nets == from_yaml.nets
    assert as_json.ports == from_yaml.ports


# ---------------------------------------------------------------------------
# a circuit with feedback in it


def test_the_racetrack_resonates_where_its_loop_length_says_it_should(kit: PDK) -> None:
    """The case a chain of transfer functions cannot do.

    The loop leaves the coupler and returns to it, so there is no order to run
    the five blocks in. What is checked is not that a spectrum appeared but that
    its free spectral range is ``c / (n_g L)`` for the loop the netlist actually
    describes — 100.000 um, summed from the five instances' own lengths.
    """
    netlist = load_netlist(RACETRACK)
    assert missing_cells(netlist, kit) == []

    loop = sum(float(i.info["length"]) for name, i in netlist.instances.items() if name != "cp")
    assert loop == pytest.approx(100.0)

    frequencies = np.linspace(REFERENCE - 8e11, REFERENCE + 8e11, 200001)
    power = circuit_from_netlist(netlist, kit, frequencies).solve().power("through", "in")

    threshold = power.min() + 0.5 * (power.max() - power.min())
    interior = np.arange(1, power.size - 1)
    dips = interior[
        (power[1:-1] < power[:-2]) & (power[1:-1] <= power[2:]) & (power[1:-1] < threshold)
    ]
    assert dips.size >= 2, "1.6 THz should hold more than one resonance"

    spacing = float(np.mean(np.diff(frequencies[dips])))
    assert spacing == pytest.approx(C_LIGHT / (4.20 * loop * 1e-6), rel=1e-3)
    assert power.min() < 0.2, "near critical coupling this is a real notch"


def test_a_window_narrower_than_one_free_spectral_range_finds_nothing(kit: PDK) -> None:
    """Measured while writing the example, and kept because it looks like a bug.

    A 500 GHz scan of a ring whose resonances are 714 GHz apart can miss every
    one of them, and what comes back is a flat line at 0.995 — which reads
    exactly like a ring that is not resonating rather than a window that is too
    narrow. The fix is the window, and nothing in the output says so.
    """
    netlist = load_netlist(RACETRACK)
    narrow = np.linspace(C_LIGHT / 1552e-9, C_LIGHT / 1548e-9, 4001)
    power = circuit_from_netlist(netlist, kit, narrow).solve().power("through", "in")
    assert power.min() > 0.9, "no resonance in this window, and it looks like none exist"

    wide = np.linspace(REFERENCE - 8e11, REFERENCE + 8e11, 200001)
    assert circuit_from_netlist(netlist, kit, wide).solve().power("through", "in").min() < 0.2


# ---------------------------------------------------------------------------
# what it refuses


def test_a_cell_the_kit_does_not_model_is_refused_by_name(kit: PDK) -> None:
    """Skipping it would return a circuit that solves and is not the one drawn."""
    netlist = parse_netlist(
        {
            "name": "x",
            "instances": {"g": {"component": "grating_coupler_elliptical"}},
            "ports": {"o1": "g,o1"},
        }
    )
    assert missing_cells(netlist, kit) == ["grating_coupler_elliptical"]
    with pytest.raises(PDKError, match="grating_coupler_elliptical"):
        circuit_from_netlist(netlist, kit, grid(9))


def test_a_port_the_kit_does_not_map_is_refused(kit: PDK) -> None:
    """Guessing would wire a drop to a through, and the circuit would still solve."""
    netlist = parse_netlist(
        {
            "name": "x",
            "instances": {"s": {"component": "straight", "info": {"length": 10.0}}},
            "ports": {"a": "s,o7"},
        }
    )
    with pytest.raises(NetlistError, match="o7"):
        circuit_from_netlist(netlist, kit, grid(9))


def test_a_netlist_that_is_not_one_is_refused() -> None:
    with pytest.raises(NetlistError, match="no instances"):
        parse_netlist({"name": "empty"})
    with pytest.raises(NetlistError, match="does not say which cell"):
        parse_netlist({"instances": {"a": {}}})
    with pytest.raises(NetlistError, match="no instance"):
        parse_netlist(
            {"instances": {"a": {"component": "straight"}}, "nets": [{"p1": "b,o1", "p2": "a,o1"}]}
        )
    with pytest.raises(NetlistError, match="port reference"):
        parse_netlist(
            {"instances": {"a": {"component": "straight"}}, "nets": [{"p1": "a", "p2": "a,o1"}]}
        )
    with pytest.raises(NetlistError, match="wired twice"):
        parse_netlist(
            {
                "instances": {
                    "a": {"component": "straight"},
                    "b": {"component": "straight"},
                    "c": {"component": "straight"},
                },
                "nets": [{"p1": "a,o1", "p2": "b,o1"}, {"p1": "a,o1", "p2": "c,o1"}],
            }
        )
    with pytest.raises(NetlistError, match="wired by"):
        parse_netlist(
            {
                "instances": {"a": {"component": "straight"}, "b": {"component": "straight"}},
                "nets": [{"p1": "a,o1", "p2": "b,o1"}],
                "ports": {"x": "a,o1"},
            }
        )


def test_a_circuit_with_nothing_facing_outward_is_refused(kit: PDK) -> None:
    """There is no question to ask it. Solving would raise further in, less clearly."""
    netlist = parse_netlist(
        {
            "name": "sealed",
            "instances": {
                "a": {"component": "straight", "info": {"length": 10.0}},
                "b": {"component": "straight", "info": {"length": 10.0}},
            },
            "nets": [{"p1": "a,o2", "p2": "b,o1"}],
        }
    )
    with pytest.raises(NetlistError, match="exposes no ports"):
        circuit_from_netlist(netlist, kit, grid(9))


def test_a_netlist_path_that_is_not_there_says_which_instance(kit: PDK) -> None:
    """A kit sourcing `info.length` from a cell that has no info is a kit bug."""
    netlist = parse_netlist(
        {
            "name": "x",
            "instances": {"s": {"component": "straight"}},
            "ports": {"a": "s,o1"},
        }
    )
    with pytest.raises(NetlistError, match=r"has no 'info\.length'"):
        circuit_from_netlist(netlist, kit, grid(9))


def test_two_devices_may_not_model_the_same_cell() -> None:
    """A netlist naming it could mean either, and picking one is a coin toss."""
    document = {
        "schema_version": 1,
        "name": "ambiguous",
        "reference_wavelength": 1550.0,
        "devices": {
            "one": {"component": "Waveguide", "cell": "straight", "params": {"length": 1.0}},
            "two": {"component": "Waveguide", "cell": "straight", "params": {"length": 2.0}},
        },
    }
    with pytest.raises(PDKError, match="both"):
        pdk_from_dict(document)


def test_a_kit_may_not_source_a_parameter_the_component_has_not_got() -> None:
    document = {
        "schema_version": 1,
        "name": "wrong",
        "reference_wavelength": 1550.0,
        "devices": {
            "w": {
                "component": "Waveguide",
                "cell": "straight",
                "from_netlist": {"radius": "info.radius"},
            }
        },
    }
    with pytest.raises(PDKError, match="radius"):
        pdk_from_dict(document)


# ---------------------------------------------------------------------------
# the shipped kit stays netlist-ready


def test_every_cell_the_shipped_netlists_use_is_modelled_by_the_shipped_kit(kit: PDK) -> None:
    """A guard, because these three files are edited separately and drift quietly."""
    for path in (FROM_GDSFACTORY, RACETRACK):
        netlist = load_netlist(path)
        assert missing_cells(netlist, kit) == [], f"{path.name} uses a cell the kit lost"
        for instance in netlist.instances.values():
            entry = kit.device_for_cell(instance.cell)
            assert entry.ports, f"{entry.name} maps no layout ports"


def test_the_kit_maps_layout_ports_onto_ports_its_models_have(kit: PDK) -> None:
    """Checked here rather than at load, because only the built device knows.

    A circuit wires the *scattering matrix's* ports, which are not always the
    graph ports a link wires — a 2x2 MMI driven on one side has three graph ports
    and a four-port matrix. So the names cannot be validated against the manifest
    when the file is read, and this is what covers them instead.
    """
    frequencies = grid(5)
    for entry in kit.devices.values():
        if not entry.cell:
            continue
        device = kit.make(entry.name)
        assert isinstance(device, ScatteringDevice), f"{entry.name} is not a circuit device"
        available = set(device.scattering_matrix(frequencies).ports)
        unknown = set(entry.ports.values()) - available
        assert not unknown, f"{entry.name} maps onto {sorted(unknown)}, which it has not got"
