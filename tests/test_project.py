"""Validation of the .maiman project file format and the component registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from maiman import (
    Graph,
    ProjectError,
    SimulationContext,
    UnknownComponentError,
    load,
    manifests,
    registered_names,
    save,
)
from maiman.component import Param, PortType
from maiman.components import (
    BERAnalyzer,
    Combiner,
    CWLaser,
    ElectricalFilter,
    Fiber,
    MachZehnderModulator,
    NRZDriver,
    PINPhotodiode,
    PowerMeter,
    PRBSGenerator,
)
from maiman.components.electrical import PRBS_TAPS
from maiman.modulation import QAM_FORMATS, qam_constellation
from maiman.project import SCHEMA_VERSION, graph_to_dict, ui_from_dict
from maiman.registry import lookup


def _link_graph() -> Graph:
    """A full OOK link, so the round trip covers every kind of node and edge."""
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=512, seed=7)
    g = Graph(ctx)
    prbs = g.add(PRBSGenerator(order=15.0, label="prbs"))
    driver = g.add(NRZDriver(v_low=4.0, v_high=0.0, label="driver"))
    laser = g.add(CWLaser(power=-18.0, wavelength=1550.0, label="laser"))
    mzm = g.add(MachZehnderModulator(v_pi=4.0, extinction_ratio=30.0, label="mzm"))
    fiber = g.add(Fiber(length=40.0, attenuation=0.2, dispersion=17.0, label="fiber"))
    pin = g.add(PINPhotodiode(responsivity=0.8, shot_noise=True, label="pin"))
    lpf = g.add(ElectricalFilter(bandwidth=7.0, label="lpf"))
    analyzer = g.add(BERAnalyzer(label="ber"))

    g.chain(prbs, driver)
    g.connect(laser, mzm["optical_in"])
    g.connect(driver, mzm["electrical_in"])
    g.chain(mzm, fiber, pin, lpf)
    g.connect(lpf, analyzer["in"])
    g.connect(prbs["out"], analyzer["reference"])
    return g


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def test_built_in_components_are_registered_on_import() -> None:
    names = registered_names()
    assert "CWLaser" in names
    assert "MachZehnderModulator" in names
    assert lookup("Fiber") is Fiber


def test_unknown_component_names_the_problem_and_the_fix() -> None:
    with pytest.raises(UnknownComponentError, match="import that package"):
        lookup("SomeoneElsesLaser")


def test_a_project_file_cannot_name_an_arbitrary_import_path() -> None:
    """Resolving a name by importing a dotted path out of a file would make
    opening someone else's project equivalent to running their code. Only
    already-registered names resolve.
    """
    with pytest.raises(UnknownComponentError):
        lookup("os.system")
    with pytest.raises(UnknownComponentError):
        lookup("maiman.components.sources.CWLaser")


def test_manifests_cover_every_registered_component() -> None:
    """The GUI palette is generated from the classes, so it cannot drift."""
    generated = manifests()
    assert set(generated) == set(registered_names())
    assert generated["CWLaser"]["ports"] == {"inputs": {}, "outputs": {"out": "optical"}}


def test_a_configurable_port_count_is_reported_as_a_default_instance_has_it() -> None:
    """The ports a block will have when someone drops one on a canvas.

    A component whose port count is an argument binds its ports in ``__init__``,
    so the class attribute is empty for exactly the components an editor most
    needs to draw — a combiner used to report no inputs at all, and could not be
    drawn or wired. Reading them off a default instance reports the two it will
    actually have.
    """
    generated = manifests()
    assert generated["Combiner"]["ports"]["inputs"] == {"in0": "optical", "in1": "optical"}
    assert generated["Splitter"]["ports"]["outputs"] == {"out0": "optical", "out1": "optical"}


def test_structural_arguments_are_listed_apart_from_parameters() -> None:
    """They are not the same kind of thing and an editor must not treat them alike.

    A parameter changes a number and can be edited in place; changing the port
    count invalidates every wire already drawn to the block. The manifest keeps
    them in separate sections so that distinction survives into the interface.
    """
    generated = manifests()
    assert generated["Combiner"]["structural"] == {"num_inputs": 2}
    assert generated["Splitter"]["structural"] == {"num_outputs": 2}
    assert generated["CWLaser"]["structural"] == {}
    assert "num_inputs" not in generated["Combiner"]["parameters"]


def test_port_types_are_one_vocabulary_on_both_sides() -> None:
    """The editor decides whether a wire is legal by comparing these strings.

    It refuses a connection when the source's output type is not equal to the
    target's input type. That comparison is only meaningful if both sides are
    spelled from the same small vocabulary — one side emitting "PortType.OPTICAL"
    while the other emits "optical" would refuse every connection ever
    attempted, and refuse it for a reason nobody could see.
    """
    vocabulary = {kind.value for kind in PortType}
    for name, manifest in manifests().items():
        for side in ("inputs", "outputs"):
            for port, kind in manifest["ports"][side].items():
                assert kind in vocabulary, f"{name}.{port} has port type {kind!r}"


def test_a_manifest_describes_something_that_can_actually_be_built() -> None:
    """Every entry in the palette must be constructible with no arguments.

    Clicking a component adds it with its defaults, so a component that cannot
    be built that way would put an entry in the palette that fails the moment it
    is used. The manifest survives such a component — it falls back to the class
    — but the palette should not contain one unnoticed.
    """
    for name in registered_names():
        lookup(name)(label="probe")


# --------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------


def test_saving_and_loading_reproduces_the_results_exactly(tmp_path: Path) -> None:
    """The property that makes a project file worth having: a saved experiment
    re-run later must give the same answer, bit for bit."""
    original = _link_graph()
    before = original.run()[next(c for c in original.components if isinstance(c, BERAnalyzer))]

    reloaded = load(save(original, tmp_path / "link.maiman"))
    analyzer = next(c for c in reloaded.components if isinstance(c, BERAnalyzer))
    after = reloaded.run()[analyzer]

    assert after.q_factor == before.q_factor
    assert after.errors == before.errors
    assert after.threshold == before.threshold


def test_round_trip_preserves_structure(tmp_path: Path) -> None:
    original = _link_graph()
    reloaded = load(save(original, tmp_path / "link.maiman"))

    assert [c.label for c in reloaded.components] == [c.label for c in original.components]
    assert reloaded.edges.keys() == original.edges.keys()
    assert reloaded.ctx == original.ctx


def test_structural_config_round_trips(tmp_path: Path) -> None:
    """A combiner's port count is not a parameter — it changes the shape of the
    component — so it is stored separately and must survive the trip."""
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=4, sequence_length=16)
    g = Graph(ctx)
    for i in range(3):
        g.add(CWLaser(wavelength=1550.0 + i, label=f"ch{i}"))
    mux = g.add(Combiner(3, label="mux"))
    meter = g.add(PowerMeter(label="meter"))
    for i in range(3):
        g.connect(g.components[i], mux[f"in{i}"])
    g.connect(mux, meter)

    reloaded = load(save(g, tmp_path / "wdm.maiman"))
    restored = next(c for c in reloaded.components if isinstance(c, Combiner))

    assert restored.num_inputs == 3
    assert set(restored.inputs) == {"in0", "in1", "in2"}
    assert len(reloaded.run()[reloaded.components[-1]].bands) == 3


def test_the_file_is_readable_json(tmp_path: Path) -> None:
    path = save(_link_graph(), tmp_path / "link.maiman")
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["schema_version"] == SCHEMA_VERSION
    assert "maiman_version" in data
    assert data["context"]["bit_rate"] == 10e9
    assert {node["id"] for node in data["nodes"]} == {
        "prbs",
        "driver",
        "laser",
        "mzm",
        "fiber",
        "pin",
        "lpf",
        "ber",
    }
    assert {"from": ["mzm", "out"], "to": ["fiber", "in"]} in data["edges"]


def test_only_explicitly_set_parameters_are_stored() -> None:
    """A file records the choices its author made, not the defaults they
    accepted. That keeps diffs about the physics."""
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=4, sequence_length=16)
    g = Graph(ctx)
    laser = g.add(CWLaser(power=3.0, label="laser"))  # wavelength/linewidth left at default
    meter = g.add(PowerMeter(label="meter"))
    g.chain(laser, meter)

    node = next(n for n in graph_to_dict(g)["nodes"] if n["id"] == "laser")
    assert node["params"] == {"power": 3.0}


def test_ui_data_is_separate_from_the_physics(tmp_path: Path) -> None:
    """Node positions must not appear anywhere a physics diff would show them,
    and a project without them must still run."""
    g = _link_graph()
    positions = {"laser": {"x": 100.0, "y": 200.0}, "fiber": {"x": 300.0, "y": 200.0}}
    path = save(g, tmp_path / "link.maiman", ui=positions)
    data = json.loads(path.read_text(encoding="utf-8"))

    assert ui_from_dict(data) == positions
    for node in data["nodes"]:
        assert "x" not in node.get("params", {})

    # Strip the UI section entirely: the project still loads and still runs.
    for node in data["nodes"]:
        node.pop("ui", None)
    stripped = tmp_path / "headless.maiman"
    stripped.write_text(json.dumps(data), encoding="utf-8")

    reloaded = load(stripped)
    analyzer = next(c for c in reloaded.components if isinstance(c, BERAnalyzer))
    assert reloaded.run()[analyzer].q_factor > 0


# --------------------------------------------------------------------------
# Rejecting broken files, by name
# --------------------------------------------------------------------------


def test_a_file_without_a_schema_version_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.maiman"
    path.write_text(json.dumps({"nodes": [], "edges": []}), encoding="utf-8")
    with pytest.raises(ProjectError, match="no schema_version"):
        load(path)


def test_a_future_schema_version_is_rejected(tmp_path: Path) -> None:
    """Silently reading a newer file would be worse than refusing: the fields it
    means differently would be interpreted with today's meanings."""
    path = tmp_path / "future.maiman"
    path.write_text(
        json.dumps({"schema_version": SCHEMA_VERSION + 1, "context": {}, "nodes": []}),
        encoding="utf-8",
    )
    with pytest.raises(ProjectError, match="schema version"):
        load(path)


def test_malformed_json_is_reported_as_such(tmp_path: Path) -> None:
    path = tmp_path / "broken.maiman"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ProjectError, match="not valid JSON"):
        load(path)


def test_an_unknown_component_type_is_reported_by_name(tmp_path: Path) -> None:
    path = tmp_path / "plugin.maiman"
    path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "context": {
                    "bit_rate": 10e9,
                    "samples_per_symbol": 4,
                    "sequence_length": 16,
                    "seed": 0,
                    "precision": "single",
                },
                "nodes": [{"id": "x", "type": "QuantumFluxLaser", "params": {}}],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(UnknownComponentError, match="QuantumFluxLaser"):
        load(path)


def test_an_edge_to_a_missing_node_is_rejected(tmp_path: Path) -> None:
    data = graph_to_dict(_link_graph())
    data["edges"].append({"from": ["ghost", "out"], "to": ["fiber", "in"]})
    path = tmp_path / "dangling.maiman"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ProjectError, match="unknown node 'ghost'"):
        load(path)


def test_an_out_of_range_parameter_is_caught_on_load(tmp_path: Path) -> None:
    """Validation is not skipped just because the values came from a file."""
    data = graph_to_dict(_link_graph())
    next(n for n in data["nodes"] if n["id"] == "laser")["params"]["wavelength"] = 50.0
    path = tmp_path / "invalid.maiman"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ProjectError, match="below the minimum"):
        load(path)


def test_port_type_checking_still_applies_on_load(tmp_path: Path) -> None:
    data = graph_to_dict(_link_graph())
    data["edges"] = [{"from": ["prbs", "out"], "to": ["fiber", "in"]}]
    path = tmp_path / "mistyped.maiman"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ProjectError, match="port types differ"):
        load(path)


def test_a_parameter_may_declare_a_set_instead_of_a_range() -> None:
    """Some parameters are legal at a handful of values and nowhere between.

    A PRBS order is 7, 9, 11, 15, 23 or 31 — the polynomials the tap table
    holds. Declaring that as ``min=7, max=31`` said something false, and the
    editor believed it: the field read "range 7 … 31" and accepted 12, which
    only the engine knew was not a polynomial.
    """
    with pytest.raises(ValueError, match="is not one of 7, 9, 11, 15, 23, 31"):
        PRBSGenerator(order=12.0)
    assert PRBSGenerator(order=15.0).order == 15.0


def test_the_choices_come_from_the_table_that_decides_them() -> None:
    """Not restated. A second list is a list that goes stale."""
    generated = manifests()
    assert generated["PRBSGenerator"]["parameters"]["order"]["choices"] == [
        float(n) for n in sorted(PRBS_TAPS)
    ]
    assert generated["QAMMapper"]["parameters"]["bits_per_symbol"]["choices"] == list(QAM_FORMATS)


def test_every_declared_choice_actually_works() -> None:
    """The strongest form of the claim: each offered value builds a constellation.

    A list the interface offers is a promise. This runs it — an odd order above
    1 is a cross constellation nobody has implemented, so if one ever appeared
    in QAM_FORMATS the promise would be broken at the click and not before.
    """
    for bits in QAM_FORMATS:
        points = qam_constellation(int(bits))
        assert points.shape == (1 << int(bits),)


def test_a_default_outside_its_own_choices_is_refused() -> None:
    """A parameter that starts at a value it will not accept is a trap."""
    with pytest.raises(ValueError, match="not one of the declared choices"):
        Param(3.0, choices=(1.0, 2.0, 4.0))


def test_settings_that_disagree_are_caught_before_anything_runs() -> None:
    """Two parameters, each legal alone, that cannot both hold.

    Differential encoding relabels the alphabet by quadrant, and BPSK has no
    quadrants. Neither a range nor a set of choices on either parameter can see
    that, so the component says it itself — and says it during validation, so
    the run is refused rather than abandoned partway through.
    """
    from maiman.components import QAMMapper

    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=4, sequence_length=32, seed=1)
    graph = Graph(ctx)
    prbs = graph.add(PRBSGenerator(order=7.0, bits_per_symbol=1.0, label="prbs"))
    graph.connect(
        prbs, graph.add(QAMMapper(bits_per_symbol=1.0, differential=True, label="map"))["in"]
    )

    with pytest.raises(ValueError, match="no quadrants"):
        graph.run()

    # Turning it off is the fix the message names, and it then runs.
    graph.run(overrides={("map", "differential"): False})


def test_validation_happens_before_the_first_block_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not merely before the offending block: before any of them.

    A settings problem found halfway through a run has already spent the work,
    and reports itself further from its cause than it needs to.

    Patched rather than subclassed. Defining a Component subclass registers it
    — that is how plugins work — so a throwaway one in a test stays in the
    registry for every test after it, and the palette check downstream then
    sees a component that exists nowhere else. It did.
    """
    from maiman.components import QAMMapper

    ran: list[str] = []
    original = PRBSGenerator.run

    def watched(self: PRBSGenerator, ctx: object, inputs: object) -> object:
        ran.append(self.label)
        return original(self, ctx, inputs)  # type: ignore[arg-type]

    monkeypatch.setattr(PRBSGenerator, "run", watched)

    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=4, sequence_length=32, seed=1)
    graph = Graph(ctx)
    prbs = graph.add(PRBSGenerator(order=7.0, bits_per_symbol=1.0, label="prbs"))
    graph.connect(
        prbs, graph.add(QAMMapper(bits_per_symbol=1.0, differential=True, label="map"))["in"]
    )

    with pytest.raises(ValueError, match="no quadrants"):
        graph.run()
    assert ran == [], "a block ran before the settings were checked"
