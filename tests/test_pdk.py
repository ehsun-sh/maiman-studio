"""Process design kits: what they carry, and what they refuse to carry.

A PDK is read once and used a hundred times, so almost everything here is about
*refusal*. A typo in a kit that surfaces as a wrong number three circuits later
is the failure mode worth engineering against; a typo that stops the load with
the file and the entry named costs a minute.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pytest

from maiman import Graph, SimulationContext
from maiman.components import (
    MMI,
    CWLaser,
    DirectionalCoupler,
    MachZehnderInterferometer,
    PowerMeter,
    RingResonator,
    Waveguide,
)
from maiman.pdk import (
    CROSS_SECTION_PARAMETERS,
    SCHEMA_VERSION,
    Fit,
    PDKError,
    load_pdk,
    pdk_from_dict,
)

ROOT = Path(__file__).resolve().parent.parent
SHIPPED = ROOT / "examples" / "silicon_220nm.pdk.json"


def kit(**overrides: Any) -> dict[str, Any]:
    """A minimal, valid kit, which each test then breaks in exactly one way."""
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "name": "test-kit",
        "reference_wavelength": 1550.0,
        "valid_wavelengths": [1500.0, 1600.0],
        "cross_sections": {"strip": {"n_eff": 2.44, "n_group": 4.2, "propagation_loss": 2.0}},
        "devices": {
            "wg": {"component": "Waveguide", "cross_section": "strip", "params": {"length": 100.0}},
            "dc": {"component": "DirectionalCoupler", "params": {"coupling": [0.48, 3.9e-3]}},
        },
    }
    document.update(overrides)
    return document


# ---------------------------------------------------------------------------
# fits


def test_a_bare_number_is_a_constant_fit() -> None:
    """Because most parameters are one, and writing ``[0.5]`` for that is noise."""
    fit = Fit.parse(0.5, where="test")
    assert fit.is_constant
    assert fit.at(1500.0, 1550.0) == 0.5
    assert fit.at(1600.0, 1550.0) == 0.5


def test_a_fit_is_a_polynomial_in_the_detuning_in_nanometres() -> None:
    """Nanometres because that is the unit a foundry quotes a fit in.

    Converting on the way in would mean every coefficient in every kit had to be
    rescaled by whoever wrote it, which is a conversion done once per file by
    hand instead of once in the loader.
    """
    fit = Fit.parse([2.44, -1.13e-3, 5e-7], where="test")
    for wavelength in (1500.0, 1550.0, 1600.0):
        delta = wavelength - 1550.0
        assert fit.at(wavelength, 1550.0) == pytest.approx(2.44 - 1.13e-3 * delta + 5e-7 * delta**2)


@pytest.mark.parametrize("value", ["0.5", None, [], [1.0, "x"], {"a": 1}, True])
def test_a_fit_that_is_not_a_fit_is_refused(value: Any) -> None:
    """Including ``True``, which is an int in Python and is not a coupling ratio."""
    with pytest.raises(PDKError, match="expected a number"):
        Fit.parse(value, where="somewhere")


# ---------------------------------------------------------------------------
# what a kit refuses


def test_a_document_without_a_schema_version_is_not_a_pdk() -> None:
    document = kit()
    del document["schema_version"]
    with pytest.raises(PDKError, match="no schema_version"):
        pdk_from_dict(document)


def test_a_future_schema_is_refused_rather_than_guessed_at() -> None:
    with pytest.raises(PDKError, match="schema version"):
        pdk_from_dict(kit(schema_version=SCHEMA_VERSION + 1))


def test_a_device_naming_a_component_this_build_has_never_heard_of() -> None:
    document = kit()
    document["devices"]["dc"]["component"] = "PlasmonicNanoantenna"
    with pytest.raises(PDKError, match="PlasmonicNanoantenna"):
        pdk_from_dict(document)


def test_a_device_setting_a_parameter_its_component_does_not_declare() -> None:
    """The single most likely typo in a hand-written kit, and it names both sides."""
    document = kit()
    document["devices"]["dc"]["params"]["couplng"] = 0.5
    with pytest.raises(PDKError, match=r"couplng.*DirectionalCoupler does not declare"):
        pdk_from_dict(document)


def test_a_device_drawn_in_a_cross_section_that_does_not_exist() -> None:
    document = kit()
    document["devices"]["wg"]["cross_section"] = "ridge"
    with pytest.raises(PDKError, match="ridge"):
        pdk_from_dict(document)


def test_a_cross_section_setting_something_that_is_not_a_waveguide_property() -> None:
    """A cross-section describes the guide, not the device drawn in it.

    Letting ``coupling`` live on a cross-section would make one number apply to
    every device sharing that waveguide, which is not what a cross-section means
    and is a mistake that would look like it worked.
    """
    document = kit()
    document["cross_sections"]["strip"]["coupling"] = 0.5
    with pytest.raises(PDKError, match="not a waveguide property"):
        pdk_from_dict(document)
    assert "coupling" not in CROSS_SECTION_PARAMETERS


def test_a_cross_section_on_a_device_with_no_waveguide_in_it() -> None:
    """An MMI is a lumped matrix. Giving it an index is a statement nothing can honour.

    Dropping it quietly would be worse than refusing: the kit would look as
    though it had specified something it had not, and the number would be absent
    from the model without being absent from the file.
    """
    document = kit()
    document["devices"]["mmi"] = {
        "component": "MMI",
        "cross_section": "strip",
        "structural": {"ports": 2},
    }
    with pytest.raises(PDKError, match="no waveguide in it"):
        pdk_from_dict(document)


def test_a_reference_wavelength_outside_the_kit_s_own_window() -> None:
    with pytest.raises(PDKError, match="outside the kit's own"):
        pdk_from_dict(kit(reference_wavelength=1310.0))


def test_a_window_that_is_not_a_window() -> None:
    for bad in ([1600.0, 1500.0], [1550.0], "C band", [1500.0, 1600.0, 1700.0]):
        with pytest.raises(PDKError):
            pdk_from_dict(kit(valid_wavelengths=bad))


# ---------------------------------------------------------------------------
# what a kit does


def test_the_fits_are_evaluated_where_you_ask() -> None:
    pdk = pdk_from_dict(kit())
    assert pdk.parameters("dc")["coupling"] == pytest.approx(0.48)
    assert pdk.parameters("dc", wavelength=1600.0)["coupling"] == pytest.approx(0.48 + 3.9e-3 * 50)
    assert pdk.parameters("dc", wavelength=1500.0)["coupling"] == pytest.approx(0.48 - 3.9e-3 * 50)


def test_evaluating_outside_the_window_is_refused_not_extrapolated() -> None:
    """The reason this exists at all, stated as a number.

    A first-order C-band fit of this coupler, run out to 1310 nm, returns minus
    0.46 — a negative power fraction out of arithmetic that never complained.
    The check is what turns that into a sentence naming the window.
    """
    pdk = pdk_from_dict(kit())
    raw = Fit.parse([0.48, 3.9e-3], where="t").at(1310.0, 1550.0)
    assert raw < 0.0, "the premise of this test: the extrapolation really is nonsense"

    with pytest.raises(PDKError, match="1500-1600 nm and was asked for 1310"):
        pdk.parameters("dc", wavelength=1310.0)
    with pytest.raises(PDKError, match="fitted over"):
        pdk.make("dc", wavelength=1310.0)


def test_a_device_may_narrow_the_window_the_kit_declares() -> None:
    """A kit characterised across the C band can still carry one device measured over ten nm."""
    document = kit()
    document["devices"]["dc"]["valid_wavelengths"] = [1545.0, 1555.0]
    pdk = pdk_from_dict(document)

    assert pdk.parameters("dc", wavelength=1552.0)["coupling"] > 0.0
    assert pdk.parameters("wg", wavelength=1595.0)["n_eff"] == pytest.approx(2.44)
    with pytest.raises(PDKError, match="1545-1555"):
        pdk.parameters("dc", wavelength=1595.0)


def test_the_cross_section_reaches_every_device_drawn_in_it() -> None:
    """Which is the point of having one: define the strip once, not per block."""
    pdk = pdk_from_dict(kit())
    waveguide = pdk.make("wg", label="w")
    assert isinstance(waveguide, Waveguide)
    assert waveguide.n_eff == pytest.approx(2.44)
    assert waveguide.n_group == pytest.approx(4.2)
    assert waveguide.propagation_loss == pytest.approx(2.0)
    assert waveguide.length == pytest.approx(100.0)


def test_the_component_is_told_where_its_indices_were_quoted() -> None:
    """Or its own frequency expansion would be anchored somewhere else than the fit."""
    pdk = pdk_from_dict(kit())
    at_reference = pdk.make("wg")
    detuned = pdk.make("wg", wavelength=1580.0)
    assert isinstance(at_reference, Waveguide) and isinstance(detuned, Waveguide)
    assert at_reference.reference_wavelength == pytest.approx(1550.0)
    assert detuned.reference_wavelength == pytest.approx(1580.0)


def test_an_override_wins_over_the_kit() -> None:
    """A PDK is where a design starts, not a cage.

    Sweeping a foundry's nominal coupling to see what a process corner would
    cost is exactly the thing this has to make easy.
    """
    pdk = pdk_from_dict(kit())
    nominal = pdk.make("dc")
    overridden = pdk.make("dc", coupling=0.5)
    assert isinstance(nominal, DirectionalCoupler) and isinstance(overridden, DirectionalCoupler)
    assert nominal.coupling == pytest.approx(0.48)
    assert overridden.coupling == pytest.approx(0.5)


def test_structural_arguments_reach_the_constructor() -> None:
    """Port count is not a parameter — it decides the shape of the block."""
    document = kit()
    document["devices"]["splitter"] = {
        "component": "MMI",
        "structural": {"ports": 4, "driven": 1},
        "params": {"excess_loss": 0.4},
    }
    splitter = pdk_from_dict(document).make("splitter", label="s")
    assert isinstance(splitter, MMI)
    assert list(splitter.outputs) == ["out1", "out2", "out3", "out4"]
    assert list(splitter.inputs) == ["in1"]
    assert splitter.excess_loss == pytest.approx(0.4)


def test_asking_for_a_device_that_is_not_in_the_kit_lists_the_ones_that_are() -> None:
    with pytest.raises(PDKError, match=r"has no device 'nope'.*\['dc', 'wg'\]"):
        pdk_from_dict(kit()).make("nope")


# ---------------------------------------------------------------------------
# nothing is executed


def test_reading_a_kit_runs_nothing_of_its_author_s() -> None:
    """A `.pdk` is JSON, and a component name is looked up in the registry.

    The same rule the project format follows: opening a file somebody sent you
    must not be equivalent to running their code. A dotted path here would be a
    remote-execution hole in a design tool.
    """
    document = kit()
    document["devices"]["dc"]["component"] = "os.system"
    with pytest.raises(PDKError, match=re.escape("os.system")):
        pdk_from_dict(document)


# ---------------------------------------------------------------------------
# the kit that ships


def test_the_shipped_kit_loads_and_every_device_in_it_builds() -> None:
    """It is documentation, and documentation that does not run is a claim.

    Every device is built at three wavelengths across the declared window, so a
    fit whose slope carries a parameter out of its own valid range is caught
    here rather than by whoever copies the file.
    """
    pdk = load_pdk(SHIPPED)
    assert pdk.devices, "the shipped kit is empty"
    low, high = pdk.valid_wavelengths or (0.0, math.inf)

    for name in pdk.devices:
        for wavelength in (low, pdk.reference_wavelength, high):
            component = pdk.make(name, wavelength=wavelength, label="probe")
            assert component.label == "probe"


def test_the_shipped_kit_is_the_file_the_readme_points_at() -> None:
    """And it is JSON, parseable by anything, with no code path involved."""
    data = json.loads(SHIPPED.read_text(encoding="utf-8"))
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["valid_wavelengths"] == [1500.0, 1600.0]
    assert {"strip", "rib", "nitride"} <= set(data["cross_sections"])


def test_the_shipped_kit_builds_each_kind_of_block_it_claims_to() -> None:
    pdk = load_pdk(SHIPPED)
    assert isinstance(pdk.make("wg_strip"), Waveguide)
    assert isinstance(pdk.make("dc_3db"), DirectionalCoupler)
    assert isinstance(pdk.make("mmi_2x2"), MMI)
    assert isinstance(pdk.make("mzi_switch"), MachZehnderInterferometer)
    assert isinstance(pdk.make("ring_addrop_10um"), RingResonator)


def test_the_nitride_guide_really_is_the_quiet_one() -> None:
    """A kit is only useful if its cross-sections differ where the process does.

    Nitride is the standard low-loss platform and silicon strip is not; if these
    two came back the same, the cross-sections would be decoration.
    """
    pdk = load_pdk(SHIPPED)
    strip = pdk.parameters("wg_strip")
    nitride = pdk.parameters("wg_nitride")
    assert nitride["propagation_loss"] < strip["propagation_loss"] / 10.0
    assert nitride["n_eff"] < strip["n_eff"]


def test_a_kit_device_runs_in_a_link() -> None:
    """The end of the whole point: foundry numbers, in a graph, producing a number.

    A 1 x 4 MMI from the shipped kit, with its excess loss and its imbalance, so
    the four arms are *not* equal — which is exactly what a real splitter tree
    does and what a textbook one does not.
    """
    pdk = load_pdk(SHIPPED)
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=64, seed=1)
    graph = Graph(ctx)
    laser = graph.add(CWLaser(power=0.0, wavelength=1550.0, label="tx"))
    splitter = pdk.make("mmi_1x4", label="mmi")
    graph.add(splitter)
    meters = [graph.add(PowerMeter(label=f"pm{k}")) for k in range(4)]
    graph.connect(laser, splitter["in1"])
    for index, meter in enumerate(meters):
        graph.connect(splitter[f"out{index + 1}"], meter["in"])

    results = graph.run()
    levels = [results[meter].power_dbm for meter in meters]

    # Below the ideal -6.02 dB, by the kit's own excess loss.
    excess = pdk.parameters("mmi_1x4")["excess_loss"]
    assert max(levels) < -10.0 * math.log10(4.0)
    assert sum(10.0 ** (level / 10.0) for level in levels) == pytest.approx(
        10.0 ** (-excess / 10.0), rel=1e-6
    )
    # And the arms are not equal, because the kit says the device is imbalanced.
    spread = max(levels) - min(levels)
    assert spread == pytest.approx(pdk.parameters("mmi_1x4")["imbalance"], abs=1e-6)
