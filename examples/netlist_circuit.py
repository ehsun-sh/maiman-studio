"""Solve a circuit that a layout tool drew, with a PDK's numbers in it.

A layout tool knows the topology and the geometry. It does not know what any of
it does optically -- gdsfactory's ``CrossSection`` carries width, offset, layer
and bend radius, and forbids extra fields, so there is no effective index
anywhere in it. This library is the other half.

So: the netlist says what is wired to what and how long the router made each
piece, the ``.pdk`` says which model each layout cell is and what this process
measures, and :meth:`maiman.circuit.Circuit.solve` does the rest.

Two circuits. The first is a file **gdsfactory actually emitted**, shipped
unmodified in ``tests/data/``; its answer is checked against the loss a hand
calculation gives. The second has feedback in it, which is the case a chain of
transfer functions cannot do and a scattering-matrix reduction can.

Nothing here imports gdsfactory. See ``src/maiman/netlist.py``.

Run: ``python examples/netlist_circuit.py``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from maiman.netlist import (
    Netlist,
    NetlistError,
    circuit_from_netlist,
    load_netlist,
    missing_cells,
    parse_netlist,
)
from maiman.pdk import PDK, PDKError, load_pdk
from maiman.units import C_LIGHT

ROOT = Path(__file__).resolve().parent.parent
KIT = ROOT / "examples" / "silicon_220nm.pdk.json"
FROM_GDSFACTORY = ROOT / "tests" / "data" / "gdsfactory_straight_with_bend.yml"
RACETRACK = ROOT / "examples" / "ring_racetrack.netlist.yml"

REFERENCE = C_LIGHT / 1550e-9
LOOP_UM = 100.0
N_GROUP = 4.20
LOSS_DB_PER_CM = 2.0


def read(path: Path) -> Netlist:
    """Load a netlist. YAML is what a layout tool writes, so that is what is shipped."""
    return load_netlist(path)


def notches(frequencies: np.ndarray, power: np.ndarray) -> np.ndarray:
    threshold = power.min() + 0.5 * (power.max() - power.min())
    interior = np.arange(1, power.size - 1)
    dips = interior[
        (power[1:-1] < power[:-2]) & (power[1:-1] <= power[2:]) & (power[1:-1] < threshold)
    ]
    return frequencies[dips]


def a_file_gdsfactory_wrote(kit: PDK) -> None:
    print("A netlist gdsfactory emitted, unmodified")
    netlist = read(FROM_GDSFACTORY)
    print(f"  {netlist}")
    print(f"  cells: {', '.join(netlist.cells())}")
    print(f"  cells this kit does not model: {missing_cells(netlist, kit) or 'none'}")

    for name, instance in netlist.instances.items():
        print(f"    {name[:44]:46s} {instance.cell:12s} {instance.info.get('length')} um")

    frequencies = np.linspace(REFERENCE - 1.2e12, REFERENCE + 1.2e12, 2001)
    solved = circuit_from_netlist(netlist, kit, frequencies).solve()
    measured = 10.0 * np.log10(solved.power("o2", "o1").mean())

    total = sum(float(i.info["length"]) for i in netlist.instances.values())
    expected = -LOSS_DB_PER_CM * total * 1e-4
    print(f"\n  transmission  {measured:+.4f} dB")
    print(f"  {total:.3f} um at {LOSS_DB_PER_CM:g} dB/cm = {expected:+.4f} dB")
    print(
        "\n  The bend contributes 16.637 um, which is its *arc* length and is not\n"
        "  recoverable from its radius without the Euler parameter. The kit asks for\n"
        "  it with `from_netlist: {length: info.length}` -- so every routed bend and\n"
        "  connecting straight carries the length the router gave that instance,\n"
        "  rather than one nominal value standing in for all of them.\n"
    )


def a_circuit_with_feedback_in_it(kit: PDK) -> None:
    print("A racetrack ring, hand-written in the same schema")
    netlist = read(RACETRACK)
    print(f"  {netlist}")

    frequencies = np.linspace(REFERENCE - 8e11, REFERENCE + 8e11, 200001)
    solved = circuit_from_netlist(netlist, kit, frequencies).solve()
    power = solved.power("through", "in")
    found = notches(frequencies, power)

    spacing = float(np.mean(np.diff(found)))
    predicted = C_LIGHT / (N_GROUP * LOOP_UM * 1e-6)
    print(f"\n  {len(found)} resonances in 1.6 THz")
    print(f"  free spectral range   {spacing / 1e9:8.2f} GHz")
    print(f"  c / (n_g * L)         {predicted / 1e9:8.2f} GHz")
    print(f"  notch depth           {10 * np.log10(power.min()):8.2f} dB")
    print(
        "\n  The loop returns into the coupler it left, so no order of running the\n"
        "  five blocks produces this: the answer at every port depends on the answer\n"
        "  at every other one at the same time. That is what the reduction is for.\n"
    )


def what_it_refuses(kit: PDK) -> None:
    print("What it refuses, and why refusing beats carrying on")
    broken = {
        "name": "unmapped",
        "instances": {"x": {"component": "grating_coupler_elliptical", "info": {}, "settings": {}}},
        "ports": {"o1": "x,o1"},
    }
    try:
        circuit_from_netlist(parse_netlist(broken), kit, np.array([REFERENCE]))
    except (NetlistError, PDKError) as error:
        print(f"  unmapped cell -> {type(error).__name__}")
        print(f"    {error}")
    print(
        "\n  A reader that skipped the cells it did not recognise would hand back a\n"
        "  circuit that solves, plots like a spectrum, and is not the circuit on the\n"
        "  mask. That is the worst of the three things that can happen."
    )


def main() -> None:
    kit = load_pdk(KIT)
    print(f"{kit}\n")
    try:
        a_file_gdsfactory_wrote(kit)
    except NetlistError as error:
        if "YAML" not in str(error):
            raise
        print(f"  {error}")
        return
    a_circuit_with_feedback_in_it(kit)
    what_it_refuses(kit)
    print(
        "\nWhat the netlist cannot give, and the PDK has to: every optical number.\n"
        "A layout cross-section is width, offset, layer and bend radius. The kit is\n"
        "where n_eff, n_group, loss and dispersion come from, and where this process\n"
        "says that its `straight` is a Waveguide and its `o1` is that model's `in`."
    )


if __name__ == "__main__":
    main()
