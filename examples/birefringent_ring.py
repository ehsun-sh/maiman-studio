"""A silicon ring seen by both of its guided modes.

A 220 nm-thick strip confines TE strongly and TM weakly, so the two modes have
different effective and group indices — 2.44/4.20 against 1.78/3.80 at 1550 nm.
That makes a ring resonate at two sets of wavelengths on two different free
spectral ranges, which turns it into a polarization-selective filter whether its
designer wanted one or not.

Two tables. The first is the device, solved as a scattering matrix: where each
comb sits and how far apart its teeth are, against ``c / (n_g L)``. The second
is the same ring as a block on a graph, with light launched at 45 degrees so
there is something on both axes for it to treat differently.

Run: ``python examples/birefringent_ring.py``
"""

from __future__ import annotations

import numpy as np

from maiman import Graph, SimulationContext
from maiman.circuit import Circuit, SMatrix
from maiman.components import CWLaser, PolarizationRotator, RingResonator
from maiman.photonics import (
    POLARIZATIONS,
    SILICON_STRIP_NEFF,
    SILICON_STRIP_NEFF_TM,
    SILICON_STRIP_NGROUP,
    SILICON_STRIP_NGROUP_TM,
    directional_coupler,
    dual_polarization,
    expose_polarizations,
    link_polarizations,
    straight_waveguide,
)
from maiman.units import C_LIGHT

REFERENCE = C_LIGHT / 1550e-9
LENGTH_UM = 100.0
LENGTH = LENGTH_UM * 1e-6
COUPLING = 0.05
LOSS_DB_PER_M = 200.0

INDICES = {
    "te": (SILICON_STRIP_NEFF, SILICON_STRIP_NGROUP),
    "tm": (SILICON_STRIP_NEFF_TM, SILICON_STRIP_NGROUP_TM),
}


def solved_ring(frequencies: np.ndarray) -> SMatrix:
    """The ring assembled from dual-polarization devices and reduced.

    Assembled rather than written down, the same way the single-polarization
    ring is: a coupler, a loop of waveguide, and the reduction. Nothing in
    :mod:`maiman.circuit` knows this circuit carries two modes — the ports are
    named, so two modes are simply twice as many ports.
    """
    coupler = dual_polarization(
        dict.fromkeys(POLARIZATIONS, directional_coupler(frequencies, coupling=COUPLING))
    )
    loop = dual_polarization(
        {
            name: straight_waveguide(
                frequencies,
                length=LENGTH,
                n_eff=n_eff,
                n_group=n_group,
                reference_frequency=REFERENCE,
                loss_db_per_m=LOSS_DB_PER_M,
            )
            for name, (n_eff, n_group) in INDICES.items()
        }
    )
    circuit = Circuit().add("c", coupler).add("w", loop)
    link_polarizations(circuit, "c", "out2", "w", "in")
    link_polarizations(circuit, "w", "out", "c", "in2")
    expose_polarizations(circuit, "in", "c", "in1")
    expose_polarizations(circuit, "through", "c", "out1")
    return circuit.solve()


def notches(frequencies: np.ndarray, power: np.ndarray) -> np.ndarray:
    threshold = power.min() + 0.5 * (power.max() - power.min())
    interior = np.arange(1, power.size - 1)
    dips = interior[
        (power[1:-1] < power[:-2]) & (power[1:-1] <= power[2:]) & (power[1:-1] < threshold)
    ]
    return frequencies[dips]


def the_device() -> list[float]:
    print(f"A {LENGTH_UM:g} um silicon ring, both guided modes")
    print(f"{'mode':>6} {'n_eff':>8} {'n_group':>9} {'FSR':>11} {'c/(n_g L)':>12} {'depth':>8}")

    frequencies = np.linspace(REFERENCE - 1.5e12, REFERENCE + 1.5e12, 60001)
    solved = solved_ring(frequencies)
    resonances: list[float] = []
    for name, (n_eff, n_group) in INDICES.items():
        power = solved.power(f"through@{name}", f"in@{name}")
        found = notches(frequencies, power)
        spacing = float(np.mean(np.diff(found)))
        print(
            f"{name.upper():>6} {n_eff:8.2f} {n_group:9.2f} {spacing / 1e9:10.1f}G "
            f"{C_LIGHT / (n_group * LENGTH) / 1e9:11.1f}G {power.min():8.3f}"
        )
        resonances.append(float(C_LIGHT / found[len(found) // 2] * 1e9))

    print("\n  Both match their own group index, and only their own: using TE's for")
    print("  TM would put that row 75 GHz out, which is a tenth of a free spectral")
    print("  range and would still look like a plausible ring.\n")
    print(f"{'mode':>6} {'a resonance at':>16}")
    for name, wavelength in zip(INDICES, resonances, strict=True):
        print(f"{name.upper():>6} {wavelength:15.2f}nm")
    print()
    return resonances


def the_block(resonances: list[float]) -> None:
    print("The same ring as a block, launched at 45 degrees so both axes carry light")
    print(f"{'wavelength':>12} {'|Ex|^2':>12} {'|Ey|^2':>12}   what it is")

    labels = {round(resonances[0], 2): "TE resonance", round(resonances[1], 2): "TM resonance"}
    probes = sorted({*(round(r, 2) for r in resonances), 1550.0, 1554.0})
    for wavelength in probes:
        ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=64, seed=1)
        graph = Graph(ctx)
        laser = graph.add(CWLaser(power=0.0, wavelength=wavelength, label="tx"))
        rotator = graph.add(PolarizationRotator(angle=45.0, label="rot"))
        ring = graph.add(
            RingResonator(length=LENGTH_UM, coupling=COUPLING, birefringent=True, label="ring")
        )
        graph.connect(laser, rotator["in"])
        graph.connect(rotator["out"], ring["in"])
        band = graph.run(keep=[ring]).port(ring, "through").bands[0]
        ex = float(np.mean(np.abs(np.asarray(band.Ex)) ** 2))
        ey = float(np.mean(np.abs(np.asarray(band.Ey)) ** 2))
        print(
            f"{wavelength:11.2f}n {ex:12.4e} {ey:12.4e}   "
            f"{labels.get(round(wavelength, 2), 'between combs')}"
        )

    print(
        "\n  One component in the notch and the other passing is the whole effect:"
        "\n  the ring is a polarization filter. With `birefringent` off the two"
        "\n  columns are equal at every wavelength, which is where this library was."
    )


def main() -> None:
    the_block(the_device())
    print(
        "\nWhat is shared between the modes, and should not be: propagation loss and"
        "\ndispersion, and the coupler's split ratio. All three are polarization"
        "\ndependent in a real device and all three are per-process numbers this"
        "\nlibrary will not invent -- so the two combs come out at the right places"
        "\nwith the same notch depth, where a real pair would differ in both."
    )


if __name__ == "__main__":
    main()
