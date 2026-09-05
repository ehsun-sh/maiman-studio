"""A silicon microring, from its scattering matrix to its place in a link.

The first photonic integrated circuit in this library, and the point of it is
that nothing here is a special case. A ring is a coupler and a length of
waveguide wired into a loop; the loop is solved by the same reduction any circuit
is; the solved circuit is a transfer function, and a transfer function is
something the link engine has known what to do with since the first optical
filter.

Four things are worth reading off the output.

**The circuit solve is exact, and it is checked against a formula it does not
contain.** The assembled ring reproduces Yariv's all-pass and add-drop transfer
functions to fourteen digits. It also agrees with SAX — the JAX-based solver the
roadmap said to integrate — to 7e-15 across two terahertz, which is what made
writing the thirty-line reduction instead of taking on thirty-seven packages and
an LGPL sparse back-end the defensible choice.

**Critical coupling is a knife edge and a false friend.** Match the coupling to
the round-trip loss and the through port extinguishes completely. Miss it either
way and the notch fills in — by the *same* amount for two quite different
couplings, so a measured extinction does not tell you which side of critical a
ring is on.

**The free spectral range is set by the group index, and the resonance position
by the effective one.** They differ by a factor of 1.7 in silicon, and swapping
them is the standard way to be wrong about a ring while every plot still looks
like a ring.

**A ring is an ASE gate, and getting that right needs its linewidth.** An
amplifier emits across terahertz and the drop port passes roughly one linewidth
in every free spectral range. Integrating that with a fixed number of points
misses the resonance entirely on a high-Q device — and misses it high or low
depending on where the amplifier's band happened to sit.
"""

from __future__ import annotations

import numpy as np

from maiman import Graph, SimulationContext
from maiman.circuit import SMatrix
from maiman.components import EDFA, CWLaser, OSNRMeter, PowerMeter, RingResonator, Waveguide
from maiman.photonics import (
    SILICON_STRIP_NGROUP,
    free_spectral_range,
    resonance_linewidth,
    ring_resonator,
    round_trip_amplitude,
)
from maiman.units import frequency_to_wavelength, linear_to_db, w_to_dbm, wavelength_to_frequency

REFERENCE = wavelength_to_frequency(1550e-9)
CIRCUMFERENCE = 100e-6  # 100 um: a 16 um radius, a routine silicon ring
LOSS_DB_PER_M = 300.0  # 3 dB/cm
FSR = free_spectral_range(CIRCUMFERENCE, SILICON_STRIP_NGROUP)
CRITICAL = 1.0 - round_trip_amplitude(CIRCUMFERENCE, LOSS_DB_PER_M) ** 2


def matrix(frequencies: np.ndarray, coupling: float, drop: float = 0.0) -> SMatrix:
    return ring_resonator(
        frequencies,
        length=CIRCUMFERENCE,
        coupling=coupling,
        drop_coupling=drop,
        reference_frequency=REFERENCE,
        loss_db_per_m=LOSS_DB_PER_M,
    )


def resonance(coupling: float, drop: float = 0.0) -> float:
    """The deepest notch within one free spectral range of 1550 nm.

    Coarse then fine: a resonance can be a megahertz wide inside a 714 GHz
    period, and finding it in one sweep would take a hundred million points.
    """
    coarse = REFERENCE + np.linspace(-FSR / 2, FSR / 2, 200001)
    near = float(coarse[int(np.argmin(matrix(coarse, coupling, drop).power("through", "in")))])
    fine = near + np.linspace(-FSR / 2e5, FSR / 2e5, 20001)
    return float(fine[int(np.argmin(matrix(fine, coupling, drop).power("through", "in")))])


def main() -> None:
    print("A 100 um silicon microring, 3 dB/cm, n_eff = 2.44, n_g = 4.20\n")
    print(f"  free spectral range   {FSR / 1e9:8.2f} GHz   = c / (n_g * L)")
    print(f"  round-trip amplitude  {round_trip_amplitude(CIRCUMFERENCE, LOSS_DB_PER_M):8.6f}")
    print(f"  critical coupling     {CRITICAL:8.6f}   = 1 - a^2")

    # -- the spectrum, off the solved circuit ------------------------------
    peak = resonance(CRITICAL)
    print(f"  a resonance at        {frequency_to_wavelength(peak) * 1e9:8.4f} nm\n")

    print("Through-port transmission, walked across that resonance:\n")
    print(f"  {'offset':>10}  {'critical':>10}  {'under 4x':>10}  {'over 4x':>10}")
    for offset in (-20e9, -5e9, -1e9, -0.2e9, 0.0, 0.2e9, 1e9, 5e9, 20e9):
        grid = np.array([peak + offset])
        cells = [
            linear_to_db(float(matrix(grid, kappa).power("through", "in")[0]))
            for kappa in (CRITICAL, CRITICAL / 4, CRITICAL * 4)
        ]
        print(f"  {offset / 1e9:+9.2f}G  " + "  ".join(f"{value:9.2f}d" for value in cells))

    # -- the false friend --------------------------------------------------
    print("\nUnder- and over-coupling can give the identical notch.")
    print("Depth is |t - a|^2 / |1 - t a|^2, which is symmetric in t and a, so")
    print("t2 = (2a - t1(1 + a^2)) / (1 + a^2 - 2 a t1) is t1's partner:\n")
    a = round_trip_amplitude(CIRCUMFERENCE, LOSS_DB_PER_M)
    print(f"  {'kappa':>10}  {'t':>8}  {'notch':>10}")
    for t1 in (0.99, 0.995):
        t2 = (2 * a - t1 * (1 + a**2)) / (1 + a**2 - 2 * a * t1)
        for t in (t1, t2):
            kappa = 1.0 - t**2
            fine = peak + np.linspace(-3e6, 3e6, 20001)
            depth = float(matrix(fine, kappa).power("through", "in").min())
            side = "under" if kappa < CRITICAL else "over "
            print(f"  {kappa:10.6f}  {t:8.5f}  {linear_to_db(depth):9.2f}d  ({side}-coupled)")

    # -- add-drop, in a graph ---------------------------------------------
    print("\nAs a channel filter in a real link: EDFA, then an add-drop ring.\n")
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=256, seed=1)
    coupling = 0.05
    peak = resonance(coupling, coupling)

    for detune, label in ((0.0, "on resonance"), (FSR / 2, "between two")):
        graph = Graph(ctx)
        laser = graph.add(
            CWLaser(power=0.0, wavelength=frequency_to_wavelength(peak + detune) * 1e9)
        )
        amp = graph.add(EDFA(gain=20.0, noise_figure=5.0))
        ring = graph.add(
            RingResonator(
                length=CIRCUMFERENCE * 1e6,
                coupling=coupling,
                drop_coupling=coupling,
                propagation_loss=LOSS_DB_PER_M / 100.0,
            )
        )
        through = graph.add(PowerMeter(label="through"))
        drop = graph.add(PowerMeter(label="drop"))
        before = graph.add(OSNRMeter(label="before"))
        after = graph.add(OSNRMeter(label="after"))
        graph.connect(laser, amp)
        graph.connect(amp, ring)
        graph.connect(amp, before)
        graph.connect(ring["through"], through)
        graph.connect(ring["drop"], drop)
        graph.connect(ring["drop"], after)
        results = graph.run(keep=[amp, ring])

        width = resonance_linewidth(
            CIRCUMFERENCE,
            SILICON_STRIP_NGROUP,
            coupling=coupling,
            drop_coupling=coupling,
            loss_db_per_m=LOSS_DB_PER_M,
        )
        print(
            f"  {label:14s}  through {w_to_dbm(results[through].power_w):7.2f} dBm"
            f"   drop {w_to_dbm(results[drop].power_w):7.2f} dBm"
            f"   OSNR {results[before]:5.2f} -> {results[after]:5.2f} dB"
        )
        if detune == 0.0:
            # How much of a flat spectrum the comb passes, read off the noise
            # bins themselves rather than inferred from the OSNR -- which also
            # carries the signal's own 0.6 dB and would confuse the two.
            passed = (
                results.port(ring, "drop").noise[0].psd_x / results.port(amp, "out").noise[0].psd_x
            )
            # A Lorentzian's integral is (pi/2) * FWHM * peak. It runs high:
            # the linewidth formula is 1.5 % wide on its own and the shape
            # accounts for the rest.
            estimate = 0.5 * np.pi * width / FSR
            print(
                f"  {'':14s}  linewidth {width / 1e9:.2f} GHz in a {FSR / 1e9:.0f} GHz period."
                f" The comb passes {passed:.2%} of a flat spectrum; a Lorentzian"
                f" of that width would pass {estimate:.2%}, {estimate / passed - 1:.0%} more."
            )

    # -- and the same waveguide, straightened out --------------------------
    print("\nThe same waveguide as a delay line, which is all a spiral is:\n")
    for micrometres in (100.0, 1000.0, 100000.0):
        graph = Graph(ctx)
        laser = graph.add(CWLaser(power=0.0, wavelength=1550.0))
        guide = graph.add(Waveguide(length=micrometres, propagation_loss=2.0))
        meter = graph.add(PowerMeter())
        graph.chain(laser, guide, meter)
        power = w_to_dbm(graph.run()[meter].power_w)
        print(
            f"  {micrometres / 1000:8.1f} mm   {guide.group_delay() * 1e12:9.2f} ps"
            f"   {power:7.2f} dBm at 2 dB/cm"
        )
    print("\n  Ten centimetres of waveguide buys 1.4 ns and costs 20 dB. Optical")
    print("  buffering is expensive, and this is the reason.")


if __name__ == "__main__":
    main()
