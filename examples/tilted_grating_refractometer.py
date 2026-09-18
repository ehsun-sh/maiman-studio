"""A tilted grating reads a liquid, and a vector mode solver says where its notches are.

A Bragg grating written square to the fibre axis reflects the core mode into
itself and is blind to everything outside the glass: that is what makes it a
strain gauge, and it is also what makes it useless as a chemical sensor. Tilt the
fringes by a few degrees and the core mode can reach backward *cladding* modes as
well -- light that runs along the glass-air boundary sixty microns out, and so
feels whatever the fibre is dipped in. What comes out is a comb of narrow notches
below the Bragg line.

Four things are worth reading off the output.

**The tilt is a divider, not a switch.** Square on, all of the coupling goes into
the mirror. At four degrees most of it has left, and by six the Bragg line is
nearly gone while the comb carries the light instead. The table below is that
trade, computed from the overlap of the fringe pattern with the core mode -- not
fitted, and not a rule of thumb.

**The comb moves and the Bragg line does not.** Dip the fibre in water and every
cladding notch shifts while the core's reflection stays exactly where it was.
Temperature moves both together, so the comb measured *against* the Bragg line is
a refractive-index reading with its own temperature reference built in. That is
the device.

**Sensitivity is not the same across the comb.** The modes that reach furthest
into the surroundings move most, and they are the ones lowest in effective index
-- deepest in the comb, furthest from the Bragg line.

**And where a notch sits depends on which solver you ask.** The scalar LP modes
every textbook starts from are an approximation that the glass-air boundary
breaks: the true vector modes sit below them, by enough to move a long-period
grating's notch a full nanometre. The last table measures it.

Takes about a minute: every notch below the Bragg line is a mode solve, and
there are dozens of them.

Run: ``python examples/tilted_grating_refractometer.py``
"""

from __future__ import annotations

import math

import numpy as np

from maiman.components import LongPeriodGrating
from maiman.components.tilted import TiltedFiberBraggGrating
from maiman.modes import StepIndexFibre, cladding_modes, core_modes
from maiman.photonics import tilted_grating_coupling, tilted_grating_spectrum
from maiman.units import C_LIGHT

PERIOD_NM = 535.0
TILT_DEG = 4.0
LENGTH_MM = 10.0
MODULATION = 5e-4
WATER = 1.333


def grating(tilt: float = TILT_DEG, surrounding: float = 1.0) -> TiltedFiberBraggGrating:
    return TiltedFiberBraggGrating(
        period=PERIOD_NM,
        tilt=tilt,
        length=LENGTH_MM,
        index_modulation=MODULATION,
        surrounding_index=surrounding,
        azimuthal_orders=3.0,
        label="tfbg",
    )


def divider() -> None:
    fibre = StepIndexFibre()
    guided = core_modes(fibre, 1.55e-6)[0]
    mode = cladding_modes(fibre, 1.55e-6, order=1, count=5)[4]
    print("1. What the tilt does with the coupling")
    print(f"     {'tilt':>6} {'into itself':>12} {'into LP15':>11}")
    for degrees in (0.0, 2.0, 4.0, 6.0, 10.0):
        settings = {
            "period": PERIOD_NM * 1e-9,
            "tilt": math.radians(degrees),
            "index_modulation": 1e-4,
        }
        mirror = tilted_grating_coupling(guided, guided, **settings)
        comb = tilted_grating_coupling(guided, mode, **settings)
        print(f"     {degrees:5.1f}d {mirror:11.1f}/m {comb:10.1f}/m")
    print(
        "     (per metre, for dn = 1e-4) The mirror weakens all the way down; a single\n"
        "     cladding mode peaks and falls again, because past a couple of degrees the\n"
        "     fringe turns over inside the core faster than that mode does and the light\n"
        "     goes to higher azimuthal orders instead.\n"
    )


def comb() -> None:
    dry = grating()
    bragg = dry.bragg_wavelength()
    band = (bragg - 12e-9, bragg + 0.5e-9)
    wet = grating(surrounding=WATER)
    in_air = dry.resonances(band)
    by_mode = {(order, rank): wavelength for order, rank, wavelength, _ in wet.resonances(band)}

    print(f"2. The comb, and what water does to it (Bragg line at {bragg * 1e9:.3f} nm)")
    print(f"     {'mode':>8} {'in air':>12} {'moved by':>10} {'coupling':>10}")
    for order, rank, wavelength, coupling in in_air[::-1][:4] + in_air[:4]:
        shifted = by_mode.get((order, rank))
        moved = "" if shifted is None else f"{(shifted - wavelength) * 1e12:8.1f}pm"
        name = "Bragg" if rank == 0 else f"LP{order},{rank}"
        print(
            f"     {name:>8} {wavelength * 1e9:10.4f}nm {moved:>10} {coupling * MODULATION:8.1f}/m"
        )
    print(
        "     The Bragg line does not move at all -- it never leaves the core. The notches\n"
        "     just below it hardly move either: those modes are still well inside the\n"
        "     glass. Twelve nanometres down, where the modes graze the boundary, the same\n"
        "     water moves them by seventy-five picometres. A tilted grating is read at the\n"
        "     bottom of its comb, against the Bragg line at the top of it.\n"
    )


def spectrum() -> None:
    dry = grating()
    bragg = dry.bragg_wavelength()
    wavelengths = np.linspace(bragg - 3e-9, bragg + 0.6e-9, 601)
    transmitted, reflected, cladding = tilted_grating_spectrum(
        C_LIGHT / wavelengths,
        fibre=dry.fibre(),
        period=dry.si("period"),
        tilt=dry.si("tilt"),
        length=dry.si("length"),
        index_modulation=MODULATION,
        max_order=3,
    )
    print("3. Where the light goes across the band")
    power = np.abs(transmitted) ** 2
    print(f"     deepest notch of the comb: {10 * math.log10(power.min()):.2f} dB")
    print(f"     most into the cladding:    {cladding.max() * 100:.1f} % of the input")
    print(f"     Bragg reflection:          {(np.abs(reflected) ** 2).max() * 100:.1f} %")
    total = power + np.abs(reflected) ** 2 + cladding
    print(f"     transmitted + reflected + stripped - 1: {np.max(np.abs(total - 1.0)):.1e}\n")


def solvers() -> None:
    print("4. Scalar LP modes against the true vector modes, for a long-period grating")
    band = (1.575e-6, 1.595e-6)
    scalar = LongPeriodGrating(period=500.0, length=25.0, cladding_modes=4.0, label="lpg")
    vector = LongPeriodGrating(
        period=500.0, length=25.0, cladding_modes=8.0, vector=True, label="lpg"
    )
    (scalar_notch,) = scalar.resonances(band)
    (vector_notch,) = vector.resonances(band)
    print(f"     LP04 phase matches at   {scalar_notch[1] * 1e9:9.3f} nm")
    print(f"     HE14, which it stands for, at {vector_notch[1] * 1e9:9.3f} nm")
    print(f"     difference: {(scalar_notch[1] - vector_notch[1]) * 1e9:.3f} nm")
    print(
        "     The LP approximation assumes every index step is small. The core's is. The\n"
        "     cladding-air step is a third of an index, and this is what assuming\n"
        "     otherwise costs -- on a device whose entire output is where its notch sits."
    )


def main() -> None:
    divider()
    comb()
    spectrum()
    solvers()


if __name__ == "__main__":
    main()
