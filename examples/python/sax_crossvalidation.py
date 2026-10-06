"""The circuit reduction against SAX: same models, same wiring, same grid.

:mod:`maiman.circuit` is a reduction written here rather than a solver taken
from elsewhere, and the case for that is that the two agree -- to 5e-15 on an
add-drop ring across two terahertz. That was first measured once, in a scratch
environment, and not kept; this is the measurement kept, so it can be run again
rather than remembered.

SAX is **not a dependency** and this is not in the test suite: it resolves to 37
packages including jax, and its sparse back-end ``klujax`` is LGPL-2.0-only. So
it runs in an environment of its own::

    python -m venv sax-env
    sax-env/bin/pip install -e . sax
    sax-env/bin/python examples/python/sax_crossvalidation.py

and in CI only when asked for, from the *SAX cross-validation* workflow. It
exits non-zero if any transmission disagrees by more than :data:`TOLERANCE`.

**What is compared.** Both solvers are handed the same device models --
:func:`maiman.photonics.directional_coupler` and
:func:`maiman.photonics.straight_waveguide`, evaluated once and passed to SAX
as its S-dictionaries -- wired into the same loop
:func:`maiman.photonics.ring_resonator` builds. So what differs between the two
answers is the reduction and nothing else.
"""

from __future__ import annotations

import sys

import numpy as np

from maiman.circuit import SMatrix
from maiman.photonics import directional_coupler, ring_resonator, straight_waveguide
from maiman.units import wavelength_to_frequency

REFERENCE = wavelength_to_frequency(1550e-9)
CIRCUMFERENCE = 100e-6
LOSS_DB_PER_M = 300.0
COUPLING = 0.1
DROP_COUPLING = 0.05
#: 2 THz holds three resonances of a 100 um silicon ring.
FREQUENCIES = REFERENCE + np.linspace(-1e12, 1e12, 4001)
#: A few thousand units in the last place, where the two agree to about twenty:
#: anything past this is a change in one reduction or the other, not rounding. A
#: coupling moved by one part in a billion is 7e-10 here.
TOLERANCE = 1e-12

#: The transmissions the README's table reports, output then input.
PATHS = (("through", "in"), ("drop", "in"), ("through", "add"), ("drop", "add"))


def sdict(matrix: SMatrix) -> dict[tuple[str, str], np.ndarray]:
    """An S-matrix as SAX's dictionary: ``(input, output)`` to the amplitude."""
    return {
        (into, out): matrix.s[:, i, j]
        for j, into in enumerate(matrix.ports)
        for i, out in enumerate(matrix.ports)
    }


def through_sax() -> dict[tuple[str, str], np.ndarray]:
    """The ring as SAX solves it, from maiman's own device models."""
    import jax  # type: ignore[import-not-found]
    import sax  # type: ignore[import-not-found]

    jax.config.update("jax_enable_x64", True)
    arc = straight_waveguide(
        FREQUENCIES,
        length=CIRCUMFERENCE / 2.0,
        reference_frequency=REFERENCE,
        loss_db_per_m=LOSS_DB_PER_M,
    )
    models = {
        "bus": lambda: sdict(directional_coupler(FREQUENCIES, coupling=COUPLING)),
        "drop_bus": lambda: sdict(directional_coupler(FREQUENCIES, coupling=DROP_COUPLING)),
        "arc": lambda: sdict(arc),
    }
    netlist = {
        "instances": {"bus": "bus", "drop_bus": "drop_bus", "upper": "arc", "lower": "arc"},
        "connections": {
            "bus,out2": "upper,in",
            "upper,out": "drop_bus,in2",
            "drop_bus,out2": "lower,in",
            "lower,out": "bus,in2",
        },
        "ports": {
            "in": "bus,in1",
            "through": "bus,out1",
            "add": "drop_bus,in1",
            "drop": "drop_bus,out1",
        },
    }
    circuit, _ = sax.circuit(netlist=netlist, models=models)
    solved = circuit()
    return {(out, into): np.asarray(solved[into, out]) for out, into in PATHS}


def main() -> int:
    ours = ring_resonator(
        FREQUENCIES,
        length=CIRCUMFERENCE,
        coupling=COUPLING,
        drop_coupling=DROP_COUPLING,
        reference_frequency=REFERENCE,
        loss_db_per_m=LOSS_DB_PER_M,
    )
    theirs = through_sax()
    print(f"add-drop ring, {FREQUENCIES.size} frequencies over 2 THz")
    worst = 0.0
    for out, into in PATHS:
        gap = float(np.max(np.abs(ours.transmission(out, into) - theirs[out, into])))
        worst = max(worst, gap)
        print(f"  {into:>4} -> {out:<8} max |maiman - sax| = {gap:.1e}")
    if worst > TOLERANCE:
        print(f"disagreement {worst:.1e} exceeds {TOLERANCE:.0e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
