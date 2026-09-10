"""The ``maiman`` command: the front door for someone who installed the wheel.

``python -m maiman.server`` has always worked from a checkout, and still does.
It is not what a person who ran ``pip install maiman`` will guess, though, and
the interface is the part of this project that most needs to be guessable — the
whole point of shipping the studio page inside the distribution is that opening
it should not require knowing where the source tree is.

There is one subcommand today. It is written as a subcommand anyway, rather than
as a bare ``maiman`` that starts a server, because a command whose only
behaviour is a side effect nobody named is a command that cannot grow one more.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

from . import backend, server


def build_parser() -> argparse.ArgumentParser:
    """The whole command line, in one place, so ``--help`` is the specification."""
    parser = argparse.ArgumentParser(
        prog="maiman",
        description="Maiman Studio: an open-source optical communication simulator.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True, metavar="command")

    serve = subcommands.add_parser(
        "serve",
        help="run the session server and serve the studio",
        description=(
            "Start the session server and print the URL to open. Binds loopback by "
            "default: /api/run executes the graph it is given."
        ),
    )
    # Declared by the server, not restated here. Two front doors, one set of flags.
    server.add_serve_arguments(serve)
    serve.set_defaults(handler=server.serve_from_args)

    devices = subcommands.add_parser(
        "devices",
        help="report the array back-ends available, and check any device against NumPy",
        description=(
            "List the array libraries this interpreter can use and, for each one "
            "that is not NumPy, propagate a soliton on it and compare the field "
            "against the same span on NumPy. Exits non-zero only if a back-end is "
            "present and disagrees; having no device at all is an answer, not a "
            "failure."
        ),
    )
    devices.set_defaults(handler=_report_devices)

    return parser


def _report_devices(args: argparse.Namespace) -> int:
    """``maiman devices``: what is installed, and whether it agrees with NumPy."""
    report = backend.check_device()

    print("Array back-ends")
    for name, present in report.backends.items():
        print(f"  {name:12} {'available' if present else 'not installed'}")

    if not report.checks:
        print("\nOnly NumPy here, so there is nothing to cross-check.")
        print("A GPU needs `pip install cupy-cuda12x` and a CUDA device.")
        return 0

    print("\nOne soliton period, propagated on each and compared against NumPy")
    for check in report.checks:
        if not check.ran:
            print(f"  {check.name:12} could not run — {check.why}")
        else:
            verdict = "agrees" if check.agrees else "DISAGREES"
            print(f"  {check.name:12} {verdict}, largest difference {check.max_difference:.3e}")
    return 0 if report.agrees else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Annotated rather than asserted: argparse hands back Any, and an assert
    # used for narrowing is an assert that disappears under -O.
    handler: Callable[[argparse.Namespace], int] = args.handler
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
