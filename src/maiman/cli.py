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

from . import server


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

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Annotated rather than asserted: argparse hands back Any, and an assert
    # used for narrowing is an assert that disappears under -O.
    handler: Callable[[argparse.Namespace], int] = args.handler
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
