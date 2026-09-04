"""The session server: the engine, over HTTP, for the interface to call.

The studio has always been a picture of an application. This is the part that
makes it one — the thing that lets pressing Run produce a number that did not
exist before.

**It is an ordinary client of the public API.** Every endpoint here is a thin
wrapper over something a script can already call: :func:`maiman.manifests`,
:func:`maiman.project.graph_from_dict`, :meth:`maiman.Graph.run`. Nothing in
this module knows any physics, and nothing in the engine knows this module
exists. If a feature is not reachable from Python it is not reachable from the
interface either, which is the constraint that keeps the two from drifting.

**No new dependencies.** The engine's only dependency is numpy, deliberately,
and a local design tool talking to one browser tab does not need a web
framework to do it. ``http.server`` is enough, is in the standard library, and
adds nothing to what someone installing this has to trust.

**Loopback only, by default.** ``/api/run`` accepts a graph description and
executes it. The registry already makes that far less dangerous than it sounds
— a project file may only *name* components that are already registered, never
import a dotted path, so opening one is not equivalent to running its author's
code. What remains is resource exhaustion: a graph is arbitrary work, and a
window big enough will occupy a core for a long time. So the socket binds to
127.0.0.1 unless a host is given explicitly, and a run is refused before it
starts if its window exceeds :data:`MAX_SAMPLES`.

Run it with ``python -m maiman.server`` and open the printed URL.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import traceback
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import manifests
from .encoding import EncodingError, encode, encode_results, scalars
from .graph import Graph, GraphError, Results
from .project import ProjectError, graph_from_dict, ui_from_dict
from .registry import UnknownComponentError
from .sweep import sweep

#: Largest simulation window a request may ask for, in samples per port.
#: ``sequence_length * samples_per_symbol``. Generous for interactive work — the
#: coherent example runs at 16384 — and small enough that a mistyped zero cannot
#: take the server away from the person using it.
MAX_SAMPLES = 4_194_304

#: Largest request body accepted, in bytes. A project file describing a link is
#: kilobytes; anything approaching this is not one.
MAX_BODY = 8 * 1024 * 1024

#: Where the interface is served from. The studio is a single file with no build
#: step, which is what lets the whole application be opened from a checkout.
STUDIO = Path(__file__).resolve().parent.parent.parent / "docs" / "ui-mockup.html"


class RequestError(Exception):
    """A request that is wrong in a way the client can fix.

    Carries the status to answer with, so the handler does not have to guess
    which failures are the caller's fault and which are the server's.
    """

    def __init__(self, status: HTTPStatus, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.detail = detail


def _build(document: dict[str, Any]) -> Graph:
    """The graph a project document describes, or a refusal saying whose fault it is.

    Shared by every endpoint that takes a project, so that a malformed document
    is answered the same way whether it arrived to be run once or swept.

    The window is checked *before* the graph is built, because the point is to
    refuse the work rather than to discover part-way through that it was too
    much.
    """
    if not isinstance(document, dict):
        raise RequestError(HTTPStatus.BAD_REQUEST, "the request body must be a JSON object")

    context = document.get("context")
    if isinstance(context, dict):
        window = int(context.get("sequence_length", 0)) * int(context.get("samples_per_symbol", 1))
        if window > MAX_SAMPLES:
            raise RequestError(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                f"a window of {window} samples exceeds the server limit of {MAX_SAMPLES}",
                "Reduce sequence_length or samples_per_symbol, or run it from Python, "
                "where nothing is holding a browser open while it works.",
            )

    try:
        return graph_from_dict(document)
    except UnknownComponentError as error:
        raise RequestError(HTTPStatus.BAD_REQUEST, str(error)) from error
    except (ProjectError, GraphError, ValueError, TypeError, KeyError) as error:
        raise RequestError(
            HTTPStatus.BAD_REQUEST, f"the project could not be built: {error}"
        ) from error


def _context(graph: Graph) -> dict[str, Any]:
    """What the run was actually done at, echoed back so the client need not assume."""
    return {
        "bit_rate": graph.ctx.bit_rate,
        "samples_per_symbol": graph.ctx.samples_per_symbol,
        "sequence_length": graph.ctx.sequence_length,
        "seed": graph.ctx.seed,
        "precision": graph.ctx.precision,
        "num_samples": graph.ctx.num_samples,
    }


def _execute(graph: Graph) -> Results:
    """Run a graph, mapping the two ways it can refuse onto the caller's fault."""
    try:
        return graph.run()
    except GraphError as error:
        # A wiring or validation problem: the graph as described cannot run, and
        # the person who described it is the one who can fix it.
        raise RequestError(HTTPStatus.UNPROCESSABLE_ENTITY, str(error)) from error
    except (ValueError, TypeError) as error:
        # A component rejecting its own parameters or its input. Also the
        # caller's to fix, and its message is written for them.
        raise RequestError(HTTPStatus.UNPROCESSABLE_ENTITY, str(error)) from error


def run_project(document: dict[str, Any]) -> dict[str, Any]:
    """Build the graph a project document describes, run it, and reduce the results.

    The whole of what the server does that is worth testing, separated from the
    HTTP that carries it so that it can be tested without a socket.
    """
    graph = _build(document)
    results = _execute(graph)
    try:
        encoded = encode_results(results)
    except EncodingError as error:
        raise RequestError(HTTPStatus.INTERNAL_SERVER_ERROR, str(error)) from error

    return {
        "results": encoded,
        "context": _context(graph),
        "ui": ui_from_dict(document),
    }


#: Most points one sweep may ask for. A curve is read, not counted: past a
#: couple of hundred the line stops gaining shape and starts costing minutes.
MAX_SWEEP_POINTS = 256

#: Most repeated runs per point. Repeats are how a noisy measurement becomes a
#: distribution rather than one sample, and the cost is multiplied by the point
#: count, so the product is bounded too.
MAX_SWEEP_RUNS = 64

#: And the product of the two, which is what actually decides how long a browser
#: sits waiting.
MAX_SWEEP_TOTAL = 1024


def run_sweep(request: dict[str, Any]) -> dict[str, Any]:
    """Run one parameter over a range and return the numbers at each point.

    ``{"project": {...}, "axis": {"node": "tx", "parameter": "power",
    "values": [...]}, "runs": 1}``

    One axis, because that is what a curve is and what the interface offers.
    :func:`maiman.sweep` takes any number of them and this is a thin wrapper
    over it, so a script that wants a surface still has one.

    Only the scalar results come back — see :func:`maiman.encoding.scalars`.
    """
    if not isinstance(request, dict):
        raise RequestError(HTTPStatus.BAD_REQUEST, "the request body must be a JSON object")

    document = request.get("project")
    if not isinstance(document, dict):
        raise RequestError(HTTPStatus.BAD_REQUEST, "no 'project' in the request")

    axis = request.get("axis")
    if not isinstance(axis, dict):
        raise RequestError(HTTPStatus.BAD_REQUEST, "no 'axis' in the request")
    node = axis.get("node")
    parameter = axis.get("parameter")
    values = axis.get("values")
    if not isinstance(node, str) or not isinstance(parameter, str):
        raise RequestError(HTTPStatus.BAD_REQUEST, "the axis needs a 'node' and a 'parameter'")
    if not isinstance(values, list) or not values:
        raise RequestError(HTTPStatus.BAD_REQUEST, "the axis needs a non-empty 'values' list")
    if not all(isinstance(v, int | float) and not isinstance(v, bool) for v in values):
        raise RequestError(HTTPStatus.BAD_REQUEST, "axis values must be numbers")
    if len(values) > MAX_SWEEP_POINTS:
        raise RequestError(
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            f"{len(values)} points exceeds the server limit of {MAX_SWEEP_POINTS}",
        )

    try:
        runs = int(request.get("runs", 1))
    except (TypeError, ValueError) as error:
        raise RequestError(HTTPStatus.BAD_REQUEST, "'runs' must be a whole number") from error
    if runs < 1 or runs > MAX_SWEEP_RUNS:
        raise RequestError(
            HTTPStatus.BAD_REQUEST, f"'runs' must be between 1 and {MAX_SWEEP_RUNS}, got {runs}"
        )
    if len(values) * runs > MAX_SWEEP_TOTAL:
        raise RequestError(
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            f"{len(values)} points x {runs} runs is {len(values) * runs} simulations, "
            f"over the server limit of {MAX_SWEEP_TOTAL}",
            "Run it from Python, where nothing is holding a browser open while it works.",
        )

    graph = _build(document)
    target = next((c for c in graph.components if c.label == node), None)
    if target is None:
        raise RequestError(
            HTTPStatus.BAD_REQUEST,
            f"no block labelled {node!r} in this project",
            f"blocks are: {', '.join(sorted(c.label for c in graph.components))}",
        )
    if parameter not in type(target).param_specs():
        raise RequestError(
            HTTPStatus.BAD_REQUEST,
            f"{node} has no parameter {parameter!r}",
            f"its parameters are: {', '.join(sorted(type(target).param_specs()))}",
        )

    try:
        result = sweep(graph, {(node, parameter): list(values)}, runs=runs)
    except GraphError as error:
        raise RequestError(HTTPStatus.UNPROCESSABLE_ENTITY, str(error)) from error
    except (ValueError, TypeError) as error:
        raise RequestError(HTTPStatus.UNPROCESSABLE_ENTITY, str(error)) from error

    key = f"{node}.{parameter}"
    points = [
        {
            "index": point.index,
            "value": point.values[key],
            "runs": [
                {
                    f"{label}.{port}": scalars(encode(signal))
                    for (label, port), signal in run.items()
                }
                for run in point.runs
            ],
        }
        for point in result.points
    ]
    return {
        "axis": {"node": node, "parameter": parameter, "values": list(values)},
        "runs": runs,
        "points": points,
        "context": _context(graph),
    }


class StudioHandler(BaseHTTPRequestHandler):
    """Routes. Deliberately few."""

    server_version = "maiman"
    sys_version = ""

    # -- plumbing ---------------------------------------------------------

    def _send(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _fail(self, error: RequestError) -> None:
        payload: dict[str, Any] = {"error": error.message}
        if error.detail:
            payload["detail"] = error.detail
        self._send(error.status, payload)

    def _body(self) -> Any:
        """The decoded request body, whatever JSON type it turned out to be.

        Deliberately not annotated ``dict``: a body of ``[1, 2, 3]`` is valid
        JSON and an invalid request, and claiming here that it is a dictionary
        would move that check somewhere it cannot be made. :func:`run_project`
        is where the shape is decided, and it answers with a 400.
        """
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise RequestError(HTTPStatus.BAD_REQUEST, "malformed Content-Length") from error
        if length <= 0:
            raise RequestError(HTTPStatus.BAD_REQUEST, "empty request body")
        if length > MAX_BODY:
            raise RequestError(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                f"request body of {length} bytes exceeds the limit of {MAX_BODY}",
            )
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RequestError(
                HTTPStatus.BAD_REQUEST, f"body is not valid JSON: {error}"
            ) from error

    def log_message(self, format: str, *args: Any) -> None:
        # One line per request, to stderr, without the default's double
        # timestamp. Quiet enough to leave running in a terminal beside the work.
        sys.stderr.write(f"  {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}\n")

    # -- routes -----------------------------------------------------------

    def do_GET(self) -> None:
        route = self.path.split("?", 1)[0].rstrip("/") or "/"
        try:
            if route == "/api/health":
                self._send(HTTPStatus.OK, {"status": "ok", "components": len(manifests())})
            elif route == "/api/manifests":
                self._send(HTTPStatus.OK, {"manifests": manifests()})
            elif route in ("/", "/index.html"):
                self._studio()
            else:
                raise RequestError(HTTPStatus.NOT_FOUND, f"no route {route!r}")
        except RequestError as error:
            self._fail(error)

    def do_POST(self) -> None:
        route = self.path.split("?", 1)[0].rstrip("/")
        try:
            if route == "/api/run":
                self._send(HTTPStatus.OK, run_project(self._body()))
            elif route == "/api/sweep":
                self._send(HTTPStatus.OK, run_sweep(self._body()))
            else:
                raise RequestError(HTTPStatus.NOT_FOUND, f"no route {route!r}")
        except RequestError as error:
            self._fail(error)
        except Exception as error:
            # A component failing in a way nobody anticipated is the server's
            # problem, not the caller's, and the traceback belongs in the
            # server's log rather than in the browser.
            traceback.print_exc()
            self._fail(
                RequestError(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    f"the run failed unexpectedly: {type(error).__name__}: {error}",
                    "The full traceback is in the server's terminal.",
                )
            )

    def _studio(self) -> None:
        if not STUDIO.is_file():
            raise RequestError(
                HTTPStatus.NOT_FOUND,
                "the studio page is not present",
                f"expected it at {STUDIO}. Serving the API alone still works; "
                "point a client at /api/manifests and /api/run.",
            )
        body = STUDIO.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def serve(
    host: str = "127.0.0.1", port: int = 8765, *, ready: Callable[[str], None] | None = None
) -> ThreadingHTTPServer:
    """Start the session server and return it, still running.

    Returns rather than blocks so that a test can start one, call it, and shut
    it down; :func:`main` is what blocks.
    """
    httpd = ThreadingHTTPServer((host, port), StudioHandler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, name="maiman-session", daemon=True)
    thread.start()
    if ready is not None:
        ready(f"http://{host}:{httpd.server_address[1]}/")
    return httpd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m maiman.server", description="Run the Maiman Studio session server."
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind. Defaults to loopback; see this module's docstring "
        "before changing it, because /api/run executes the graph it is given.",
    )
    args = parser.parse_args(argv)

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"  warning: binding {args.host}, not loopback. /api/run executes the graph\n"
            f"  it is given, and nothing here authenticates the caller.",
            file=sys.stderr,
        )

    httpd = serve(args.host, args.port)
    url = f"http://{args.host}:{httpd.server_address[1]}/"
    print(f"Maiman Studio session server\n  {len(manifests())} components\n  {url}\n")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
