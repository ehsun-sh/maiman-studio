# Security

## Reporting a vulnerability

Use GitHub's private vulnerability reporting — the **Security** tab, then *Report a
vulnerability*. That keeps the report out of the public issue tracker until there is something to
say about it.

If that is not available, email **ehsun.ca@gmail.com**. Please do not open a public issue for a
vulnerability first.

Expect an acknowledgement within a week. This is a 0.x project maintained by one person in their
own time, and pretending to a response time it cannot meet would be worse than saying so.

## Supported versions

| Version | Supported |
| :--- | :--- |
| `0.1.x` | Yes |
| `main` | Yes |
| Anything older | There is nothing older |

Fixes land on `main` and go out in the next `0.1.x`. While the version stays below `1.0`, a
security fix may arrive alongside behavioural changes rather than on its own — pinning an exact
version and never updating is the wrong shape of caution here, because there is no branch to
backport to and there will not be one until the interface stops moving.

## What the threat model actually is

Worth being concrete, because "a physics simulator" sounds like it has no attack surface and that
is not quite true.

### Project and PDK files are data, not code

A `.maiman` project and a `.pdk` kit are JSON. Loading one may only **name** components that are
already in the registry; it can never import a dotted path. So opening a file someone sent you is
not equivalent to running their code, and `"type": "os.system"` is refused by name like any other
unregistered component — there are tests for exactly that in
[`tests/test_project.py`](tests/test_project.py) and [`tests/test_pdk.py`](tests/test_pdk.py).

What a hostile file *can* do is describe a very expensive simulation. See below.

### The session server binds to loopback

`maiman serve` starts an HTTP server whose `/api/run` accepts a graph and executes it. That is the
point of it, and it is why the socket binds to **127.0.0.1** unless a host is given explicitly.
Passing `--host` something else prints a warning, deliberately.

Do not expose it to a network you do not control. There is no authentication, and there is not
meant to be: it is a local tool whose front end happens to be a browser page.

### Resource exhaustion is the real risk

A graph is arbitrary work. A large enough window will occupy a core for a long time and allocate
accordingly, which is a denial of service against the machine running it whether the request came
from a person or a file. The server refuses a run whose window exceeds `MAX_SAMPLES`
(4,194,304 samples) before it starts, rather than discovering the problem while allocating.

That limit is a guard rail, not a sandbox. Long sweeps and deep spans can still be slow by design.

### The studio page

`src/maiman/studio/index.html` is a single file with no build step and no runtime dependency on a
CDN. It talks to the local server and to nothing else.

## What is not a vulnerability

* **A wrong physical result.** That is a bug, and an important one — open a public issue with the
  reference it should match. See [`CONTRIBUTING.md`](CONTRIBUTING.md).
* **A slow simulation.** Unless it is disproportionate to the window, in which case it is a
  performance bug.
* **The server executing a graph you sent it.** That is its documented purpose on loopback.

## Dependencies

The runtime dependency is NumPy, and that is deliberate: the fewer things in the tree, the fewer
things to audit. FFTW is deliberately absent — it is GPL-2.0-or-later, and linking it would change
this project's licence rather than its security, but the effect on the dependency surface is the
same. Development additionally uses pytest, mypy, ruff, build and hatchling.
