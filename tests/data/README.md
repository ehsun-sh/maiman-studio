# Test data

## `gdsfactory_straight_with_bend.yml`

Copied **verbatim** from gdsfactory 9.45.0,
`gdsfactory/samples/netlists/straight_with_bend.yml`.

gdsfactory is MIT licensed, Copyright (c) 2020 PsiQuantum Corp —
<https://github.com/gdsfactory/gdsfactory>.

It is here so that [`tests/test_netlist.py`](../test_netlist.py) parses a file a
layout tool actually emitted, rather than one written from a memory of the
format. The two differ in exactly the places that matter: the instance names are
the router's generated ones, `info` carries the derived arc length of the bend
(16.637 um for a 90-degree Euler bend of radius 10, which is not recoverable from
the radius without the Euler parameter), and the keys are ordered the way the
writer ordered them.

Nothing in this package imports gdsfactory. See
[`src/maiman/netlist.py`](../../src/maiman/netlist.py) for why: it requires
kfactory, which requires klayout, which is GPL-3.0-or-later.
