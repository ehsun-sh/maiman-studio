# Examples

Two folders, one for each way of using Maiman:

- [`python/`](python/) — scripts. Each builds a link, runs it and prints what it measured:
  `python examples/python/ook_link.py`. Some of them also write a project into `maiman/`. The data
  files a script reads (a PDK, a layout netlist) sit beside it, and
  [`lessons.py`](python/lessons.py) holds the notes and canvas marks the studio's projects carry.
- [`maiman/`](maiman/) — projects. `.maiman` files the studio opens (File → Open, or the templates
  in the File menu) and `maiman.load()` reads. Every one is written by a script in `python/`, so
  regenerate it rather than editing it by hand.

What comes next, phase by phase, is in [the examples roadmap](../docs/EXAMPLES_ROADMAP.md).
