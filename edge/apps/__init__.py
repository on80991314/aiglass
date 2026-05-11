"""Edge-side application entry points.

Each module in this package exposes:
   add_args(parser)  — append CLI flags for this app
   run(args)         — start the app

The launcher in ../main.py wires them together as subcommands.
"""

from __future__ import annotations
