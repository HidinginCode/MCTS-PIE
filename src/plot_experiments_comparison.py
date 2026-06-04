"""Backward-compatibility shim that forwards to :mod:`plot_e1_main`."""

from __future__ import annotations

from plot_e1_main import main


if __name__ == "__main__":
    main()
