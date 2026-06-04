"""Backward-compatibility shim.

The original prototype driver has been replaced by the paper-grade E1 family.
This file forwards to :mod:`run_e1_main_comparison` so existing commands keep
working.
"""

from __future__ import annotations

from run_e1_main_comparison import main


if __name__ == "__main__":
    main()
