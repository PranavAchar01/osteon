# Presence of this file puts the repo root on sys.path for pytest, so tests can
# import orchestrator and the split_* packages.
#
# It also guards against the multiple-OpenMP-runtime crash: torch (bundled libomp) and
# sfepy/scipy/MKL each ship an OpenMP runtime, and under pytest's import order (sfepy-heavy
# modules collected first, then torch imported by Split A's smoke test) the two collide and
# segfault at the first torch.tensor call. Setting these before any heavy import makes
# `pytest` green out of the box. Threading runtime only — never affects a numerical result.
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
