"""Split C guardrails (STANDARDIZATION.md §9, declared in gateway/guardrails.yaml).

    pre-invoke   implant/mesh-watertight-gate : block the solver on an invalid mesh
                 so we never burn ~90s tetrahedralising / solving garbage.
    post-invoke  implant/report-nan-gate      : reject StressReports with NaN / inf /
                 negative FoS before B consumes them — kills cascading errors.

Guardrail rejections raise RejectedOutput (code E_BAD_OUTPUT), which the fallback
ladder treats as a reason to advance to the next rung.
"""

from __future__ import annotations

import math
import os

from common.errors import RejectedOutput


def mesh_watertight_gate(mesh_path: str) -> bool:
    """PRE-INVOKE. Raise RejectedOutput unless `mesh_path` is a watertight surface mesh.

    Used by fea-mcp.meshing_to_fe before tetrahedralisation."""
    if not mesh_path or not os.path.exists(mesh_path):
        raise RejectedOutput(f"mesh-watertight-gate: mesh not found: {mesh_path!r}")
    try:
        import trimesh

        mesh = trimesh.load(mesh_path, force="mesh")
    except Exception as exc:  # unreadable / not a mesh
        raise RejectedOutput(f"mesh-watertight-gate: unreadable mesh: {exc}")
    if getattr(mesh, "is_empty", True) or len(getattr(mesh, "faces", [])) == 0:
        raise RejectedOutput("mesh-watertight-gate: empty mesh")
    if not mesh.is_watertight:
        raise RejectedOutput("mesh-watertight-gate: mesh is not watertight")
    return True


def _bad_number(x) -> bool:
    try:
        return math.isnan(x) or math.isinf(x)
    except TypeError:
        return True


def report_nan_gate(report) -> object:
    """POST-INVOKE. Raise RejectedOutput if a StressReport carries NaN/inf or a
    physically impossible value (non-positive FoS / stress, out-of-range index).

    Accepts a StressReport pydantic model or a plain dict; returns it unchanged if clean."""
    g = report.get if isinstance(report, dict) else lambda k, d=None: getattr(report, k, d)

    checks = {
        "peak_von_mises_MPa": g("peak_von_mises_MPa"),
        "factor_of_safety": g("factor_of_safety"),
        "displacement_max_mm": g("displacement_max_mm"),
        "stress_shielding_index": g("stress_shielding_index"),
    }
    for name, val in checks.items():
        if val is None or _bad_number(val):
            raise RejectedOutput(f"report-nan-gate: {name} is not finite ({val!r})")

    if checks["factor_of_safety"] <= 0:
        raise RejectedOutput("report-nan-gate: non-positive factor_of_safety")
    if checks["peak_von_mises_MPa"] < 0:
        raise RejectedOutput("report-nan-gate: negative peak_von_mises")
    if not (0.0 <= checks["stress_shielding_index"] <= 1.0):
        raise RejectedOutput("report-nan-gate: stress_shielding_index out of [0,1]")
    return report
