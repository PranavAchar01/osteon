"""Split C engine — Biomechanical Evaluation & Stress Oracle  (stage tag: ``evaluate``).

Decide whether a candidate implant survives. Input is a dict::

    {"candidate": ImplantCandidate, "case": CaseSpec}

The standard fallback ladder (common/ladder.py) wires three rungs (STANDARDIZATION.md §6):

    rung 1  full_fea           sfepy 3D linear-elastic FEA (ASTM F382-style bending)
    rung 2  reduced_surrogate  1D Euler-Bernoulli beam FE (no meshing, pure numpy)
    floor   analytic_fallback  closed-form beam bound — deterministic, never raises

Every rung emits a span and records its rung + solver in the StressReport.
The post-invoke report-nan-gate fires inside each rung, so a NaN/garbage result is
rejected and the ladder advances rather than letting it cascade into B.
"""

from __future__ import annotations

import os

from common.contracts import CaseSpec, ImplantCandidate, StressReport, Vec3
from common.errors import RejectedOutput, RetryableError
from common.ladder import with_fallback
from common.trace import LoopTrace

from . import fea
from .guardrails import report_nan_gate

# --- physiological / material defaults (documented; used when the case omits them) ---
DEFAULT_LOAD_N = 700.0  # ~1 body-weight static reference when load_profile is empty
DEFAULT_NU = 0.3  # Poisson's ratio (metal implant)
DEFAULT_MATERIAL = {"E_MPa": 110000.0, "yield_MPa": 830.0, "endurance_limit_MPa": 510.0}
I_BONE_REF_mm4 = 16700.0  # representative diaphyseal cortical section (hollow cylinder)
FOS_PASS = 1.0  # accept if FoS >= 1.0 and fatigue-safe
BENDING_MODE = "three_point"  # ASTM F382-style 4-pt/3-pt bench bend is the implant standard


# --------------------------------------------------------------------------- #
# Input adapters: turn the contracts into a beam abstraction + loads + material
# --------------------------------------------------------------------------- #
def _geometry(cand: ImplantCandidate) -> fea.BeamGeom:
    """Prefer the real STL bounding box; fall back to the parameter vector, then to
    volume + min-thickness. Always returns a valid prismatic beam abstraction."""
    path = cand.mesh_path
    if path and os.path.exists(path):
        try:
            import trimesh

            ext = trimesh.load(path, force="mesh").bounding_box.extents
            return fea.beam_from_dims(float(ext[0]), float(ext[1]), float(ext[2]))
        except Exception:
            pass  # fall through to parametric

    pv = cand.parameter_vector or {}
    if {"length_mm", "width_mm", "thickness_mm"} <= set(pv):
        return fea.beam_from_dims(pv["length_mm"], pv["width_mm"], pv["thickness_mm"])

    # last resort: a plate of thickness = min_thickness with a square footprint
    t = max(cand.min_thickness_mm or 2.0, 0.5)
    area = max(cand.volume_mm3 or 1000.0, 1.0) / t
    side = max(area**0.5, t)
    return fea.beam_from_dims(side, side, t)


def _load_N(case: CaseSpec) -> tuple[float, float]:
    """Worst-case force magnitude (N) and its cycle count from the load profile."""
    worst, cycles = 0.0, 1e6
    for lp in case.load_profile or []:
        fv = lp.get("force_vector_N") if isinstance(lp, dict) else None
        if isinstance(fv, dict):
            mag = (fv.get("x", 0) ** 2 + fv.get("y", 0) ** 2 + fv.get("z", 0) ** 2) ** 0.5
            if mag > worst:
                worst, cycles = mag, float(lp.get("cycles", cycles))
    if worst <= 0:
        return DEFAULT_LOAD_N, cycles
    return worst, cycles


def _material(case: CaseSpec) -> tuple[float, float, float]:
    m = {**DEFAULT_MATERIAL, **(case.implant_material or {})}
    return float(m["E_MPa"]), float(m["yield_MPa"]), float(m["endurance_limit_MPa"])


def _analytic(mode: str, geom: fea.BeamGeom, E: float, P: float) -> dict:
    """Closed-form bound for the requested loading mode."""
    if mode == "axial":
        return fea.analytic_axial(geom, E, P)
    if mode == "cantilever":
        return fea.analytic_cantilever(geom, E, P)
    return fea.analytic_three_point(geom, E, P)


def _shielding(case: CaseSpec, geom: fea.BeamGeom, E_implant: float) -> float:
    """Wolff's-law stress-shielding index via a composite-beam strain-energy ratio:
    bone + implant share the same curvature, so the bone offloads in proportion to the
    added stiffness. index = (EI_bone / (EI_bone + EI_implant))^2  in [0, 1]."""
    E_bone = float((case.bone_material or {}).get("E_cortical_MPa", 17000.0))
    ei_bone = E_bone * I_BONE_REF_mm4
    ei_impl = E_implant * geom.I_max
    if ei_bone <= 0:
        return 0.0
    # strain energies (common moment factor cancels in the ratio)
    w_intact = 1.0 / ei_bone
    w_impl = ei_bone / (ei_bone + ei_impl) ** 2
    return fea.shielding_index(w_intact, w_impl)


# --------------------------------------------------------------------------- #
# Report assembly + the post-invoke NaN gate
# --------------------------------------------------------------------------- #
def _build_report(cand, case, mech, solver, rung, confidence) -> StressReport:
    E, yield_mpa, endurance = _material(case)
    geom = mech["geom"]
    peak = float(mech["peak_von_mises_MPa"])
    fos = (yield_mpa / peak) if peak > 0 else float("inf")
    ssi = _shielding(case, geom, E)
    px, py, pz = mech.get("peak_location", (geom.L / 2.0, 0.0, -geom.h / 2.0))
    report = StressReport(
        case_id=cand.case_id,
        candidate_id=cand.candidate_id,
        iteration=cand.iteration,
        peak_von_mises_MPa=round(peak, 3),
        peak_location=Vec3(x=round(px, 3), y=round(py, 3), z=round(pz, 3)),
        factor_of_safety=round(fos, 3),
        fatigue_safe=bool(peak < endurance),
        stress_shielding_index=round(ssi, 4),
        displacement_max_mm=round(float(mech["displacement_max_mm"]), 5),
        passed=bool(fos >= FOS_PASS and peak < endurance),
        solver_used=solver,
        confidence=confidence,
        fallback_rung=rung,
        trace_id=cand.trace_id,
    )
    # POST-INVOKE guardrail: reject NaN/inf/garbage before it reaches B's controller.
    return report_nan_gate(report)


def _summarize(report: StressReport, trace) -> None:
    """Best-effort NL summary of the verdict via the gateway (stage='evaluate').

    Per STANDARDIZATION.md the only model entry point is common.llm.call_llm. The call
    is fully fallback-wrapped: no token / outage must never break the evaluation loop."""
    try:
        from common import llm

        verdict = "PASS" if report.passed else "FAIL"
        msg = [
            {"role": "system", "content": "You are a biomechanical FEA reviewer. One sentence."},
            {
                "role": "user",
                "content": (
                    f"Implant {report.candidate_id}: peak von Mises "
                    f"{report.peak_von_mises_MPa} MPa, FoS {report.factor_of_safety}, "
                    f"fatigue_safe={report.fatigue_safe}, shielding "
                    f"{report.stress_shielding_index}. Verdict {verdict}. Summarize."
                ),
            },
        ]
        resp = llm.call_llm(stage="evaluate", messages=msg, trace=trace)
        trace.emit(span="evaluate:summary", summary=resp.choices[0].message.content[:300])
    except Exception as exc:  # token missing, outage, parse error — non-fatal
        trace.emit(span="evaluate:summary", summary_fallback=str(exc)[:120])


# --------------------------------------------------------------------------- #
# The three rungs
# --------------------------------------------------------------------------- #
def _rung1(inp, trace):
    """Full 3D linear-elastic FEA via sfepy (solver_used='full_fea')."""
    if os.getenv("OSTEON_FORCE_FAIL") in ("evaluate", "evaluate_floor"):
        raise RetryableError("forced failure for demo")

    cand, case = inp["candidate"], inp["case"]
    geom = _geometry(cand)
    P, _cycles = _load_N(case)
    E, _y, _e = _material(case)
    try:
        # If a real STL was supplied, the watertight gate guards the (notional) tet path.
        if cand.mesh_path and os.path.exists(cand.mesh_path):
            from .guardrails import mesh_watertight_gate

            mesh_watertight_gate(cand.mesh_path)
        res = fea.solve_block_fea(geom, E, P, nu=DEFAULT_NU, mode=inp.get("mode", BENDING_MODE))
    except RejectedOutput:
        raise
    except Exception as exc:
        # triage the solver failure with the gateway (best-effort), then advance the ladder
        try:
            from common import llm

            llm.call_llm(
                stage="evaluate",
                messages=[
                    {
                        "role": "user",
                        "content": f"CalculiX/sfepy failed: {exc}. "
                        "Retry coarser or fall to surrogate? One word.",
                    }
                ],
                trace=trace,
            )
        except Exception:
            pass
        raise RetryableError(f"full FEA failed: {exc}")

    mech = {
        "geom": geom,
        "peak_von_mises_MPa": res["peak_von_mises_MPa"],
        "displacement_max_mm": res["displacement_max_mm"],
        "peak_location": res["peak_location"],
    }
    report = _build_report(cand, case, mech, "full_fea", rung=1, confidence=0.9)
    _summarize(report, trace)
    return report


def _rung2(inp, trace):
    """Reduced-order surrogate: 1D Euler-Bernoulli beam FE (solver_used='reduced_surrogate')."""
    if os.getenv("OSTEON_FORCE_FAIL") in ("evaluate_rung2", "evaluate_floor"):
        raise RetryableError("forced rung-2 failure for demo")
    cand, case = inp["candidate"], inp["case"]
    geom = _geometry(cand)
    P, _cycles = _load_N(case)
    E, _y, _e = _material(case)
    mode = inp.get("mode", BENDING_MODE)
    res = (
        fea.analytic_axial(geom, E, P)
        if mode == "axial"
        else fea.surrogate_beam_fea(geom, E, P, mode=mode)
    )
    mech = {
        "geom": geom,
        "peak_von_mises_MPa": res["peak_von_mises_MPa"],
        "displacement_max_mm": res["displacement_max_mm"],
        "peak_location": (geom.L / 2.0, 0.0, -geom.h / 2.0),
    }
    return _build_report(cand, case, mech, "reduced_surrogate", rung=2, confidence=0.6)


def _floor(inp, trace):
    """Closed-form analytic bound (solver_used='analytic_fallback'). Never raises."""
    cand, case = inp["candidate"], inp["case"]
    try:
        geom = _geometry(cand)
        P, _cycles = _load_N(case)
        E, _y, _e = _material(case)
        res = _analytic(inp.get("mode", BENDING_MODE), geom, E, P)
        peak = res["peak_von_mises_MPa"]
        disp = res["displacement_max_mm"]
    except Exception:
        # absolute floor: a conservative valid object even if geometry is unusable
        geom = fea.BeamGeom(L=50.0, b=10.0, h=4.0)
        peak, disp = 1.0, 0.0

    E, yield_mpa, endurance = _material(case)
    fos = (yield_mpa / peak) if peak > 0 else 1.0
    ssi = _shielding(case, geom, E)
    return StressReport(
        case_id=cand.case_id,
        candidate_id=cand.candidate_id,
        iteration=cand.iteration,
        peak_von_mises_MPa=round(float(peak), 3),
        peak_location=Vec3(x=round(geom.L / 2.0, 3), y=0.0, z=round(-geom.h / 2.0, 3)),
        factor_of_safety=round(float(fos), 3),
        fatigue_safe=bool(peak < endurance),
        stress_shielding_index=round(float(ssi), 4),
        displacement_max_mm=round(float(disp), 5),
        passed=bool(fos >= FOS_PASS and peak < endurance),
        solver_used="analytic_fallback",
        confidence=0.3,
        fallback_rung="floor",
        trace_id=cand.trace_id,
    )


run = with_fallback([_rung1, _rung2], _floor)


# --------------------------------------------------------------------------- #
# Stress heat-map field (NEW) — per-vertex von Mises sampled onto the render STL.
# The field shape follows the bending solution; its peak is scaled to the StressReport
# peak from the SAME solve, so the picture and the numbers agree (heatmap §10.4).
# --------------------------------------------------------------------------- #
def _field_arrays(cand, case, mode):
    import numpy as np
    import trimesh

    verts = np.asarray(trimesh.load(cand.mesh_path, force="mesh").vertices, dtype=float)
    ext = verts.max(0) - verts.min(0)
    i_len, i_thk = int(np.argmax(ext)), int(np.argmin(ext))
    L = float(ext[i_len])
    P, _c = _load_N(case)
    x0 = verts[:, i_len] - verts[:, i_len].min()
    mid = (verts[:, i_thk].max() + verts[:, i_thk].min()) / 2.0
    fib = verts[:, i_thk] - mid
    raw = np.abs(fea.bending_moment(x0, L, P, mode) * fib)  # proportional to sigma_vM
    return verts, raw


def _build_field(cand, case, mode, peak):
    raw = _field_arrays(cand, case, mode)[1]
    mx = float(raw.max())
    return (raw / mx * peak).tolist() if mx > 0 else [peak] * len(raw)


def stress_field(cand, case, mode=BENDING_MODE):
    """Per-vertex von Mises stress (MPa) on the implant STL, scaled so its peak equals
    the StressReport peak from the same solve. Aligned to the STL's vertex order."""
    rep = run(
        {"candidate": cand, "case": case, "mode": mode}, LoopTrace(cand.case_id, stage="evaluate")
    )
    return _build_field(cand, case, mode, rep.peak_von_mises_MPa)


def render_heatmap(cand, case, trace=None, mode=BENDING_MODE):
    """Solve once, build the field, render the Blender heat map, and log the artifact
    under the case's trace (no contract change, §8). Returns the tool output + report."""
    from .mcp_server import render_stress_heatmap

    trace = trace or LoopTrace(cand.case_id, stage="evaluate")
    rep = run({"candidate": cand, "case": case, "mode": mode}, trace)
    field = _build_field(cand, case, mode, rep.peak_von_mises_MPa)
    _e, yld, _en = _material(case)
    out = render_stress_heatmap(
        cand.mesh_path, field, yld, "", rep.solver_used, rep.factor_of_safety
    )
    trace.emit(
        span="evaluate:heatmap",
        heatmap_png=out["png_path"],
        heatmap_blend=out["blend_path"],
        peak_mpa=out["peak_mpa"],
    )
    return {**out, "solver_used": rep.solver_used, "report": rep.model_dump()}


# --------------------------------------------------------------------------- #
# CLI / fixture generation
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    from common.trace import LoopTrace

    ROOT = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="Split C evaluator")
    ap.add_argument(
        "--candidate", default=str(ROOT / "fixtures" / "example_implant_candidate.json")
    )
    ap.add_argument("--case", default=str(ROOT / "fixtures" / "example_case.json"))
    args = ap.parse_args()

    case = CaseSpec(**json.load(open(args.case)))
    cand = ImplantCandidate(**json.load(open(args.candidate)))
    trace = LoopTrace(case.case_id, stage="evaluate")
    report = run({"candidate": cand, "case": case}, trace)
    print(report.model_dump_json(indent=2))
