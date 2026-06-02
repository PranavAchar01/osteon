"""Split C engine - Biomechanical Evaluation & Stress Oracle.

PHASE 0 STUB: returns a passing StressReport. Person C replaces _rung1 with CalculiX
FEA + shielding index (see SETUP.md). Input is a dict:
{"candidate": ImplantCandidate, "case": CaseSpec}.
"""
import os

from common.contracts import StressReport, Vec3
from common.errors import RetryableError
from common.ladder import with_fallback


def _build(inp, rung, solver, confidence):
    cand = inp["candidate"]
    case = inp["case"]
    yield_mpa = case.implant_material.get("yield_MPa", 880)
    endurance = case.implant_material.get("endurance_limit_MPa", 500)
    peak = 489.0
    return StressReport(
        case_id=cand.case_id,
        candidate_id=cand.candidate_id,
        iteration=cand.iteration,
        peak_von_mises_MPa=peak,
        peak_location=Vec3(x=0, y=0, z=0),
        factor_of_safety=round(yield_mpa / peak, 2),
        fatigue_safe=peak < endurance,
        stress_shielding_index=0.74,
        displacement_max_mm=0.42,
        passed=True,
        solver_used=solver,
        confidence=confidence,
        fallback_rung=rung,
        trace_id=cand.trace_id,
    )


def _rung1(inp, trace):
    if os.getenv("OSTEON_FORCE_FAIL") == "evaluate":
        raise RetryableError("forced failure for demo")
    # TODO(C): meshing_to_fe -> run_calculix via fea-mcp
    return _build(inp, rung=1, solver="full_fea", confidence=0.88)


def _floor(inp, trace):
    # TODO(C): analytic closed-form bound
    return _build(inp, rung="floor", solver="analytic_fallback", confidence=0.3)


run = with_fallback([_rung1], _floor)
