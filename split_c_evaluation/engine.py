"""Split C engine - Biomechanical Evaluation & Stress Oracle.

Input is a dict: {"candidate": ImplantCandidate, "case": CaseSpec}.

The rungs use an analytic beam/plate stress model whose output is genuinely driven by
the inputs: applied load (load_profile), implant geometry (section from the candidate),
and material properties (implant + bone). Person C swaps _rung1 for CalculiX tet FEA via
fea-mcp (see SETUP.md); the analytic model stays as the floor. solver_used reflects which
path produced the report.
"""
import math
import os

from common.contracts import StressReport, Vec3
from common.errors import RetryableError
from common.ladder import with_fallback

# Nominal diaphyseal cortical cross-section (mm^2) for the load-sharing / shielding
# estimate. A real FEA solve replaces this with the actual bone block.
CORTICAL_AREA_MM2 = 350.0
FOS_CAP = 999.0


def _resultant_force_N(case):
    """Resultant applied force magnitude (N) and worst-case cycle count from load_profile."""
    fx = fy = fz = 0.0
    cycles = 0.0
    for load in case.load_profile or []:
        v = load.get("force_vector_N", {})
        fx += float(v.get("x", 0.0))
        fy += float(v.get("y", 0.0))
        fz += float(v.get("z", 0.0))
        cycles = max(cycles, float(load.get("cycles", 0.0)))
    return math.sqrt(fx * fx + fy * fy + fz * fz), cycles


def _section(cand):
    """Rectangular beam section (mm) from the candidate geometry.

    Prefers parameter_vector {length_mm, width_mm, thickness_mm}; otherwise derives a
    section from volume_mm3 and min_thickness_mm. Returns (b, h, span, A, I, c)."""
    pv = cand.parameter_vector or {}
    h = float(pv.get("thickness_mm", cand.min_thickness_mm or 1.0)) or 1.0
    width = pv.get("width_mm")
    length = pv.get("length_mm")
    if width is None or length is None:
        side = max((cand.volume_mm3 or 1.0) ** (1 / 3), 1.0)
        width = width or side
        length = length or side
    b = float(width)
    span = float(length)
    A = max(b * h, 1e-6)
    I = max(b * h ** 3 / 12.0, 1e-9)
    c = h / 2.0
    return b, h, span, A, I, c


def _evaluate(inp, *, rung, solver, confidence, load_factor):
    """Analytic stress report. `load_factor` scales the bending moment arm: rung-1
    surrogate models a stiffer, partially bone-supported construct; the floor is more
    conservative (longer effective lever -> higher predicted stress)."""
    cand = inp["candidate"]
    case = inp["case"]

    F, cycles = _resultant_force_N(case)
    b, h, span, A, I, c = _section(cand)

    E_impl = float(case.implant_material.get("E_MPa", 110000.0))
    yield_mpa = float(case.implant_material.get("yield_MPa", 830.0))
    endurance = float(case.implant_material.get("endurance_limit_MPa", 500.0))
    E_bone = float(case.bone_material.get("E_cortical_MPa", 17000.0))

    # Load sharing: a stiffer implant relative to bone carries a larger share of F.
    EA_impl = E_impl * A
    EA_bone = E_bone * CORTICAL_AREA_MM2
    implant_load_fraction = EA_impl / (EA_impl + EA_bone)
    F_impl = F * implant_load_fraction

    # Stresses: axial + bending (simply-supported beam, central load).
    sigma_axial = F_impl / A
    moment = F_impl * span * load_factor / 4.0
    sigma_bend = moment * c / I
    peak = sigma_axial + sigma_bend

    # Mid-span deflection of a simply-supported beam under central load.
    displacement = (F_impl * span ** 3) / (48.0 * E_impl * I) if F_impl > 0 else 0.0

    fos = FOS_CAP if peak < 1e-6 else min(yield_mpa / peak, FOS_CAP)
    fatigue_safe = peak < endurance or cycles < 1.0

    # Stress shielding: 0 = full shielding (implant takes everything), 1 = natural bone.
    shielding_index = EA_bone / (EA_bone + EA_impl)

    passed = bool(fos >= 1.5 and fatigue_safe and math.isfinite(peak))

    return StressReport(
        case_id=cand.case_id,
        candidate_id=cand.candidate_id,
        iteration=cand.iteration,
        peak_von_mises_MPa=round(peak, 2),
        peak_location=Vec3(x=0.0, y=0.0, z=round(span / 2.0, 2)),
        factor_of_safety=round(fos, 2),
        fatigue_safe=fatigue_safe,
        stress_shielding_index=round(shielding_index, 3),
        displacement_max_mm=round(displacement, 3),
        passed=passed,
        solver_used=solver,
        confidence=confidence,
        fallback_rung=rung,
        trace_id=cand.trace_id,
    )


def _rung1(inp, trace):
    if os.getenv("OSTEON_FORCE_FAIL") == "evaluate":
        raise RetryableError("forced failure for demo")
    # TODO(C): meshing_to_fe -> run_calculix via fea-mcp (solver_used="full_fea").
    # Until then, an analytic surrogate that still responds to the inputs.
    return _evaluate(inp, rung=1, solver="reduced_surrogate", confidence=0.7, load_factor=1.0)


def _floor(inp, trace):
    # Conservative closed-form bound: never raises.
    return _evaluate(inp, rung="floor", solver="analytic_fallback", confidence=0.3, load_factor=1.5)


run = with_fallback([_rung1], _floor)
