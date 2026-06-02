"""Split B engine - Synthesis & Iteration Controller.

PHASE 0 STUB: returns a valid ImplantCandidate. Person B replaces _rung1 with the bpy
parametric generator + LLM/CMA-ES controller (see SETUP.md). Input is a dict:
{"plan": PlacementPlan, "report": StressReport|None, "iteration": int}.
"""
import os

from common.contracts import ImplantCandidate
from common.errors import RetryableError
from common.ladder import with_fallback


def _build(inp, rung):
    plan = inp["plan"]
    it = inp["iteration"]
    return ImplantCandidate(
        case_id=plan.case_id,
        candidate_id=f"cand-{it}",
        iteration=it,
        parameter_vector={"length_mm": 96, "width_mm": 14, "thickness_mm": 4.0, "n_screws": 6, "contour_offset_mm": 0.5},
        mesh_path=f"fixtures/cand-{it}.stl",
        contacts_anchor_ids=[a.id for a in plan.anchor_points],
        volume_mm3=5120.0,
        min_thickness_mm=4.0,
        validity={"watertight": True, "manifold": True, "self_intersect": False},
        fallback_rung=rung,
        trace_id=plan.trace_id,
    )


def _rung1(inp, trace):
    if os.getenv("OSTEON_FORCE_FAIL") == "synthesize":
        raise RetryableError("forced failure for demo")
    # TODO(B): LLM theta-proposer (Bedrock) -> generate_mesh via blender-mcp
    return _build(inp, rung=1)


def _floor(inp, trace):
    # TODO(B): last-known-good theta (CMA-ES is rung 2)
    return _build(inp, rung="floor")


run = with_fallback([_rung1], _floor)
