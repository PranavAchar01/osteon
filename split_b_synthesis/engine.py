"""Split B engine - Synthesis & Iteration Controller (Day 1: real geometry + fixtures).

Input (from the orchestrator): {"plan": PlacementPlan, "report": StressReport|None,
"iteration": int}. Output: a pydantic-valid ImplantCandidate, with trace_id carried from
plan.trace_id. Mesh ops go through the blender-mcp tools (offline trimesh + pymeshlab).

THETA is the PLATE parametrization (the stem variant is deferred). THETA_BOUNDS below is the
single source of truth for the bounds guardrail and the (Day 2) CMA-ES search space.

Day 1 ships rung 1 (real geometry) + a real floor; Day 2 adds the LLM proposer, the CMA-ES
rung 2, the mock stress oracle, and the controller loop that consumes `report`.
"""

import os
from pathlib import Path

import numpy as np
import trimesh

from common.contracts import ImplantCandidate, PlacementPlan
from common.errors import RetryableError, ToolFailError
from common.ladder import with_fallback
from common.settings import settings
from split_b_synthesis.mcp_server import generate_mesh

ROOT = Path(__file__).resolve().parent.parent

# --- Task 1: theta schema + bounds -------------------------------------------------------
# Plate parametrization; all millimetres except n_screws (an integer count).
THETA_BOUNDS = {
    "length_mm": (40.0, 200.0),
    "width_mm": (8.0, 30.0),
    "thickness_mm": (2.0, 8.0),
    "n_screws": (2, 12),
    "screw_spacing_mm": (8.0, 40.0),
    "contour_offset_mm": (0.0, 5.0),
}
DEFAULT_THETA = {
    "length_mm": 96.0,
    "width_mm": 14.0,
    "thickness_mm": 4.0,
    "n_screws": 6,
    "screw_spacing_mm": 14.0,
    "contour_offset_mm": 0.5,
}


def clamp_theta(theta: dict) -> dict:
    """Clamp every parameter into THETA_BOUNDS; n_screws stays an int."""
    clamped = {}
    for key, (low, high) in THETA_BOUNDS.items():
        value = theta.get(key, DEFAULT_THETA[key])
        value = max(low, min(high, value))
        clamped[key] = int(round(value)) if key == "n_screws" else float(value)
    return clamped


def seed_theta(plan: PlacementPlan) -> dict:
    """Seed a plate from plan geometry: anchor span -> length, cortical -> thickness."""
    theta = dict(DEFAULT_THETA)
    anchors = plan.anchor_points
    if len(anchors) >= 2:
        points = np.array([[a.xyz.x, a.xyz.y, a.xyz.z] for a in anchors], dtype=np.float64)
        span = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
        if span > 0:
            theta["length_mm"] = span
        theta["n_screws"] = len(anchors)
        cortical = [a.cortical_thickness_mm for a in anchors if a.cortical_thickness_mm > 0]
        if cortical:
            theta["thickness_mm"] = min(cortical)
        theta["screw_spacing_mm"] = theta["length_mm"] / max(1, theta["n_screws"] - 1)
    return clamp_theta(theta)


def _candidate(plan, iteration, theta, mesh_path, validity, rung) -> ImplantCandidate:
    try:
        volume = float(abs(trimesh.load(mesh_path).volume)) if mesh_path else 0.0
    except Exception as exc:  # keep the rung's failure inside the shared taxonomy
        raise ToolFailError(f"could not measure generated mesh: {exc}")
    suffix = "-floor" if rung == "floor" else ""
    return ImplantCandidate(
        case_id=plan.case_id,
        candidate_id=f"cand-{iteration}{suffix}",
        iteration=iteration,
        parameter_vector=theta,
        mesh_path=mesh_path,
        contacts_anchor_ids=[a.id for a in plan.anchor_points],
        volume_mm3=volume,
        min_thickness_mm=float(theta["thickness_mm"]),
        validity=validity,
        fallback_rung=rung,
        trace_id=plan.trace_id,
    )


def _build(inp, rung):
    """Build a real candidate via the blender-mcp generate_mesh tool."""
    plan = inp["plan"]
    theta = seed_theta(plan)
    result = generate_mesh(theta)  # raises ToolFailError on failure -> the ladder advances
    return _candidate(plan, inp["iteration"], theta, result["mesh_path"], result["validity"], rung)


def _rung1(inp, trace):
    if os.getenv("OSTEON_FORCE_FAIL") == "synthesize":
        raise RetryableError("forced failure for demo")
    # TODO(B, Day 2): LLM theta-proposer (Bedrock via call_llm), seeded by the last StressReport.
    return _build(inp, rung=1)


def _floor(inp, trace):
    """Deterministic local floor: a guaranteed-watertight SOLID plate. Never raises.

    (Day 2 wires in last-known-good theta + a stop flag once the controller loop exists.)
    """
    plan = inp["plan"]
    iteration = inp["iteration"]
    theta = seed_theta(plan)
    try:
        box = trimesh.creation.box(
            extents=[theta["length_mm"], theta["width_mm"], theta["thickness_mm"]]
        )
        out_dir = Path(settings.OSTEON_TRACE_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        mesh_path = str(out_dir / f"cand_floor_{plan.trace_id}_{iteration}.stl")
        box.export(mesh_path)
        validity = {
            "watertight": bool(box.is_watertight),
            "manifold": bool(box.is_winding_consistent),
            "self_intersect": not bool(box.is_volume),
        }
        volume = float(abs(box.volume))
    except Exception:
        mesh_path = ""
        validity = {"watertight": False, "manifold": False, "self_intersect": True}
        volume = 0.0
    return ImplantCandidate(
        case_id=plan.case_id,
        candidate_id=f"cand-{iteration}-floor",
        iteration=iteration,
        parameter_vector=theta,
        mesh_path=mesh_path,
        contacts_anchor_ids=[a.id for a in plan.anchor_points],
        volume_mm3=volume,
        min_thickness_mm=float(theta["thickness_mm"]),
        validity=validity,
        fallback_rung="floor",
        trace_id=plan.trace_id,
    )


# Day 1: rung 1 + floor. Day 2: with_fallback([_rung1, _rung2], _floor) once CMA-ES lands.
run = with_fallback([_rung1], _floor)


if __name__ == "__main__":
    import glob
    import json
    import shutil

    from common.trace import LoopTrace

    out_dir = Path(__file__).resolve().parent / "fixtures"
    out_dir.mkdir(exist_ok=True)
    plan_files = sorted(
        glob.glob(str(ROOT / "split_a_localization" / "fixtures" / "placement_plan_*.json"))
    )
    for plan_file in plan_files:
        plan = PlacementPlan(**json.load(open(plan_file)))
        trace = LoopTrace(plan.case_id, trace_id=plan.trace_id, stage="synthesize-fixture-gen")
        candidate = run({"plan": plan, "report": None, "iteration": 0}, trace)

        stl_dst = out_dir / f"implant_candidate_{plan.case_id}.stl"
        if candidate.mesh_path and Path(candidate.mesh_path).exists():
            shutil.copyfile(candidate.mesh_path, stl_dst)

        data = json.loads(candidate.model_dump_json())
        # Point the committed fixture at the committed STL (repo-relative) for Split C.
        data["mesh_path"] = f"split_b_synthesis/fixtures/implant_candidate_{plan.case_id}.stl"
        (out_dir / f"implant_candidate_{plan.case_id}.json").write_text(json.dumps(data, indent=2))

        size_kb = stl_dst.stat().st_size / 1024 if stl_dst.exists() else 0.0
        print(
            f"fixture {plan.case_id}: rung={candidate.fallback_rung} "
            f"validity={candidate.validity} vol={candidate.volume_mm3:.1f}mm3 stl={size_kb:.0f}KB"
        )
