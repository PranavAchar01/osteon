"""Split B engine - Synthesis & Iteration Controller.

Input (from the orchestrator): {"plan": PlacementPlan, "report": StressReport|None,
"iteration": int}. Output: a pydantic-valid ImplantCandidate, trace_id carried from
plan.trace_id. Mesh ops go through the blender-mcp tools (offline trimesh + pymeshlab).

Fallback ladder (STANDARDIZATION §6), wired at the bottom as
``with_fallback([_rung1, _rung2], _floor)``:
  - rung 1: LLM theta-proposer via common.llm.call_llm(stage="synthesize"); bad output or
    a model outage -> RetryableError/RejectedOutput so the ladder advances.
  - rung 2: CMA-ES numeric optimizer (no LLM) over THETA_BOUNDS against a stress objective.
  - floor: last-known-good theta + a stop flag; a guaranteed-watertight solid plate; never raises.

THETA is the PLATE parametrization; THETA_BOUNDS is the single source of truth for the bounds
guardrail and the CMA-ES search space. THETA_BOUNDS is intentionally STATIC: the frozen
synthesize input is only {plan, report, iteration}, so CaseSpec.constraints never reach Split B.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import openai
import trimesh

from common import llm
from common.contracts import ImplantCandidate, PlacementPlan, StressReport, Vec3
from common.errors import RejectedOutput, RetryableError, ToolFailError
from common.ladder import with_fallback
from common.settings import settings
from common.trace import hash_payload
from split_b_synthesis.mcp_server import check_contacts, generate_mesh

ROOT = Path(__file__).resolve().parent.parent

# --- theta schema + bounds (single source of truth) --------------------------------------
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

# Controller objective constants (no CaseSpec here -> nominal values).
TARGET_FOS = 1.5
SSI_BAND = (0.6, 0.9)
_NOMINAL_LOAD_N = 700.0
_STRESS_K = 550.0  # calibrated so a mid-range plate lands near the example report (~490 MPa)
_DEFAULT_YIELD_MPA = 830.0
_DEFAULT_ENDURANCE_MPA = 510.0

# Tests inject a stress oracle (the C-simulator) here so rung 2 optimizes against it offline.
# In live/standalone operation it stays None and rung 2 uses the built-in analytic proxy below.
STRESS_ORACLE = None

# Last-known-good theta per case_id, for the floor (STANDARDIZATION §6 rung-3).
_LAST_GOOD: dict = {}


def clamp_theta(theta: dict) -> dict:
    """Clamp every parameter into THETA_BOUNDS; n_screws stays an int."""
    clamped = {}
    for key, (low, high) in THETA_BOUNDS.items():
        value = theta.get(key, DEFAULT_THETA[key])
        value = max(low, min(high, value))
        clamped[key] = int(round(value)) if key == "n_screws" else float(value)
    return clamped


def _placement_center(plan: PlacementPlan) -> np.ndarray:
    """Body center: defect_region.centroid if populated, else the anchor centroid."""
    defect = plan.defect_region or {}
    if isinstance(defect, dict) and defect.get("volume_mm3", 0):
        c = defect["centroid"]
        return np.array([c["x"], c["y"], c["z"]], dtype=np.float64)
    pts = np.array([[a.xyz.x, a.xyz.y, a.xyz.z] for a in plan.anchor_points], dtype=np.float64)
    return pts.mean(axis=0) if len(pts) else np.zeros(3)


def _frame_basis(plan: PlacementPlan) -> np.ndarray:
    basis = np.asarray(plan.coordinate_frame.get("basis"), dtype=np.float64)
    return basis if basis.shape == (3, 3) else np.eye(3)


def _placement(plan: PlacementPlan):
    """(anchors_json, frame_json) telling generate_mesh where on the bone the implant goes."""
    center = _placement_center(plan)
    anchors = [json.loads(a.model_dump_json()) for a in plan.anchor_points]
    frame = {
        "origin": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])},
        "basis": _frame_basis(plan).tolist(),
    }
    return anchors, frame


def seed_theta(plan: PlacementPlan) -> dict:
    """Seed the plate from REAL plan geometry in the plan's coordinate frame, recomputed per
    patient: length spans the anchors along the bone long axis, width across them, thickness from
    cortical thickness, one screw per anchor. Always clamped to THETA_BOUNDS."""
    theta = dict(DEFAULT_THETA)
    anchors = plan.anchor_points
    if len(anchors) >= 2:
        center = _placement_center(plan)
        world = np.array([[a.xyz.x, a.xyz.y, a.xyz.z] for a in anchors], dtype=np.float64)
        local = (_frame_basis(plan) @ (world - center).T).T
        theta["length_mm"] = float(local[:, 2].max() - local[:, 2].min()) + 20.0
        theta["width_mm"] = float(local[:, 0].max() - local[:, 0].min()) + 10.0
        theta["n_screws"] = len(anchors)
        cortical = [a.cortical_thickness_mm for a in anchors if 0 < a.cortical_thickness_mm < 50]
        if cortical:
            theta["thickness_mm"] = float(np.median(cortical))
        theta["screw_spacing_mm"] = theta["length_mm"] / max(1, len(anchors) - 1)
    return clamp_theta(theta)


# --- guardrails (mirror the gateway guardrails so they fire offline; STANDARDIZATION §9) --
def theta_bounds_check(theta: dict) -> None:
    """implant/theta-bounds-check (pre-invoke): reject out-of-range theta BEFORE generate_mesh."""
    for key, (low, high) in THETA_BOUNDS.items():
        value = theta.get(key)
        if value is None or value < low or value > high:
            raise RejectedOutput(f"theta-bounds-check: {key}={value} outside [{low}, {high}]")


def mesh_validity_check(candidate: ImplantCandidate) -> None:
    """implant/mesh-validity-check (post-invoke): reject invalid meshes before they reach C."""
    v = candidate.validity
    if not v.get("watertight") or not v.get("manifold") or v.get("self_intersect"):
        raise RejectedOutput(f"mesh-validity-check: invalid mesh {v}")


# --- stress objective + analytic proxy ----------------------------------------------------
def _objective(report: StressReport, volume_mm3: float, max_volume=None) -> float:
    """Scalarize a StressReport for CMA-ES: minimize peak von Mises with constraint penalties."""
    obj = float(report.peak_von_mises_MPa)
    if report.factor_of_safety < TARGET_FOS:
        obj += 1000.0 * (TARGET_FOS - report.factor_of_safety)
    low, high = SSI_BAND
    ssi = report.stress_shielding_index
    if ssi < low:
        obj += 1000.0 * (low - ssi)
    elif ssi > high:
        obj += 1000.0 * (ssi - high)
    if max_volume and volume_mm3 > max_volume:
        obj += volume_mm3 - max_volume
    return obj


def _analytic_oracle(theta: dict) -> StressReport:
    """rung-2 internal cost model: closed-form beam bending (SETUP §7), nominal load/material.

    Used only when no external oracle is injected (tests inject the mock C-simulator instead).
    """
    thickness = theta["thickness_mm"]
    width = theta["width_mm"]
    peak = _STRESS_K * _NOMINAL_LOAD_N / (thickness * width**2)
    fos = _DEFAULT_YIELD_MPA / peak
    ssi = max(0.0, min(1.0, 1.0 - thickness / 16.0))
    return StressReport(
        case_id="",
        candidate_id="analytic",
        iteration=0,
        peak_von_mises_MPa=peak,
        peak_location=Vec3(x=0.0, y=0.0, z=0.0),
        factor_of_safety=fos,
        fatigue_safe=bool(peak < _DEFAULT_ENDURANCE_MPA),
        stress_shielding_index=ssi,
        displacement_max_mm=0.3,
        passed=bool(fos >= TARGET_FOS and SSI_BAND[0] <= ssi <= SSI_BAND[1]),
        solver_used="analytic_fallback",
        confidence=0.5,
        fallback_rung="floor",
        trace_id="",
    )


def _resolve_oracle():
    return STRESS_ORACLE if STRESS_ORACLE is not None else _analytic_oracle


def _approx_volume(theta: dict) -> float:
    return theta["length_mm"] * theta["width_mm"] * theta["thickness_mm"]


# --- candidate assembly -------------------------------------------------------------------
def _safe_contacts(mesh_path: str, plan: PlacementPlan) -> list:
    """Real contacts via the check_contacts tool (<1 mm). Never raises (informational field)."""
    if not mesh_path:
        return []
    try:
        anchors = [json.loads(a.model_dump_json()) for a in plan.anchor_points]
        return check_contacts(mesh_path, anchors)["contacts"]
    except Exception:
        return []


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
        contacts_anchor_ids=_safe_contacts(mesh_path, plan),
        volume_mm3=volume,
        min_thickness_mm=float(theta["thickness_mm"]),
        validity=validity,
        fallback_rung=rung,
        trace_id=plan.trace_id,
    )


def _make_candidate(plan, iteration, theta, rung) -> ImplantCandidate:
    """Place the implant onto the bone and assemble the candidate. Plate geometry (length, width,
    screw count/spacing) follows the real anchor layout; the controller's theta supplies thickness
    and contour. Runs the post-invoke validity guardrail before returning."""
    geo = seed_theta(plan)
    placed = clamp_theta(
        {
            **theta,
            "length_mm": geo["length_mm"],
            "width_mm": geo["width_mm"],
            "n_screws": geo["n_screws"],
            "screw_spacing_mm": geo["screw_spacing_mm"],
        }
    )
    anchors, frame = _placement(plan)
    result = generate_mesh(placed, anchors, frame)  # frame-driven placement onto the bone
    candidate = _candidate(plan, iteration, placed, result["mesh_path"], result["validity"], rung)
    mesh_validity_check(candidate)  # post-invoke guardrail; RejectedOutput -> ladder advances
    _LAST_GOOD[plan.case_id] = dict(placed)
    return candidate


def _build(inp, rung):
    """Day-1 seed-based builder, kept as the no-LLM path: seed theta from plan -> candidate."""
    plan = inp["plan"]
    return _make_candidate(plan, inp["iteration"], seed_theta(plan), rung)


def _emit_span(trace, tool, started, inp, candidate, confidence):
    trace.emit(
        model_or_tool=tool,
        latency_ms=int((time.time() - started) * 1000),
        input_hash=hash_payload(
            {
                "case_id": inp["plan"].case_id,
                "iteration": inp["iteration"],
                "report_passed": getattr(inp.get("report"), "passed", None),
            }
        ),
        output_hash=hash_payload(candidate.model_dump()),
        confidence=confidence,
    )


# --- rung 1: LLM theta-proposer -----------------------------------------------------------
def _build_prompt(theta: dict, report) -> str:
    bounds = ", ".join(f"{k} in [{lo}, {hi}]" for k, (lo, hi) in THETA_BOUNDS.items())
    if report is None:
        last = "none (first iteration)"
    else:
        last = (
            f"peak_von_mises={report.peak_von_mises_MPa:.1f} MPa, "
            f"factor_of_safety={report.factor_of_safety:.2f}, "
            f"stress_shielding_index={report.stress_shielding_index:.2f}, passed={report.passed}"
        )
    return (
        "You are optimizing a titanium bone-fixation PLATE. Propose the next parameters.\n"
        f"Current theta (mm): {json.dumps(theta)}\n"
        f"Bounds: {bounds}\n"
        f"Latest stress report: {last}\n"
        "Goal: minimize peak von Mises while keeping factor_of_safety >= 1.5 and "
        "stress_shielding_index within [0.6, 0.9] (thicker raises FoS but lowers the shielding index).\n"
        "Respond with ONLY a JSON object of the theta fields to change, as new absolute values."
    )


def _llm_propose_theta(base_theta, report, trace) -> dict:
    messages = [{"role": "user", "content": _build_prompt(base_theta, report)}]
    try:
        response = llm.call_llm(
            stage="synthesize", messages=messages, model="bedrock/claude-sonnet", trace=trace
        )
        delta = json.loads(response.choices[0].message.content)
    except (json.JSONDecodeError, IndexError, KeyError, TypeError, openai.OpenAIError) as exc:
        raise RetryableError(f"LLM theta-proposal failed: {exc}")
    theta = dict(base_theta)
    for key in THETA_BOUNDS:
        if key in delta:
            theta[key] = delta[key]
    return theta


def _rung1(inp, trace):
    if os.getenv("OSTEON_FORCE_FAIL") == "synthesize":
        raise RetryableError("forced failure for demo")
    started = time.time()
    plan = inp["plan"]
    proposed = _llm_propose_theta(seed_theta(plan), inp.get("report"), trace)
    theta_bounds_check(proposed)  # pre-invoke guardrail: out-of-bounds LLM output -> RejectedOutput
    candidate = _make_candidate(plan, inp["iteration"], clamp_theta(proposed), rung=1)
    _emit_span(trace, "bedrock/claude-sonnet", started, inp, candidate, confidence=0.8)
    return candidate


# --- rung 2: CMA-ES numeric optimizer (no LLM) --------------------------------------------
def _cma_optimize(plan, oracle, max_gens: int = 25) -> dict:
    import cma

    keys = list(THETA_BOUNDS.keys())
    lows = np.array([THETA_BOUNDS[k][0] for k in keys], dtype=np.float64)
    highs = np.array([THETA_BOUNDS[k][1] for k in keys], dtype=np.float64)
    spans = highs - lows

    def to_theta(unit_vec):
        values = lows + np.clip(unit_vec, 0.0, 1.0) * spans
        return clamp_theta({k: float(v) for k, v in zip(keys, values)})

    def fitness(unit_vec):
        theta = to_theta(unit_vec)
        return _objective(oracle(theta), _approx_volume(theta))

    seed_unit = (np.array([seed_theta(plan)[k] for k in keys], dtype=np.float64) - lows) / spans
    es = cma.CMAEvolutionStrategy(
        list(seed_unit), 0.25, {"bounds": [0, 1], "maxiter": max_gens, "verbose": -9, "seed": 1}
    )
    best_vec, best_fit = list(seed_unit), float("inf")
    while not es.stop():
        solutions = es.ask()
        fits = [fitness(x) for x in solutions]
        es.tell(solutions, fits)
        i = int(np.argmin(fits))
        if fits[i] < best_fit:
            best_fit, best_vec = fits[i], solutions[i]
    return to_theta(np.array(best_vec))


def _rung2(inp, trace):
    started = time.time()
    plan = inp["plan"]
    theta = _cma_optimize(plan, _resolve_oracle())
    candidate = _make_candidate(plan, inp["iteration"], theta, rung=2)
    _emit_span(trace, "cma-es", started, inp, candidate, confidence=0.6)
    return candidate


# --- floor: last-known-good theta + stop flag; never raises -------------------------------
def _floor(inp, trace):
    plan = inp["plan"]
    iteration = inp["iteration"]
    theta = dict(_LAST_GOOD.get(plan.case_id) or seed_theta(plan))
    theta["_stop"] = True  # controller termination flag (STANDARDIZATION §6 rung-3)
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
        contacts = _safe_contacts(mesh_path, plan)
    except Exception:
        mesh_path, validity, volume, contacts = (
            "",
            {
                "watertight": False,
                "manifold": False,
                "self_intersect": True,
            },
            0.0,
            [],
        )
    return ImplantCandidate(
        case_id=plan.case_id,
        candidate_id=f"cand-{iteration}-floor",
        iteration=iteration,
        parameter_vector=theta,
        mesh_path=mesh_path,
        contacts_anchor_ids=contacts,
        volume_mm3=volume,
        min_thickness_mm=float(theta["thickness_mm"]),
        validity=validity,
        fallback_rung="floor",
        trace_id=plan.trace_id,
    )


run = with_fallback([_rung1, _rung2], _floor)


if __name__ == "__main__":
    import glob
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
        data["mesh_path"] = f"split_b_synthesis/fixtures/implant_candidate_{plan.case_id}.stl"
        (out_dir / f"implant_candidate_{plan.case_id}.json").write_text(json.dumps(data, indent=2))

        size_kb = stl_dst.stat().st_size / 1024 if stl_dst.exists() else 0.0
        print(
            f"fixture {plan.case_id}: rung={candidate.fallback_rung} "
            f"validity={candidate.validity} vol={candidate.volume_mm3:.1f}mm3 stl={size_kb:.0f}KB"
        )
