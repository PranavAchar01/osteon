"""Osteon — end-to-end dashboard (Split A → B → C).

This server drives the WHOLE resilient pipeline from a CaseSpec, exactly as
``orchestrator.design_implant`` does:

  CaseSpec → Split A localize → PlacementPlan → Split B synthesize → ImplantCandidate
           → Split C evaluate → StressReport   (B↔C feedback until it passes)

It runs the real engines (each with its 3-rung fallback ladder), shows which rung fired
at every stage, serves B's actual placed STL, and renders the dual implant+bone stress
heat-map. The run is cached per (case, load, failure) and, true to the resilience theme,
falls back to the shipped A/B fixtures if a live stage is unavailable.

Run:  cd osteon && source .venv/bin/activate && python webapp/app.py  -> http://127.0.0.1:5001
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_file

import trimesh

import split_b_synthesis.engine as b_engine
from common.contracts import CaseSpec, ImplantCandidate, PlacementPlan
from common.settings import settings
from common.trace import LoopTrace
from split_a_localization.engine import run as localize
from split_b_synthesis.engine import run as synthesize
from split_b_synthesis.mcp_server import generate_mesh as b_generate_mesh
from split_c_evaluation import engine
from split_c_evaluation.engine import run as evaluate

ROOT = Path(__file__).resolve().parent.parent
A_FIXTURES = ROOT / "split_a_localization" / "fixtures"
B_FIXTURES = ROOT / "split_b_synthesis" / "fixtures"
os.chdir(ROOT)  # so every split's relative mesh/trace path resolves regardless of launch cwd
MAX_ITERS = 5
app = Flask(__name__)

MATERIALS = {
    "Ti-6Al-4V (titanium alloy)": {"E_MPa": 110000, "yield_MPa": 830, "endurance_limit_MPa": 510},
    "CoCr (cobalt-chromium)": {"E_MPa": 210000, "yield_MPa": 600, "endurance_limit_MPa": 300},
    "316L stainless steel": {"E_MPa": 193000, "yield_MPa": 290, "endurance_limit_MPa": 240},
    "PEEK (polymer)": {"E_MPa": 3600, "yield_MPa": 100, "endurance_limit_MPa": 30},
}

# Each clinical case pairs Split A's PlacementPlan + Split B's real ImplantCandidate
# (its Blender-generated watertight STL) with the system CaseSpec context C evaluates.
CASES = {
    "c1": {
        "label": "Tibial shaft fracture — titanium",
        "bone": "Tibia",
        "material": "Ti-6Al-4V (titanium alloy)",
        "bone_E": 17000,
        "defect": "Transverse mid-shaft fracture",
        "plan": "placement_plan_test_case_01.json",
        "cand": "implant_candidate_test_case_01.json",
    },
    "c2": {
        "label": "Femoral fracture — stainless steel",
        "bone": "Femur",
        "material": "316L stainless steel",
        "bone_E": 17500,
        "defect": "Oblique mid-shaft fracture",
        "plan": "placement_plan_test_case_02.json",
        "cand": "implant_candidate_test_case_02.json",
    },
    "c3": {
        "label": "Comminuted fracture — cobalt-chrome",
        "bone": "Femur",
        "material": "CoCr (cobalt-chromium)",
        "bone_E": 16000,
        "defect": "Comminuted (multi-fragment) fracture",
        "plan": "placement_plan_test_case_03.json",
        "cand": "implant_candidate_test_case_03.json",
    },
    "c4": {
        "label": "Osteoporotic bone — titanium",
        "bone": "Tibia",
        "material": "Ti-6Al-4V (titanium alloy)",
        "bone_E": 9000,
        "defect": "Low-energy fracture, reduced bone stiffness",
        "plan": "placement_plan_test_case_04.json",
        "cand": "implant_candidate_test_case_04.json",
    },
}
LOADS = {"Walking": 700, "Stair climb": 1500, "Stumble": 2600}
FAIL_MODES = {"none", "evaluate", "evaluate_floor"}

# Demo patient records (fictional) so the dashboard reads like a clinical workspace.
PATIENTS = {
    "c1": {
        "name": "Marcus Chen",
        "age": 34,
        "sex": "M",
        "mrn": "OST-10472",
        "side": "Left",
        "procedure": "ORIF — locking compression plate",
    },
    "c2": {
        "name": "Patricia Alvarez",
        "age": 58,
        "sex": "F",
        "mrn": "OST-20815",
        "side": "Right",
        "procedure": "ORIF — locking compression plate",
    },
    "c3": {
        "name": "James O'Connor",
        "age": 41,
        "sex": "M",
        "mrn": "OST-31190",
        "side": "Right",
        "procedure": "ORIF — bridge plating",
    },
    "c4": {
        "name": "Eleanor Whitfield",
        "age": 72,
        "sex": "F",
        "mrn": "OST-40736",
        "side": "Left",
        "procedure": "ORIF — locking compression plate",
    },
}


def _load_plan(name: str) -> PlacementPlan:
    p = A_FIXTURES / name
    if not p.exists():
        p = ROOT / "fixtures" / "example_placement_plan.json"
    return PlacementPlan(**json.load(open(p)))


def _load_candidate(name: str, case_id: str) -> ImplantCandidate:
    """Split B's REAL shipped ImplantCandidate (its frame-placed, Blender-generated STL).

    B's mesh_path is repo-root-relative and the plate is placed in the bone WORLD frame, so
    we resolve the path absolutely and otherwise pass B's contract through untouched."""
    p = B_FIXTURES / name
    if not p.exists():
        p = ROOT / "fixtures" / "example_implant_candidate.json"
    cand = ImplantCandidate(**json.load(open(p)))
    cand.case_id = case_id
    mp = Path(cand.mesh_path)
    cand.mesh_path = str(mp if mp.is_absolute() else ROOT / mp)
    return cand


def _stl_path(cand: ImplantCandidate) -> Path:
    p = Path(cand.mesh_path)
    return p if p.is_absolute() else ROOT / p


def _read_spans(trace_id: str) -> list[dict]:
    p = ROOT / "traces" / f"{trace_id}.jsonl"
    out = []
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    out.sort(key=lambda s: s.get("ts", 0))
    return out


# --------------------------------------------------------------------------- #
# Full pipeline: build a CaseSpec and run Split A → B ↔ C end to end.
# --------------------------------------------------------------------------- #
def _build_case(case_key: str, load_key: str) -> CaseSpec:
    """The system input. Absolute bone path so every split's trimesh.load resolves."""
    cfg = CASES[case_key]
    mat = MATERIALS[cfg["material"]]
    load_N = LOADS.get(load_key, 700)
    bone = str((ROOT / "fixtures" / "dummy_bone.stl").resolve())
    return CaseSpec(
        case_id=case_key,
        bone_mesh_path=bone,
        bone_material={"E_cortical_MPa": cfg["bone_E"], "E_trabecular_MPa": 1000, "density": 1.9},
        defect={"type": "fracture", "region": "diaphysis", "severity": "moderate",
                "description": cfg["defect"]},
        load_profile=[{"name": load_key, "force_vector_N": {"x": 0, "y": 0, "z": load_N},
                       "application_region": "mid-diaphysis", "cycles": 1_000_000}],
        implant_material={"name": cfg["material"], **mat},
        constraints={"process": "additive"},
    )


def _abs_mesh(path: str) -> str:
    if not path:
        return path
    p = Path(path)
    return str(p if p.is_absolute() else (ROOT / p))


def _placement_json(plan: PlacementPlan):
    """Replicate Split B's placement inputs (anchors + bone-frame) from the plan, so we can
    re-mesh a resized plate onto the same anchors without reaching into B's internals."""
    anchors = [json.loads(a.model_dump_json()) for a in plan.anchor_points]
    pts = [(a.xyz.x, a.xyz.y, a.xyz.z) for a in plan.anchor_points]
    if pts:
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        cz = sum(p[2] for p in pts) / len(pts)
    else:
        cx = cy = cz = 0.0
    basis = (plan.coordinate_frame or {}).get("basis") or [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    return anchors, {"origin": {"x": cx, "y": cy, "z": cz}, "basis": basis}


def _rightsize(plan: PlacementPlan, cand: ImplantCandidate, case: CaseSpec) -> ImplantCandidate:
    """Size the plate THICKNESS to the thinnest value meeting the FoS target for this
    patient's load — the standard implant-design calc (3-point bending: peak = 1.5·P·L/(w·t²),
    so t = √(1.5·P·L·FoS_target/(w·yield))) — then re-place + re-mesh via Split B's tool.

    This closes the design loop with the patient's actual load: heavier loads / weaker
    materials get a thicker plate, lighter ones a thinner plate, so the implant is genuinely
    different per patient instead of B's load-blind default. Falls back to B's mesh on error."""
    pv = dict(cand.parameter_vector or {})
    if pv.get("_stop") or cand.fallback_rung == "floor":
        return cand  # floor candidate — leave it untouched
    length = float(pv.get("length_mm", 96.0))
    width = float(pv.get("width_mm", 14.0))
    P, _cycles = engine._load_N(case)
    _e, yld, _en = engine._material(case)
    fos_target = 1.6
    t_req = (1.5 * P * length * fos_target / (max(width, 1.0) * max(yld, 1.0))) ** 0.5
    t_req = min(max(t_req, 2.0), 8.0)  # THETA_BOUNDS thickness window
    if abs(t_req - float(pv.get("thickness_mm", 0.0))) < 0.25:
        return cand  # already sized correctly
    pv["thickness_mm"] = t_req
    try:
        anchors, frame = _placement_json(plan)
        res = b_generate_mesh(b_engine.clamp_theta(pv), anchors, frame)
        mp = _abs_mesh(res["mesh_path"])
        vol = float(abs(trimesh.load(mp, force="mesh").volume))  # read first → no partial update
        cand.parameter_vector = b_engine.clamp_theta(pv)
        cand.mesh_path = mp
        cand.validity = res["validity"]
        cand.min_thickness_mm = float(t_req)
        cand.volume_mm3 = vol
    except Exception:
        pass  # keep B's original candidate if the re-mesh fails
    return cand


def _design(case: CaseSpec, fail: str) -> dict:
    """Run A → (B ↔ C) live, like orchestrator.design_implant, but capture the rung that
    fired at every stage + the iteration count and honour the failure-injection toggle.
    The ladder floors never raise, so this returns a valid result for any input."""
    seed = int(hashlib.md5(case.case_id.encode()).hexdigest(), 16) % (2**32)
    np.random.seed(seed)  # process-independent → reproducible anchors across restarts
    trace = LoopTrace(case.case_id, stage="design")
    prev = os.environ.get("OSTEON_FORCE_FAIL")
    if fail and fail != "none":
        os.environ["OSTEON_FORCE_FAIL"] = fail
    else:
        os.environ.pop("OSTEON_FORCE_FAIL", None)
    try:
        plan = localize(case, trace.child("localize"))
        report = None
        cand = None
        iters = 0
        for i in range(MAX_ITERS):
            iters = i + 1
            cand = synthesize({"plan": plan, "report": report, "iteration": i},
                              trace.child("synthesize"))
            report = evaluate({"candidate": cand, "case": case, "mode": "three_point"},
                              trace.child("evaluate"))
            if report.passed or (cand.parameter_vector or {}).get("_stop") or cand.fallback_rung == "floor":
                break
        # size the plate to THIS patient's load, then re-evaluate the resized implant
        cand = _rightsize(plan, cand, case)
        report = evaluate({"candidate": cand, "case": case, "mode": "three_point"},
                          trace.child("evaluate"))
    finally:
        if prev is not None:
            os.environ["OSTEON_FORCE_FAIL"] = prev
        else:
            os.environ.pop("OSTEON_FORCE_FAIL", None)
    cand.mesh_path = _abs_mesh(cand.mesh_path)
    return {
        "plan": plan, "cand": cand, "report": report, "iters": iters,
        "trace_id": trace.trace_id, "live": True,
        "rungs": {"localize": plan.fallback_rung, "synthesize": cand.fallback_rung,
                  "evaluate": report.fallback_rung},
    }


def _design_from_fixtures(case_key: str, load_key: str, fail: str) -> dict:
    """Resilience fallback: if a live stage is unavailable, evaluate the shipped A+B
    fixtures with Split C so the dashboard still produces a valid result."""
    cfg = CASES[case_key]
    plan = _load_plan(cfg["plan"])
    cand = _load_candidate(cfg["cand"], plan.case_id)
    case = _build_case(case_key, load_key)
    trace = LoopTrace(plan.case_id, stage="evaluate")
    cand.trace_id = trace.trace_id
    prev = os.environ.get("OSTEON_FORCE_FAIL")
    if fail and fail != "none":
        os.environ["OSTEON_FORCE_FAIL"] = fail
    else:
        os.environ.pop("OSTEON_FORCE_FAIL", None)
    try:
        report = evaluate({"candidate": cand, "case": case, "mode": "three_point"},
                          trace.child("evaluate"))
    finally:
        if prev is not None:
            os.environ["OSTEON_FORCE_FAIL"] = prev
        else:
            os.environ.pop("OSTEON_FORCE_FAIL", None)
    return {
        "plan": plan, "cand": cand, "report": report, "iters": 1,
        "trace_id": trace.trace_id, "live": False,
        "rungs": {"localize": plan.fallback_rung, "synthesize": cand.fallback_rung,
                  "evaluate": report.fallback_rung},
    }


_PIPELINE_CACHE: dict = {}
_LAST_CAND: dict = {}  # case id -> absolute STL path of the most recent candidate (for /mesh)
_LAST_RESULT: dict = {}  # case id -> full pipeline result (for /api/heatmap on any id)


def _remember(case_key: str, res: dict) -> dict:
    """Record the most recent result for a case id so /mesh + /api/heatmap can find it."""
    _LAST_CAND[case_key] = _abs_mesh(res["cand"].mesh_path)
    _LAST_RESULT[case_key] = res
    return res


def _pipeline(case_key: str, load_key: str, fail: str) -> dict:
    """Cached full-pipeline run; live A→B→C with a fixture fallback (never raises)."""
    key = (case_key, load_key, fail)
    if key not in _PIPELINE_CACHE:
        case = _build_case(case_key, load_key)
        try:
            res = _design(case, fail)
            if not (res["cand"].mesh_path and Path(res["cand"].mesh_path).exists()):
                raise RuntimeError("live candidate mesh missing")
        except Exception:
            res = _design_from_fixtures(case_key, load_key, fail)
        res["case"] = _build_case(case_key, load_key)
        _PIPELINE_CACHE[key] = res
    return _remember(case_key, _PIPELINE_CACHE[key])


@app.route("/")
def index():
    patients = []
    for k, cfg in CASES.items():
        p = PATIENTS[k]
        patients.append(
            {
                "id": k,
                "name": p["name"],
                "initials": "".join(w[0] for w in p["name"].split()[:2]).upper(),
                "demo": f'{p["age"]} {p["sex"]}',
                "site": f'{p["side"]} {cfg["bone"].lower()} · {cfg["material"].split(" (")[0]}',
            }
        )
    return render_template(
        "index.html", patients=patients, loads=list(LOADS.keys()),
        materials=list(MATERIALS.keys()),
    )


@app.route("/mesh/<case_id>.stl")
def mesh(case_id):
    # the most recent live candidate for this id (preset OR custom); else the B fixture
    path = _LAST_CAND.get(case_id)
    if (not path or not Path(path).exists()) and case_id in CASES:
        path = _abs_mesh(_load_candidate(CASES[case_id]["cand"], "mesh").mesh_path)
    if not path or not Path(path).exists():
        abort(404)
    return send_file(path, mimetype="model/stl")


@app.route("/bone.stl")
def bone():
    """Split A's realistic femur fixture (the fit-target bone surface)."""
    p = ROOT / "fixtures" / "dummy_bone.stl"
    if not p.exists():
        abort(404)
    return send_file(p, mimetype="model/stl")


def _response(case_key: str, meta: dict, res: dict) -> dict:
    """Build the dashboard payload from a pipeline result + display metadata (used by both
    the preset /api/run and the live /api/design)."""
    plan, candidate, report = res["plan"], res["cand"], res["report"]
    g = engine._geometry(candidate)  # true plate dims from B's parameter vector (tilt-safe)
    pv = candidate.parameter_vector or {}
    stl_ok = bool(candidate.mesh_path and Path(candidate.mesh_path).exists())
    contacts = list(candidate.contacts_anchor_ids or [])
    mat = MATERIALS.get(meta["material"], next(iter(MATERIALS.values())))
    return {
        "patient": meta["patient"],
        "case": {
            "label": meta["label"], "bone": meta["bone"], "defect": meta["defect"],
            "material": meta["material"], "load_scenario": meta["load_scenario"],
            "load_N": meta["load_N"], "bone_E": meta["bone_E"],
        },
        "live": res["live"],
        "iterations": res["iters"],
        "rungs": res["rungs"],
        "from_a": {
            "anchors": len(plan.anchor_points),
            "confidence": round(plan.confidence, 2),
            "fit_target": Path(plan.fit_target_surface_path).name,
            "rung": plan.fallback_rung,
        },
        "from_b": {
            "length_mm": round(g.L, 1),
            "width_mm": round(g.b, 1),
            "thickness_mm": round(g.h, 1),
            "n_screws": pv.get("n_screws", 4),
            "volume_mm3": round(candidate.volume_mm3, 0),
            "watertight": candidate.validity.get("watertight", False),
            "stl": stl_ok,
            "mesh_file": Path(candidate.mesh_path).name,
            "contacts": len(contacts),
            "rung": candidate.fallback_rung,
        },
        "report": report.model_dump(),
        "material": {"name": meta["material"], **mat},
        "geom": {
            "L": round(g.L, 1), "b": round(g.b, 1), "h": round(g.h, 1),
            "I_mm4": round(g.I, 1), "Z_mm3": round(g.section_modulus, 1),
        },
        # ?v=<mesh stem> busts the browser STL cache when the implant geometry changes
        # (e.g. re-running the same patient at a heavier load resizes the plate)
        "mesh_url": (f"/mesh/{case_key}.stl?v={Path(candidate.mesh_path).stem}" if stl_ok else None),
        "bone_url": "/bone.stl" if (ROOT / "fixtures" / "dummy_bone.stl").exists() else None,
        "spans": _read_spans(res["trace_id"]),
    }


@app.route("/api/run", methods=["POST"])
def run():
    b = request.get_json(force=True, silent=True) or {}
    case_key = b.get("case") if b.get("case") in CASES else next(iter(CASES))
    cfg = CASES[case_key]
    load_key = b.get("load") if b.get("load") in LOADS else "Walking"
    fail = b.get("fail_mode") if b.get("fail_mode") in FAIL_MODES else "none"

    res = _pipeline(case_key, load_key, fail)  # live A → B ↔ C (cached), fixture fallback
    meta = {
        "label": cfg["label"], "bone": cfg["bone"], "defect": cfg["defect"],
        "material": cfg["material"], "load_scenario": load_key, "load_N": LOADS[load_key],
        "bone_E": cfg["bone_E"],
        "patient": {**PATIENTS.get(case_key, {}), "bone": cfg["bone"],
                    "defect": cfg["defect"], "date": "2026-06-03"},
    }
    return jsonify(_response(case_key, meta, res))


@app.route("/api/design", methods=["POST"])
def design():
    """Create a NEW patient/case from form fields and run the full A→B↔C pipeline LIVE.

    The bone scan is the shared fixture femur; everything else (material, load, bone
    stiffness, defect) is custom, computed in real time. Cached per (id, load, failure)."""
    b = request.get_json(force=True, silent=True) or {}
    name = (str(b.get("name") or "New Patient").strip() or "New Patient")[:60]
    bone = b.get("bone") if b.get("bone") in ("Tibia", "Femur") else "Femur"
    material = b.get("material") if b.get("material") in MATERIALS else next(iter(MATERIALS))
    defect = (str(b.get("defect") or "Transverse mid-shaft fracture").strip())[:140]
    load_key = b.get("load") if b.get("load") in LOADS else "Walking"
    fail = b.get("fail_mode") if b.get("fail_mode") in FAIL_MODES else "none"
    cid = (str(b.get("cid") or "")[:48] or "custom-x")
    if not cid.startswith("custom-"):
        cid = "custom-" + cid
    try:
        bone_E = min(max(float(b.get("bone_E") or 17000), 1000.0), 30000.0)
    except (TypeError, ValueError):
        bone_E = 17000.0

    mat = MATERIALS[material]
    load_N = LOADS[load_key]
    case = CaseSpec(
        case_id=cid,
        bone_mesh_path=str((ROOT / "fixtures" / "dummy_bone.stl").resolve()),
        bone_material={"E_cortical_MPa": bone_E, "E_trabecular_MPa": 1000, "density": 1.9},
        defect={"type": "fracture", "region": "diaphysis", "severity": "moderate",
                "description": defect},
        load_profile=[{"name": load_key, "force_vector_N": {"x": 0, "y": 0, "z": load_N},
                       "application_region": "mid-diaphysis", "cycles": 1_000_000}],
        implant_material={"name": material, **mat},
        constraints={"process": "additive"},
    )

    key = (cid, load_key, fail, round(bone_E), material)
    if key not in _PIPELINE_CACHE:
        try:
            res = _design(case, fail)
            if not (res["cand"].mesh_path and Path(res["cand"].mesh_path).exists()):
                raise RuntimeError("live candidate mesh missing")
        except Exception as exc:
            return jsonify({"error": f"pipeline failed: {str(exc)[:160]}"}), 200
        res["case"] = case
        _PIPELINE_CACHE[key] = res
    res = _remember(cid, _PIPELINE_CACHE[key])

    initials = ("".join(w[0] for w in name.split()[:2]).upper() or "NP")[:2]
    age, sex, side = b.get("age") or "—", b.get("sex") or "", b.get("side") or "—"
    meta = {
        "label": f"{bone} fracture — {material.split(' (')[0]}",
        "bone": bone, "defect": defect, "material": material,
        "load_scenario": load_key, "load_N": load_N, "bone_E": bone_E,
        "patient": {"name": name, "age": age, "sex": sex, "mrn": "OST-" + cid[-5:].upper(),
                    "side": side, "procedure": "ORIF — custom plan", "bone": bone,
                    "defect": defect, "date": "2026-06-05"},
    }
    out = _response(cid, meta, res)
    out["new_patient"] = {
        "id": cid, "name": name, "initials": initials,
        "demo": f"{age} {sex}".strip(),
        "site": f"{bone.lower()} · {material.split(' (')[0]}",
    }
    return jsonify(out)


def _blender_bin() -> str:
    return (
        os.environ.get("OSTEON_BLENDER")
        or shutil.which("blender")
        or "/Applications/Blender.app/Contents/MacOS/Blender"
    )


def _heatmap_paths(case_key: str):
    # the renderer names outputs after the candidate mesh stem (live, custom, or fixture)
    mp = _LAST_CAND.get(case_key)
    if not mp and case_key in CASES:
        mp = _abs_mesh(_load_candidate(CASES[case_key]["cand"], "hm").mesh_path)
    if not mp:
        return None, None
    stem = Path(mp).stem
    out = Path(settings.OSTEON_TRACE_DIR).resolve()  # absolute, so send_file works
    return out / f"{stem}_heatmap.png", out / f"{stem}_heatmap.blend"


@app.route("/api/heatmap", methods=["POST"])
def heatmap():
    """Render the dual implant+bone Blender stress model (PNG + .blend) for the case,
    using the SAME candidate the pipeline produced + A's real plan for the bone field.
    Works for both preset cases and live custom patients."""
    b = request.get_json(force=True, silent=True) or {}
    case_key = b.get("case") or next(iter(CASES))
    load_key = b.get("load") if b.get("load") in LOADS else "Walking"
    res = _LAST_RESULT.get(case_key)
    if res is None:
        if case_key in CASES:
            res = _pipeline(case_key, load_key, "none")
        else:
            return jsonify({"error": "run this case first"}), 200
    candidate, case, plan = res["cand"], res["case"], res["plan"]
    trace = LoopTrace(case.case_id, stage="evaluate")
    candidate.trace_id = trace.trace_id
    try:
        out = engine.render_heatmap(candidate, case, trace, plan=plan)  # PNG + .blend
    except Exception as exc:  # never 500 the UI
        return jsonify({"error": str(exc)[:200]}), 200
    return jsonify(
        {
            "png_url": f"/heatmap/{case_key}.png",
            "blend_url": f"/heatmap/{case_key}.blend",
            "peak_mpa": out["peak_mpa"],
            "solver_used": out["solver_used"],
            "blend_name": Path(out["blend_path"]).name,
            "bone": out.get("bone"),  # bone-load peak + balance score (cool colour system)
        }
    )


@app.route("/heatmap/<case_id>.png")
def heatmap_png(case_id):
    png, _ = _heatmap_paths(case_id)
    if not png or not png.exists():
        abort(404)
    return send_file(png, mimetype="image/png")


@app.route("/heatmap/<case_id>.blend")
def heatmap_blend(case_id):
    _, blend = _heatmap_paths(case_id)
    if not blend or not blend.exists():
        abort(404)
    return send_file(
        blend,
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name=f"osteon_{case_id}_stress.blend",
    )


@app.route("/api/open-blender", methods=["POST"])
def open_blender():
    """Launch the local Blender GUI with the rendered .blend so the user sees the model."""
    b = request.get_json(force=True, silent=True) or {}
    case_key = b.get("case") or next(iter(CASES))
    _, blend = _heatmap_paths(case_key)
    if not blend or not blend.exists():
        return jsonify({"ok": False, "error": "render the model first"}), 200
    blender = _blender_bin()
    if not (os.path.exists(blender) or shutil.which(blender)):
        return jsonify({"ok": False, "error": "Blender not found"}), 200
    try:
        subprocess.Popen([blender, str(blend)])  # detached GUI
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)[:200]}), 200
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=False)
