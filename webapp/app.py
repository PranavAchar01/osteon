"""Osteon — Split C dashboard.

Split C is the evaluator. It receives the OUTPUT of the upstream pipeline:
  • a PlacementPlan from Split A   (anchors + fit-target bone surface), and
  • an ImplantCandidate from Split B (a path to a watertight STL mesh + parametric theta),
together with the system CaseSpec, and produces a StressReport.

This server loads Split A's and Split B's shipped fixtures (B's real Blender-generated
implant STLs), runs the real Split C engine on them, and serves the actual STL to the
browser so the 3-D view shows B's true geometry — not a procedural stand-in.

Run:  cd osteon && source .venv/bin/activate && python webapp/app.py  -> http://127.0.0.1:5001
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file

from common.contracts import CaseSpec, ImplantCandidate, PlacementPlan
from common.settings import settings
from common.trace import LoopTrace
from split_c_evaluation import engine
from split_c_evaluation.engine import run as evaluate

ROOT = Path(__file__).resolve().parent.parent
A_FIXTURES = ROOT / "split_a_localization" / "fixtures"
B_FIXTURES = ROOT / "split_b_synthesis" / "fixtures"
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
    return render_template("index.html", patients=patients, loads=list(LOADS.keys()))


@app.route("/mesh/<case_id>.stl")
def mesh(case_id):
    if case_id not in CASES:
        abort(404)
    cand = _load_candidate(CASES[case_id]["cand"], "mesh")
    path = Path(cand.mesh_path)
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="model/stl")


@app.route("/bone.stl")
def bone():
    """Split A's realistic femur fixture (the fit-target bone surface)."""
    p = ROOT / "fixtures" / "dummy_bone.stl"
    if not p.exists():
        abort(404)
    return send_file(p, mimetype="model/stl")


@app.route("/api/run", methods=["POST"])
def run():
    b = request.get_json(force=True, silent=True) or {}
    case_key = b.get("case") if b.get("case") in CASES else next(iter(CASES))
    cfg = CASES[case_key]
    load_key = b.get("load") if b.get("load") in LOADS else "Walking"
    load_N = LOADS[load_key]
    fail = b.get("fail_mode") if b.get("fail_mode") in FAIL_MODES else "none"
    mat = MATERIALS[cfg["material"]]

    plan = _load_plan(cfg["plan"])
    candidate = _load_candidate(cfg["cand"], plan.case_id)  # Split B's REAL frame-placed implant
    trace = LoopTrace(plan.case_id, stage="evaluate")
    candidate.trace_id = trace.trace_id

    case = CaseSpec(
        case_id=plan.case_id,
        bone_mesh_path=plan.fit_target_surface_path,
        bone_material={"E_cortical_MPa": cfg["bone_E"], "E_trabecular_MPa": 1000, "density": 1.9},
        defect={
            "type": "fracture",
            "region": "diaphysis",
            "severity": "moderate",
            "description": cfg["defect"],
        },
        load_profile=[
            {
                "name": load_key,
                "force_vector_N": {"x": 0, "y": 0, "z": load_N},
                "application_region": "mid-diaphysis",
                "cycles": 1_000_000,
            }
        ],
        implant_material={"name": cfg["material"], **mat},
        constraints={"process": "additive"},
    )

    prev = os.environ.get("OSTEON_FORCE_FAIL")
    if fail != "none":
        os.environ["OSTEON_FORCE_FAIL"] = fail
    else:
        os.environ.pop("OSTEON_FORCE_FAIL", None)
    try:
        report = evaluate(
            {"candidate": candidate, "case": case, "mode": "three_point"}, trace.child("evaluate")
        )
    finally:
        if prev is not None:
            os.environ["OSTEON_FORCE_FAIL"] = prev
        else:
            os.environ.pop("OSTEON_FORCE_FAIL", None)

    g = engine._geometry(candidate)  # true plate dims from B's parameter vector (tilt-safe)
    pv = candidate.parameter_vector
    stl_ok = _stl_path(candidate).exists()
    contacts = list(candidate.contacts_anchor_ids or [])
    return jsonify(
        {
            "patient": {
                **PATIENTS.get(case_key, {}),
                "bone": cfg["bone"],
                "defect": cfg["defect"],
                "date": "2026-06-03",
            },
            "case": {
                "label": cfg["label"],
                "bone": cfg["bone"],
                "defect": cfg["defect"],
                "material": cfg["material"],
                "load_scenario": load_key,
                "load_N": load_N,
                "bone_E": cfg["bone_E"],
            },
            "from_a": {
                "anchors": len(plan.anchor_points),
                "confidence": round(plan.confidence, 2),
                "fit_target": Path(plan.fit_target_surface_path).name,
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
            "material": {"name": cfg["material"], **mat},
            "geom": {
                "L": round(g.L, 1),
                "b": round(g.b, 1),
                "h": round(g.h, 1),
                "I_mm4": round(g.I, 1),
                "Z_mm3": round(g.section_modulus, 1),
            },
            "mesh_url": f"/mesh/{case_key}.stl" if stl_ok else None,
            "bone_url": "/bone.stl" if (ROOT / "fixtures" / "dummy_bone.stl").exists() else None,
            "spans": _read_spans(report.trace_id),
        }
    )


def _blender_bin() -> str:
    return (
        os.environ.get("OSTEON_BLENDER")
        or shutil.which("blender")
        or "/Applications/Blender.app/Contents/MacOS/Blender"
    )


def _make_case(cfg: dict, plan: PlacementPlan, load_key: str) -> CaseSpec:
    mat = MATERIALS[cfg["material"]]
    load_N = LOADS.get(load_key, 700)
    return CaseSpec(
        case_id=plan.case_id,
        bone_mesh_path=plan.fit_target_surface_path,
        bone_material={"E_cortical_MPa": cfg["bone_E"], "E_trabecular_MPa": 1000, "density": 1.9},
        defect={
            "type": "fracture",
            "region": "diaphysis",
            "severity": "moderate",
            "description": cfg["defect"],
        },
        load_profile=[
            {
                "name": load_key,
                "force_vector_N": {"x": 0, "y": 0, "z": load_N},
                "application_region": "mid-diaphysis",
                "cycles": 1_000_000,
            }
        ],
        implant_material={"name": cfg["material"], **mat},
        constraints={"process": "additive"},
    )


def _heatmap_paths(case_key: str):
    # the renderer names outputs after B's mesh stem (e.g. implant_candidate_test_case_01)
    stem = Path(_load_candidate(CASES[case_key]["cand"], "hm").mesh_path).stem
    out = Path(settings.OSTEON_TRACE_DIR).resolve()  # absolute, so send_file works
    return out / f"{stem}_heatmap.png", out / f"{stem}_heatmap.blend"


@app.route("/api/heatmap", methods=["POST"])
def heatmap():
    """Render the Blender stress model (PNG + .blend) for the selected case."""
    b = request.get_json(force=True, silent=True) or {}
    case_key = b.get("case") if b.get("case") in CASES else next(iter(CASES))
    load_key = b.get("load") if b.get("load") in LOADS else "Walking"
    cfg = CASES[case_key]
    plan = _load_plan(cfg["plan"])
    candidate = _load_candidate(cfg["cand"], plan.case_id)
    trace = LoopTrace(plan.case_id, stage="evaluate")
    candidate.trace_id = trace.trace_id
    case = _make_case(cfg, plan, load_key)
    try:
        out = engine.render_heatmap(candidate, case, trace)  # -> PNG + .blend under traces/
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
    if case_id not in CASES:
        abort(404)
    png, _ = _heatmap_paths(case_id)
    if not png.exists():
        abort(404)
    return send_file(png, mimetype="image/png")


@app.route("/heatmap/<case_id>.blend")
def heatmap_blend(case_id):
    if case_id not in CASES:
        abort(404)
    _, blend = _heatmap_paths(case_id)
    if not blend.exists():
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
    case_key = b.get("case") if b.get("case") in CASES else next(iter(CASES))
    _, blend = _heatmap_paths(case_key)
    if not blend.exists():
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
