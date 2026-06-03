"""Osteon — Split C dashboard.

Split C is the evaluator. It receives the OUTPUT of the upstream pipeline:
  • a PlacementPlan from Split A  (anchors + fit-target bone surface), and
  • an ImplantCandidate from Split B  (the parametric implant + screws),
together with the system CaseSpec, and produces a StressReport.

This server runs the real Split B and Split C engines on Split A's shipped
PlacementPlan fixtures — so what you see is genuinely "from the output of A and B".
No torch / no network in the request path, so it stays fast.

Run:  cd osteon && source .venv/bin/activate && python webapp/app.py  -> http://127.0.0.1:5001
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from common.contracts import CaseSpec, PlacementPlan
from common.trace import LoopTrace
from split_b_synthesis.engine import run as synthesize
from split_c_evaluation import fea
from split_c_evaluation.engine import run as evaluate

ROOT = Path(__file__).resolve().parent.parent
app = Flask(__name__)

MATERIALS = {
    "Ti-6Al-4V (titanium alloy)": {"E_MPa": 110000, "yield_MPa": 830, "endurance_limit_MPa": 510},
    "CoCr (cobalt-chromium)": {"E_MPa": 210000, "yield_MPa": 600, "endurance_limit_MPa": 300},
    "316L stainless steel": {"E_MPa": 193000, "yield_MPa": 290, "endurance_limit_MPa": 240},
    "PEEK (polymer)": {"E_MPa": 3600, "yield_MPa": 100, "endurance_limit_MPa": 30},
}
A_FIXTURES = ROOT / "split_a_localization" / "fixtures"

# Clinical cases. Each pairs Split A's shipped PlacementPlan (its real output) with the
# system CaseSpec context (bone, implant material, defect) that Split C evaluates against.
CASES = {
    "tibia_ti": {
        "label": "Tibial shaft fracture — titanium plate",
        "bone": "Tibia",
        "material": "Ti-6Al-4V (titanium alloy)",
        "bone_E": 17000,
        "defect": "Transverse mid-shaft fracture",
        "plan": "placement_plan_test_case_01.json",
    },
    "femur_steel": {
        "label": "Femoral fracture — stainless plate",
        "bone": "Femur",
        "material": "316L stainless steel",
        "bone_E": 17500,
        "defect": "Oblique mid-shaft fracture",
        "plan": "placement_plan_test_case_02.json",
    },
    "comminuted_cocr": {
        "label": "Comminuted fracture — cobalt-chrome plate",
        "bone": "Femur",
        "material": "CoCr (cobalt-chromium)",
        "bone_E": 16000,
        "defect": "Comminuted (multi-fragment) fracture",
        "plan": "placement_plan_test_case_03.json",
    },
    "osteoporotic_ti": {
        "label": "Osteoporotic bone — titanium plate",
        "bone": "Tibia",
        "material": "Ti-6Al-4V (titanium alloy)",
        "bone_E": 9000,
        "defect": "Low-energy fracture, reduced bone stiffness",
        "plan": "placement_plan_test_case_04.json",
    },
}
LOADS = {"Walking": 700, "Stair climb": 1500, "Stumble": 2600}
FAIL_MODES = {"none", "evaluate", "evaluate_floor"}


def _load_plan(name: str) -> PlacementPlan:
    path = A_FIXTURES / name
    if not path.exists():  # fall back to the canonical example
        path = ROOT / "fixtures" / "example_placement_plan.json"
    return PlacementPlan(**json.load(open(path)))


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
    cases = [{"id": k, "label": v["label"]} for k, v in CASES.items()]
    return render_template("index.html", cases=cases, loads=list(LOADS.keys()))


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
    trace = LoopTrace(plan.case_id, stage="evaluate")
    plan.trace_id = trace.trace_id  # fresh trace per run

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
        # Split B's real engine produces the candidate from A's plan ...
        candidate = synthesize(
            {"plan": plan, "report": None, "iteration": 0}, trace.child("synthesize")
        )
        # ... and Split C (this project) evaluates it.
        report = evaluate(
            {"candidate": candidate, "case": case, "mode": "three_point"}, trace.child("evaluate")
        )
    finally:
        if prev is not None:
            os.environ["OSTEON_FORCE_FAIL"] = prev
        else:
            os.environ.pop("OSTEON_FORCE_FAIL", None)

    pv = candidate.parameter_vector
    g = fea.beam_from_dims(pv["length_mm"], pv["width_mm"], pv["thickness_mm"])
    return jsonify(
        {
            "case": {
                "label": cfg["label"],
                "bone": cfg["bone"],
                "defect": cfg["defect"],
                "material": cfg["material"],
                "load_scenario": load_key,
                "load_N": load_N,
            },
            "from_a": {
                "anchors": len(plan.anchor_points),
                "confidence": round(plan.confidence, 2),
                "rung": plan.fallback_rung,
                "fit_target": Path(plan.fit_target_surface_path).name,
            },
            "from_b": {
                "length_mm": pv["length_mm"],
                "width_mm": pv["width_mm"],
                "thickness_mm": pv["thickness_mm"],
                "n_screws": pv.get("n_screws", 4),
                "volume_mm3": candidate.volume_mm3,
                "watertight": candidate.validity.get("watertight", False),
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
            "spans": _read_spans(report.trace_id),
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=False)
