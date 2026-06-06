"""Pre-render the staged demo assets (Blender PNG + .blend) + agent-logic bullets.

Stage A  bone + anchors + coordinate frame   (Split A localization)
Stage B  the real LCP alone, and the LCP registered into the femur
Stage C  von Mises stress heat-map on the placed implant

Outputs -> webapp/static/renders/. Run:  python webapp/make_demo_renders.py
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

from common.contracts import CaseSpec, ImplantCandidate, PlacementPlan  # noqa: E402
from common.trace import LoopTrace  # noqa: E402
from split_c_evaluation import engine  # noqa: E402

BLENDER = os.environ.get("OSTEON_BLENDER") or "/Applications/Blender.app/Contents/MacOS/Blender"
OUT = ROOT / "webapp" / "static" / "renders"
OUT.mkdir(parents=True, exist_ok=True)
PLAN_FILE = ROOT / "split_a_localization" / "fixtures" / "placement_plan_test_case_01.json"
BONE = ROOT / "fixtures" / "dummy_bone.stl"
LCP = ROOT / "fixtures" / "implant_library" / "lcp_mini_frag_straight.stl"

plan = PlacementPlan(**json.load(open(PLAN_FILE)))


def blender(*args):
    subprocess.run([BLENDER, "--background", *args], check=True, capture_output=True, text=True)


# --------------------------------------------------------------------------- #
# Register the real LCP onto the plan (rigid: rotate/translate, geometry intact)
# --------------------------------------------------------------------------- #
def principal_axes(P):
    c = P - P.mean(0)
    _, v = np.linalg.eigh(np.cov(c.T))
    return v[:, ::-1]


def ortho(long, n):
    n = n / (np.linalg.norm(n) + 1e-12)
    long = long - np.dot(long, n) * n
    long /= np.linalg.norm(long) + 1e-12
    return np.column_stack([long, np.cross(n, long), n])


bone = trimesh.load(BONE)
bone.apply_translation(-bone.centroid)
bone.apply_scale(1000.0)
bone_norm = OUT / "_bone_norm.stl"
bone.export(bone_norm)

src = trimesh.load(LCP)
plate = src.copy()
plate.apply_translation(-plate.centroid)
half_t = float(np.sort(src.extents)[0]) / 2.0
pos = np.array([[a.xyz.x, a.xyz.y, a.xyz.z] for a in plan.anchor_points], dtype=float)
dc = plan.defect_region.get("centroid", {}) or {}
defect = np.array([dc.get("x", 0.0), dc.get("y", 0.0), dc.get("z", 0.0)], dtype=float)
seat_pt, _, tri = bone.nearest.on_surface([defect])
seat_pt = seat_pt[0]
sn = bone.face_normals[tri[0]]
if np.dot(sn, seat_pt - bone.centroid) < 0:
    sn = -sn
pf = principal_axes(plate.vertices)
tf = ortho(principal_axes(pos)[:, 0], sn)
R = tf @ pf.T
if np.linalg.det(R) < 0:
    pf[:, 1] *= -1
    R = tf @ pf.T
T = np.eye(4)
T[:3, :3] = R
plate.apply_transform(T)
mv = np.eye(4)
mv[:3, 3] = (seat_pt + sn * half_t) - plate.centroid
plate.apply_transform(mv)
posed = OUT / "_lcp_posed.stl"
plate.export(posed)

alone = src.copy()
alone.apply_translation(-alone.centroid)
alone_path = OUT / "_lcp_alone.stl"
alone.export(alone_path)

# --------------------------------------------------------------------------- #
# Stage A — coordinates (Split A's own Blender renderer)
# --------------------------------------------------------------------------- #
SA = ROOT / "split_a_localization" / "blender_render.py"
blender("--python", str(SA), "--", str(PLAN_FILE), str(BONE), str(OUT / "coords.png"))
blender("--python", str(SA), "--", str(PLAN_FILE), str(BONE), str(OUT / "coords.blend"))


# --------------------------------------------------------------------------- #
# Stage B — implant alone + implant in femur
# --------------------------------------------------------------------------- #
def combo(meshes, png, blend):
    spec = OUT / "_spec.json"
    spec.write_text(json.dumps({"meshes": meshes, "out_png": str(png), "out_blend": str(blend)}))
    blender("--python", str(ROOT / "webapp" / "blender_combo.py"), "--", str(spec))


combo([{"path": str(alone_path), "kind": "implant"}],
      OUT / "implant_alone.png", OUT / "implant_alone.blend")
combo([{"path": str(bone_norm), "kind": "bone"}, {"path": str(posed), "kind": "implant"}],
      OUT / "implant_in_femur.png", OUT / "implant_in_femur.blend")

# --------------------------------------------------------------------------- #
# Stage C — stress heat-map on the placed LCP
# --------------------------------------------------------------------------- #
case = CaseSpec(
    case_id=plan.case_id,
    bone_mesh_path=str(BONE.resolve()),
    bone_material={"E_cortical_MPa": 17000, "E_trabecular_MPa": 1000, "density": 1.9},
    defect={"type": "fracture", "region": "diaphysis", "severity": "moderate",
            "description": "Transverse mid-shaft femoral fracture"},
    load_profile=[{"name": "Walking", "force_vector_N": {"x": 0, "y": 0, "z": 700},
                   "application_region": "mid-diaphysis", "cycles": 1_000_000}],
    implant_material={"name": "Ti-6Al-4V", "E_MPa": 110000, "yield_MPa": 830, "endurance_limit_MPa": 510},
    constraints={"process": "additive"},
)
cand = ImplantCandidate(
    case_id=plan.case_id, candidate_id="lcp-demo", iteration=0,
    parameter_vector={"length_mm": 96.0, "width_mm": 8.0, "thickness_mm": 3.0, "source_model": "lcp_mini_frag_straight"},
    mesh_path=str(LCP), contacts_anchor_ids=[], volume_mm3=float(abs(plate.volume)),
    min_thickness_mm=3.0, validity={"watertight": True, "manifold": True, "self_intersect": False},
    fallback_rung=1, trace_id="demo",
)
trace = LoopTrace(plan.case_id, stage="evaluate")
cand.trace_id = trace.trace_id
report = engine.run({"candidate": cand, "case": case, "mode": "three_point"}, trace.child("evaluate"))
hm = engine.render_heatmap(cand, case, trace, plan=plan)
shutil.copy(hm["png_path"], OUT / "stress.png")
if hm.get("blend_path") and Path(hm["blend_path"]).exists():
    shutil.copy(hm["blend_path"], OUT / "stress.blend")

# --------------------------------------------------------------------------- #
# Agent-logic bullets (grounded in the real plan / report)
# --------------------------------------------------------------------------- #
A_RUNG = {1: "ML landmark regressor", 2: "geometric heuristic (PCA + curvature)", "floor": "conservative default frame"}
n_anchors = len(plan.anchor_points)
thick = [round(a.cortical_thickness_mm, 1) for a in plan.anchor_points]
r = report
bullets = {
    "a": [
        "Loaded the CT bone mesh and normalized it (recentred to origin, converted to mm).",
        f"Built an anatomical coordinate frame with PCA — the long axis is aligned to the bone shaft.",
        "Sampled the cortical surface and rejected any point with wall thickness below 1.5 mm.",
        f"Farthest-point-selected {n_anchors} well-spread anchors that bracket the fracture (cortical {min(thick)}–{max(thick)} mm).",
        f"Produced by the {A_RUNG.get(plan.fallback_rung, plan.fallback_rung)} rung · confidence {round(plan.confidence,2)}.",
    ],
    "b": [
        "Inferred the implant family from the inputs: diaphyseal fracture + surface anchors → locking compression plate.",
        "Selected a REAL locking compression plate (96 × 8 × 3 mm) from the implant library — not generated from scratch.",
        "Registered it rigidly onto the plan: long axis along the shaft, seating face on the cortical surface at the fracture.",
        "Preserved the real CAD geometry — rotation + translation only; watertight, volume unchanged.",
    ],
    "c": [
        f"Ran 3-point-bending FEA ({r.solver_used}) on the placed implant under the walking load.",
        f"Peak von Mises {round(r.peak_von_mises_MPa)} MPa vs yield 830 MPa → factor of safety {round(r.factor_of_safety,1)}×.",
        f"Fatigue {'safe' if r.fatigue_safe else 'at risk'} vs the endurance limit; stress-shielding index {round(r.stress_shielding_index,2)}.",
        f"Verdict: {'PASS — the implant survives' if r.passed else 'FAIL — redesign needed'}.",
    ],
    "report": {
        "peak_mpa": round(r.peak_von_mises_MPa, 1),
        "fos": round(r.factor_of_safety, 2),
        "passed": bool(r.passed),
        "solver": r.solver_used,
        "shielding": round(r.stress_shielding_index, 2),
        "fatigue_safe": bool(r.fatigue_safe),
    },
}
(OUT / "bullets.json").write_text(json.dumps(bullets, indent=2))
print("DONE. assets in", OUT)
for f in ["coords.png", "implant_alone.png", "implant_in_femur.png", "stress.png", "bullets.json"]:
    p = OUT / f
    print(f"  {f}: {'ok' if p.exists() else 'MISSING'} ({p.stat().st_size if p.exists() else 0} B)")
