"""Split A engine - Localization & Anchoring.

PHASE 0 STUB: rungs return a conservative valid PlacementPlan so the loop runs green.
Person A replaces _rung1 with the real ML + geometric methods (see SETUP.md).
The contract (CaseSpec -> PlacementPlan) and the ladder shape are FINAL.
"""
import argparse
import json
import os
from pathlib import Path

from common.contracts import AnchorPoint, CaseSpec, PlacementPlan, Vec3
from common.errors import RetryableError
import trimesh
from common.ladder import with_fallback

ROOT = Path(__file__).resolve().parent.parent


def _build(case, trace, rung, confidence):
    # This remains a stub for now, to be replaced with real logic
    return PlacementPlan(
        case_id=case.case_id,
        coordinate_frame={"origin": {"x": 0, "y": 0, "z": 0}, "basis": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        anchor_points=[
            AnchorPoint(id="a0", xyz=Vec3(x=-8, y=0, z=40), normal=Vec3(x=-1, y=0, z=0), cortical_thickness_mm=4.2),
            AnchorPoint(id="a1", xyz=Vec3(x=-8, y=0, z=-40), normal=Vec3(x=-1, y=0, z=0), cortical_thickness_mm=3.9),
        ],
        resection_planes=[],
        defect_region={"centroid": {"x": 0, "y": 0, "z": 0}, "obb": [], "volume_mm3": 320.0},
        fit_target_surface_path="fixtures/fit_target.stl",
        confidence=confidence,
        fallback_rung=rung,
        trace_id=trace.trace_id,
    )


import torch
from . import model as localization_model
from common import llm

import openai

def _call_llm_for_defect_spec(defect_description: str, trace: "LoopTrace"):
    prompt = f"""
    You are an orthopedic surgeon's assistant. Your task is to interpret a free-text description of a bone defect
    and convert it into a structured specification.

    Description: "{defect_description}"

    Based on this, provide a structured representation including the defect's center, its approximate dimensions (length, width, height in mm),
    and a classification (e.g., 'transverse fracture', 'comminuted fracture', 'wedge resection').
    Return ONLY a JSON object with keys 'centroid', 'dimensions', 'classification'.
    """
    messages = [{"role": "user", "content": prompt}]
    try:
        response = llm.call_llm(stage="localize-defect", messages=messages, trace=trace)
        return json.loads(response.choices[0].message.content)
    except (json.JSONDecodeError, IndexError, openai.AuthenticationError) as e:
        raise RetryableError(f"LLM call failed: {e}")

def _call_llm_for_sanity_check(plan: PlacementPlan, trace: "LoopTrace"):
    prompt = f"""
    You are an orthopedic surgeon's assistant performing a sanity check on a surgical plan.
    Given the following anchor points for an implant, are they anatomically plausible for the specified bone?

    Bone: Femur (for now, assume femur)
    Anchor Points (x, y, z in mm):
    {[(a.xyz.x, a.xyz.y, a.xyz.z) for a in plan.anchor_points]}
    
    Respond with ONLY a JSON object with a single key "plausible" (boolean) and "reason" (string).
    """
    messages = [{"role": "user", "content": prompt}]
    try:
        response = llm.call_llm(stage="localize-sanity-check", messages=messages, trace=trace)
        result = json.loads(response.choices[0].message.content)
        if not result.get("plausible"):
            raise RetryableError(f"LLM sanity check failed: {result.get('reason')}")
    except (json.JSONDecodeError, IndexError, openai.AuthenticationError) as e:
        raise RetryableError(f"LLM call failed: {e}")

def _rung1(case: CaseSpec, trace: "LoopTrace"):
    if os.getenv("OSTEON_FORCE_FAIL") == "localize":
        raise RetryableError("forced failure for demo")

    mesh = trimesh.load(case.bone_mesh_path)
    
    # 1. Get landmarks from ML model
    point_cloud = torch.tensor(mesh.vertices, dtype=torch.float32)
    landmarks = localization_model.get_landmarks(point_cloud) # (num_landmarks, 3)

    # 2. Get structured defect from LLM
    defect_spec = _call_llm_for_defect_spec(case.defect.get("description", ""), trace)

    # 3. Use landmarks and defect to create a plan (simplified logic)
    # For this dummy implementation, we'll use the landmarks as anchor points
    # and derive a simple coordinate frame.
    
    origin = np.mean(landmarks, axis=0)
    # A real implementation would use the landmarks to define anatomical axes
    vectors = mesh.principal_inertia_vectors
    z_axis, y_axis, _ = vectors
    x_axis = np.cross(y_axis, z_axis)
    basis = [x_axis.tolist(), y_axis.tolist(), z_axis.tolist()]
    coordinate_frame = {"origin": {"x": origin[0], "y": origin[1], "z": origin[2]}, "basis": basis}

    anchor_points = []
    for i, landmark in enumerate(landmarks):
        _, _, face_id = trimesh.proximity.closest_point(mesh, [landmark])
        normal = mesh.face_normals[face_id[0]]
        thickness = _measure_thickness(mesh, landmark, normal)
        anchor_points.append(AnchorPoint(
            id=f"a{i}",
            xyz=Vec3(x=landmark[0], y=landmark[1], z=landmark[2]),
            normal=Vec3(x=normal[0], y=normal[1], z=normal[2]),
            cortical_thickness_mm=thickness
        ))

    plan = PlacementPlan(
        case_id=case.case_id,
        coordinate_frame=coordinate_frame,
        anchor_points=anchor_points,
        resection_planes=[],
        defect_region=defect_spec,
        fit_target_surface_path=case.bone_mesh_path,
        confidence=0.91,
        fallback_rung=1,
        trace_id=trace.trace_id,
    )

    # 4. LLM Sanity Check
    _call_llm_for_sanity_check(plan, trace)
    
    return plan

import numpy as np

def _measure_thickness(mesh, point, normal):
    ray_origin = point
    ray_direction = -normal
    locations, _, _ = mesh.ray.intersects_location(
        ray_origins=[ray_origin],
        ray_directions=[ray_direction]
    )
    if len(locations) == 0:
        return 0.0
    return np.linalg.norm(locations[0] - ray_origin)

def farthest_point_sampling(points, num_points):
    if len(points) == 0:
        return []
    farthest_pts = np.zeros((num_points, 3))
    farthest_pts[0] = points[np.random.randint(len(points))]
    distances = np.linalg.norm(points - farthest_pts[0], axis=1)
    for i in range(1, num_points):
        farthest_pts[i] = points[np.argmax(distances)]
        distances = np.minimum(distances, np.linalg.norm(points - farthest_pts[i], axis=1))
    return farthest_pts

def _rung2(case: CaseSpec, trace: "LoopTrace"):
    """Geometric heuristic (PCA + curvature)."""
    mesh = trimesh.load(case.bone_mesh_path)
    mesh.apply_translation(-mesh.centroid)
    mesh.apply_scale(1000) # m -> mm

    # 1. Coordinate frame via PCA
    origin = mesh.centroid
    vectors = mesh.principal_inertia_vectors
    z_axis, y_axis, _ = vectors
    x_axis = np.cross(y_axis, z_axis)
    basis = [x_axis.tolist(), y_axis.tolist(), z_axis.tolist()]
    coordinate_frame = {"origin": {"x": origin[0], "y": origin[1], "z": origin[2]}, "basis": basis}

    # 2. Find anchor points
    samples, face_indices = trimesh.sample.sample_surface(mesh, 1000)
    normals = mesh.face_normals[face_indices]
    
    valid_points = []
    for point, normal in zip(samples, normals):
        thickness = _measure_thickness(mesh, point, normal)
        if thickness >= 1.5:
            valid_points.append(point)

    if not valid_points:
        # No suitable anchor points found, maybe return here or use a different strategy
        pass

    num_anchors = 4 # Or get from config
    anchor_coords = farthest_point_sampling(np.array(valid_points), num_anchors)

    anchor_points = []
    for i, point in enumerate(anchor_coords):
        # We need the normal at the anchor point, not the sampled point.
        # This is tricky without a direct mapping. For now, we'll reuse the sampled normal.
        # A better approach would be to find the closest point on the mesh to the FPS point.
        _, _, face_id = trimesh.proximity.closest_point(mesh, [point])
        normal = mesh.face_normals[face_id[0]]
        thickness = _measure_thickness(mesh, point, normal)

        anchor_points.append(AnchorPoint(
            id=f"a{i}",
            xyz=Vec3(x=point[0], y=point[1], z=point[2]),
            normal=Vec3(x=normal[0], y=normal[1], z=normal[2]),
            cortical_thickness_mm=thickness
        ))

    return PlacementPlan(
        case_id=case.case_id,
        coordinate_frame=coordinate_frame,
        anchor_points=anchor_points,
        resection_planes=[], # Placeholder
        defect_region={"centroid": {"x": 0, "y": 0, "z": 0}, "obb": [], "volume_mm3": 0.0}, # Placeholder
        fit_target_surface_path=case.bone_mesh_path,
        confidence=0.65,
        fallback_rung=2,
        trace_id=trace.trace_id,
    )

def _floor(case: CaseSpec, trace: "LoopTrace"):
    """Conservative geometric default (PCA frame)."""
    try:
        mesh = trimesh.load(case.bone_mesh_path)
        mesh.apply_translation(-mesh.centroid)
        mesh.apply_scale(1000) # m -> mm
        origin = mesh.centroid
        vectors = mesh.principal_inertia_vectors
        z_axis, y_axis, _ = vectors
        x_axis = np.cross(y_axis, z_axis)
        basis = [x_axis.tolist(), y_axis.tolist(), z_axis.tolist()]
        coordinate_frame = {"origin": {"x": origin[0], "y": origin[1], "z": origin[2]}, "basis": basis}
    except Exception:
        # If even PCA fails, return a default identity frame
        coordinate_frame = {"origin": {"x": 0, "y": 0, "z": 0}, "basis": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}

    return PlacementPlan(
        case_id=case.case_id,
        coordinate_frame=coordinate_frame,
        anchor_points=[],
        resection_planes=[],
        defect_region={"centroid": {"x": 0, "y": 0, "z": 0}, "obb": [], "volume_mm3": 0.0},
        fit_target_surface_path=case.bone_mesh_path,
        confidence=0.2,
        fallback_rung="floor",
        trace_id=trace.trace_id,
    )


run = with_fallback([_rung1, _rung2], _floor)


if __name__ == "__main__":
    from common.trace import LoopTrace
    import glob

    # To generate fixtures, we run rung2 directly.
    output_dir = ROOT / "split_a_localization" / "fixtures"
    output_dir.mkdir(exist_ok=True)
    
    case_files = glob.glob(str(ROOT / "fixtures" / "example_case*.json"))

    for case_file in case_files:
        case = CaseSpec(**json.load(open(case_file)))
        plan = _rung2(case, LoopTrace(case.case_id, stage="localize-fixture-gen"))
        
        output_path = output_dir / f"placement_plan_{case.case_id}.json"
        with open(output_path, "w") as f:
            f.write(plan.model_dump_json(indent=2))
        print(f"Generated fixture: {output_path}")

    # Original main execution
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--case", default=str(ROOT / "fixtures" / "example_case.json"))
    # args = ap.parse_args()
    # case = CaseSpec(**json.load(open(args.case)))
    # plan = run(case, LoopTrace(case.case_id, stage="localize"))
    # print(plan.model_dump_json(indent=2))
