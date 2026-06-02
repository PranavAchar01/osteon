"""The frozen integration interface. Field names, types, and units are fixed.

Units: mm (length), N (force), MPa (stress), degrees (angle). Add fields only by PR
with all three owners approving. See STANDARDIZATION.md section 3.
"""
from typing import Literal

from pydantic import BaseModel


class Vec3(BaseModel):
    x: float
    y: float
    z: float  # mm


class CaseSpec(BaseModel):  # SYSTEM INPUT
    case_id: str
    bone_mesh_path: str  # watertight STL, mm
    bone_material: dict  # {E_cortical_MPa, E_trabecular_MPa, density}
    defect: dict  # {type: fracture|resection|void, region, severity, description}
    load_profile: list  # [{name, force_vector_N: Vec3, application_region, cycles}]
    implant_material: dict  # {name, E_MPa, yield_MPa, endurance_limit_MPa}
    constraints: dict  # {min_thickness_mm, max_volume_mm3, process: additive|subtractive}


class AnchorPoint(BaseModel):
    id: str
    xyz: Vec3
    normal: Vec3
    cortical_thickness_mm: float


class PlacementPlan(BaseModel):  # A -> B
    case_id: str
    coordinate_frame: dict  # {origin: Vec3, basis: 3x3 list}
    anchor_points: list[AnchorPoint]
    resection_planes: list  # [{point: Vec3, normal: Vec3}]
    defect_region: dict  # {centroid: Vec3, obb, volume_mm3}
    fit_target_surface_path: str  # submesh STL the implant must conform to
    confidence: float  # 0..1
    fallback_rung: int | Literal["floor"]
    trace_id: str


class ImplantCandidate(BaseModel):  # B -> C
    case_id: str
    candidate_id: str
    iteration: int
    parameter_vector: dict  # named theta
    mesh_path: str  # watertight STL
    contacts_anchor_ids: list[str]
    volume_mm3: float
    min_thickness_mm: float
    validity: dict  # {watertight: bool, manifold: bool, self_intersect: bool}
    fallback_rung: int | Literal["floor"]
    trace_id: str


class StressReport(BaseModel):  # C -> B
    case_id: str
    candidate_id: str
    iteration: int
    peak_von_mises_MPa: float
    peak_location: Vec3
    factor_of_safety: float
    fatigue_safe: bool
    stress_shielding_index: float  # 0 = full shielding, 1 = natural bone
    displacement_max_mm: float
    passed: bool
    solver_used: Literal["full_fea", "reduced_surrogate", "analytic_fallback"]
    confidence: float
    fallback_rung: int | Literal["floor"]
    trace_id: str
