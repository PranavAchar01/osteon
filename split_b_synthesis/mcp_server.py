"""blender-mcp server - Split B parametric synthesis tools (offline, trimesh-primary).

Geometry runs WITHOUT bpy/Blender so tests and the resilience demo work offline:
  - trimesh builds the plate box + screw cylinders, computes validity, casts contact
    rays, and exports the STL;
  - pymeshlab (CGAL) performs the watertight boolean hole-cut. trimesh's own boolean
    needs `manifold3d` or the Blender binary, neither of which is available offline here,
    so the screw holes are cut with pymeshlab's generate_boolean_difference instead.

Run offline:  python split_b_synthesis/mcp_server.py
Server name MUST match gateway/mcp-registry.yaml.
"""

import concurrent.futures
from pathlib import Path
from uuid import uuid4

import numpy as np
import trimesh
from mcp.server.fastmcp import FastMCP

from common.errors import ToolFailError
from common.mcp_base import osteon_tool
from common.settings import settings

mcp = FastMCP("blender-mcp")

# Screw-hole sizing (mm). Holes never breach the plate edge, which keeps the plate watertight.
_SCREW_RADIUS_FRAC_OF_WIDTH = 0.20
_HOLE_EDGE_MARGIN_MM = 0.5
_CYLINDER_SECTIONS = 24


def _trace_dir() -> Path:
    out = Path(settings.OSTEON_TRACE_DIR)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _plate_box(theta: dict) -> trimesh.Trimesh:
    """Solid plate: length (X) x width (Y) x thickness (Z), centred at the origin."""
    return trimesh.creation.box(
        extents=[theta["length_mm"], theta["width_mm"], theta["thickness_mm"]]
    )


def _screw_cylinders(theta: dict) -> list:
    """Through-thickness (Z) cylinders, in a centred run along the length (X) axis."""
    length = theta["length_mm"]
    width = theta["width_mm"]
    thickness = theta["thickness_mm"]
    n_screws = int(theta["n_screws"])
    spacing = theta["screw_spacing_mm"]

    radius = min(width * _SCREW_RADIUS_FRAC_OF_WIDTH, spacing * 0.35)
    radius = max(0.8, min(radius, width / 2 - 0.6, length / 2 - 0.6))

    cylinders = []
    run = spacing * (n_screws - 1)
    x0 = -run / 2.0
    for i in range(n_screws):
        x = x0 + i * spacing
        if abs(x) > length / 2 - radius - _HOLE_EDGE_MARGIN_MM:
            continue  # would breach the plate end; skip so the result stays watertight
        cyl = trimesh.creation.cylinder(
            radius=radius, height=thickness * 4.0, sections=_CYLINDER_SECTIONS
        )
        cyl.apply_translation([x, 0.0, 0.0])
        cylinders.append(cyl)
    return cylinders


def _boolean_difference(base: trimesh.Trimesh, cutters: list) -> trimesh.Trimesh:
    """Watertight CSG difference via pymeshlab/CGAL (offline; no bpy, no manifold3d)."""
    import pymeshlab  # lazy import: keeps module load light and fully bpy-free

    mesh_set = pymeshlab.MeshSet()
    mesh_set.add_mesh(
        pymeshlab.Mesh(
            vertex_matrix=np.asarray(base.vertices, dtype=np.float64),
            face_matrix=np.asarray(base.faces, dtype=np.int32),
        )
    )
    result_id = mesh_set.current_mesh_id()
    for cyl in cutters:
        mesh_set.add_mesh(
            pymeshlab.Mesh(
                vertex_matrix=np.asarray(cyl.vertices, dtype=np.float64),
                face_matrix=np.asarray(cyl.faces, dtype=np.int32),
            )
        )
        cutter_id = mesh_set.current_mesh_id()
        mesh_set.generate_boolean_difference(first_mesh=result_id, second_mesh=cutter_id)
        result_id = mesh_set.current_mesh_id()

    mesh_set.set_current_mesh(result_id)
    out = mesh_set.current_mesh()
    return trimesh.Trimesh(
        vertices=np.asarray(out.vertex_matrix(), dtype=np.float64),
        faces=np.asarray(out.face_matrix(), dtype=np.int64),
        process=True,
    )


def _validity(mesh: trimesh.Trimesh) -> dict:
    """{watertight, manifold, self_intersect}. self_intersect is a Day-1 solid-validity proxy."""
    try:
        watertight = bool(mesh.is_watertight)
        manifold = bool(mesh.is_winding_consistent)
        is_volume = bool(mesh.is_volume)
    except Exception:
        watertight = manifold = False
        is_volume = False
    # A clean closed positive-volume solid is treated as non-self-intersecting for Day 1;
    # Day 2 can swap in a true CGAL self-intersection test.
    return {"watertight": watertight, "manifold": manifold, "self_intersect": not is_volume}


def _export_stl(mesh: trimesh.Trimesh, prefix: str) -> str:
    path = _trace_dir() / f"{prefix}_{uuid4().hex[:8]}.stl"
    mesh.export(path)
    return str(path)


def _generate_mesh_impl(theta: dict) -> dict:
    base = _plate_box(theta)
    cutters = _screw_cylinders(theta)
    mesh = _boolean_difference(base, cutters) if cutters else base
    if mesh is None or len(mesh.faces) == 0:
        raise ToolFailError("generate_mesh produced an empty mesh")
    return {"mesh_path": _export_stl(mesh, "cand"), "validity": _validity(mesh)}


@osteon_tool(mcp)
def generate_mesh(theta: dict) -> dict:
    """Build the parametric plate (box minus screw cylinders) and export a watertight STL.

    Returns {mesh_path, validity}. Enforces OSTEON_DEADLINE_MS itself via concurrent.futures
    (osteon_tool normalizes errors + bounds size but does NOT add a timeout); raises
    ToolFailError on timeout or any geometry failure.
    """
    deadline_s = max(0.1, settings.OSTEON_DEADLINE_MS / 1000.0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_generate_mesh_impl, theta)
        try:
            return future.result(timeout=deadline_s)
        except concurrent.futures.TimeoutError:
            raise ToolFailError(
                f"generate_mesh exceeded OSTEON_DEADLINE_MS={settings.OSTEON_DEADLINE_MS}"
            )


@osteon_tool(mcp)
def repair_mesh(path: str) -> dict:
    """Fill holes, drop degenerate/duplicate faces, fix winding/normals; re-validate."""
    mesh = trimesh.load(path)
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)
    return {"mesh_path": _export_stl(mesh, "repaired"), "validity": _validity(mesh)}


@osteon_tool(mcp)
def check_contacts(mesh_path: str, anchors: list) -> dict:
    """Each anchor whose distance to the mesh surface is < 1 mm counts as a contact."""
    mesh = trimesh.load(mesh_path)
    if not anchors:
        return {"contacts": [], "all_touch": True}
    points = np.array(
        [[a["xyz"]["x"], a["xyz"]["y"], a["xyz"]["z"]] for a in anchors], dtype=np.float64
    )
    distance = np.abs(trimesh.proximity.signed_distance(mesh, points))
    contacts = [anchors[i]["id"] for i, dist in enumerate(distance) if dist < 1.0]
    return {"contacts": contacts, "all_touch": len(contacts) == len(anchors)}


if __name__ == "__main__":
    mcp.run()
