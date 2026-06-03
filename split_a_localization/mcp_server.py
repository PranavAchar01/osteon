"""localization-mcp server. PHASE 0 STUB - Person A implements the tool bodies.

Run: python split_a_localization/mcp_server.py
Server name MUST match gateway/mcp-registry.yaml.
"""
from pathlib import Path

import trimesh
import numpy as np
from mcp.server.fastmcp import FastMCP

from common.mcp_base import osteon_tool

mcp = FastMCP("localization-mcp")


@osteon_tool(mcp)
def load_bone_mesh(path: str) -> dict:
    """Loads a mesh, centers it, and converts units to mm.

    Returns:
        A dictionary with the mesh's vertices and faces, serialized.
    """
    mesh = trimesh.load(path)
    mesh.apply_translation(-mesh.centroid)
    # Assuming input is in meters, convert to mm
    mesh.apply_scale(1000)

    return {
        "vertices": mesh.vertices.tolist(),
        "faces": mesh.faces.tolist(),
    }


@osteon_tool(mcp)
def measure_cortical_thickness(mesh_vertices: list, mesh_faces: list, xyz: list, normal: list) -> dict:
    """Measures cortical thickness by ray-casting.

    Args:
        mesh_vertices: The vertices of the mesh.
        mesh_faces: The faces of the mesh.
        xyz: The point on the surface.
        normal: The surface normal at the point.

    Returns:
        A dictionary containing the thickness in mm.
    """
    mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)
    ray_origin = np.array(xyz)
    ray_direction = -np.array(normal)

    # Perform the ray-mesh intersection
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=[ray_origin],
        ray_directions=[ray_direction]
    )

    if len(locations) == 0:
        return {"thickness_mm": 0.0}

    # The thickness is the distance to the first intersection point
    distance = np.linalg.norm(locations[0] - ray_origin)
    return {"thickness_mm": distance}


import subprocess
import tempfile
import json
import shutil

from common.errors import ToolFailError
import os


def _blender_bin() -> str:
    """Resolve the Blender executable: OSTEON_BLENDER, then PATH, then macOS default."""
    return (
        os.environ.get("OSTEON_BLENDER")
        or shutil.which("blender")
        or "/Applications/Blender.app/Contents/MacOS/Blender"
    )

@osteon_tool(mcp)
def render_markers(plan: dict) -> dict:
    """Renders a Blender scene and returns the path to the PNG.

    Args:
        plan: The PlacementPlan dictionary.

    Returns:
        A dictionary containing the path to the rendered PNG.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as plan_file:
        json.dump(plan, plan_file)
        plan_path = plan_file.name

    output_png_path = tempfile.mktemp(suffix=".png")
    bone_mesh_path = plan.get("fit_target_surface_path")
    
    blender_script_path = str(Path(__file__).resolve().parent / "blender_render.py")

    if not bone_mesh_path or not os.path.exists(bone_mesh_path):
        bone_mesh_path = ""

    cmd = [
        _blender_bin(), "--background", "--python", blender_script_path,
        "--", plan_path, bone_mesh_path, output_png_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        os.unlink(plan_path)
        raise ToolFailError(f"Blender rendering failed: {e}") from e
    
    os.unlink(plan_path)
    return {"png_path": output_png_path}


def open_in_blender(plan: dict, save_blend: str | None = None) -> dict:
    """Open the live 3D marker scene in the Blender GUI (interactive, not a flat PNG).

    Local dev/visualization helper — not an MCP tool. Launches Blender non-headless so
    you can orbit/inspect the bone, anchors, defect, and frame axes. If `save_blend` is
    given, also writes a reusable .blend file.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as plan_file:
        json.dump(plan, plan_file)
        plan_path = plan_file.name

    bone_mesh_path = plan.get("fit_target_surface_path")
    if not bone_mesh_path or not os.path.exists(bone_mesh_path):
        bone_mesh_path = ""

    blender_script_path = str(Path(__file__).resolve().parent / "blender_render.py")
    output_arg = save_blend if save_blend else "view"

    cmd = [
        _blender_bin(), "--python", blender_script_path,
        "--", plan_path, bone_mesh_path, output_arg,
    ]
    try:
        proc = subprocess.Popen(cmd)  # non-blocking: GUI stays open
    except FileNotFoundError as e:
        os.unlink(plan_path)
        raise ToolFailError(f"Blender launch failed: {e}") from e

    return {"pid": proc.pid, "blend_path": save_blend, "plan_path": plan_path}


if __name__ == "__main__":
    mcp.run()
