"""localization-mcp server. PHASE 0 STUB - Person A implements the tool bodies.

Run: python split_a_localization/mcp_server.py
Server name MUST match gateway/mcp-registry.yaml.
"""
from mcp.server.fastmcp import FastMCP

from common.mcp_base import osteon_tool

mcp = FastMCP("localization-mcp")


@osteon_tool(mcp)
def load_bone_mesh(path: str) -> dict:
    raise NotImplementedError("TODO(A): load + normalize STL to mm")


@osteon_tool(mcp)
def measure_cortical_thickness(xyz: dict) -> dict:
    raise NotImplementedError("TODO(A): ray-cast inner/outer wall distance at a point")


@osteon_tool(mcp)
def render_markers(plan: dict) -> dict:
    raise NotImplementedError("TODO(A): Blender scene PNG of anchors/defect/frame")


if __name__ == "__main__":
    mcp.run()
