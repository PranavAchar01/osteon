"""blender-mcp server. PHASE 0 STUB - Person B implements the tool bodies.

Run: blender --background --python split_b_synthesis/mcp_server.py
Server name MUST match gateway/mcp-registry.yaml.
"""
from mcp.server.fastmcp import FastMCP

from common.mcp_base import osteon_tool

mcp = FastMCP("blender-mcp")


@osteon_tool(mcp)
def generate_mesh(theta: dict) -> dict:
    raise NotImplementedError("TODO(B): build parametric implant mesh; enforce a timeout")


@osteon_tool(mcp)
def repair_mesh(path: str) -> dict:
    raise NotImplementedError("TODO(B): fill holes, remove self-intersections")


@osteon_tool(mcp)
def check_contacts(mesh_path: str, anchors: list) -> dict:
    raise NotImplementedError("TODO(B): verify mesh touches all anchor points")


if __name__ == "__main__":
    mcp.run()
