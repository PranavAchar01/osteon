"""fea-mcp server. PHASE 0 STUB - Person C implements the tool bodies.

Run: python split_c_evaluation/mcp_server.py
Server name MUST match gateway/mcp-registry.yaml.
"""
from mcp.server.fastmcp import FastMCP

from common.mcp_base import osteon_tool

mcp = FastMCP("fea-mcp")


@osteon_tool(mcp)
def meshing_to_fe(mesh_path: str) -> dict:
    raise NotImplementedError("TODO(C): tetrahedralize + materials + BCs -> .inp")


@osteon_tool(mcp)
def run_calculix(inp_path: str) -> dict:
    raise NotImplementedError("TODO(C): run ccx; enforce a timeout + result-size bound")


@osteon_tool(mcp)
def compute_shielding_index(intact_path: str, implanted_path: str) -> dict:
    raise NotImplementedError("TODO(C): bone strain-energy ratio intact vs implanted")


if __name__ == "__main__":
    mcp.run()
