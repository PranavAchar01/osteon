"""Standard MCP tool wrapper so all three split servers behave identically.

Wraps each tool with: any error -> ToolFailError, plus a result-size bound.
Long-running tools (generate_mesh, run_calculix) must enforce their own timeout.
"""
import functools
import inspect
import json

from common.errors import ToolFailError


def osteon_tool(mcp, *, max_bytes: int = 5_000_000):
    """Register `fn` as an MCP tool on `mcp`, normalized for resilience."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                out = fn(*args, **kwargs)
            except ToolFailError:
                raise
            except Exception as exc:  # normalize everything else
                raise ToolFailError(str(exc))
            if len(json.dumps(out, default=str)) > max_bytes:
                raise ToolFailError("result exceeds size bound")
            return out

        wrapper.__signature__ = inspect.signature(fn)  # preserve schema for FastMCP
        return mcp.tool()(wrapper)

    return deco
