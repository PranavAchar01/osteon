"""The standardized 3-rung fallback ladder. Identical shape in all three splits."""
from common.errors import RejectedOutput, RetryableError


def with_fallback(rungs, floor):
    """Build a resilient stage runner.

    rungs: list of callables ordered best -> worst, each `(inp, trace) -> output`.
    floor: deterministic callable `(inp, trace) -> output` that cannot fail.

    Returns `run(inp, trace)` which NEVER raises: if every rung fails, the floor
    returns a valid low-confidence result, so the design loop cannot break.
    """

    def run(inp, trace):
        for i, rung in enumerate(rungs, start=1):
            try:
                out = rung(inp, trace)
                trace.emit(rung=i, fallback=i > 1)
                return out
            except (RetryableError, RejectedOutput) as exc:
                trace.emit(rung=i, error=exc.code, fallback=True)
                continue
        out = floor(inp, trace)
        trace.emit(rung="floor", fallback=True, confidence="low")
        return out

    return run
