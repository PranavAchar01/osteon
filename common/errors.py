"""Shared error taxonomy. Map every external failure onto one of these at the boundary.

Guardrails and the fallback ladder branch on `.code`, never on the class name.
"""


class OsteonError(Exception):
    code: str = "E_UNKNOWN"


class RetryableError(OsteonError):
    code = "E_RETRYABLE"


class RateLimitError(RetryableError):
    code = "E_RATE_LIMIT"


class ProviderOutageError(RetryableError):  # 5xx
    code = "E_PROVIDER_OUTAGE"


class StageTimeoutError(RetryableError):
    code = "E_TIMEOUT"


class ToolFailError(RetryableError):
    code = "E_TOOL_FAIL"


class RejectedOutput(OsteonError):  # raised by a guardrail
    code = "E_BAD_OUTPUT"


class CascadeError(OsteonError):
    code = "E_CASCADE"


# Alias to match the name used in STANDARDIZATION.md
TimeoutError_ = StageTimeoutError


def from_openai(exc: Exception) -> OsteonError:
    """Translate an OpenAI/gateway exception into the shared taxonomy."""
    status = getattr(exc, "status_code", None)
    if status == 429:
        return RateLimitError(str(exc))
    if status is not None and 500 <= status < 600:
        return ProviderOutageError(str(exc))
    return ToolFailError(str(exc))
