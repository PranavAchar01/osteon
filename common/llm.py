"""The ONLY entry point to a model.

Calls go through the TrueFoundry AI Gateway (OpenAI-compatible) to AWS Bedrock. The gateway
URL + token live in .env (common/settings.py). Model-level fallback (primary -> fallback) is
applied here so a rate-limit/outage on the primary degrades to a second Bedrock model instead
of failing the call; the per-stage fallback ladder (common/ladder.py) is the next line of
defense above this.
"""
import hashlib
import json
import os

import openai
from openai import OpenAI

from common.settings import settings

client = OpenAI(api_key=settings.TFY_TOKEN or "MISSING", base_url=settings.TFY_GATEWAY_URL)

# Friendly aliases -> real Bedrock slugs on the gateway. Callers may pass either form.
MODEL_ALIASES = {
    "bedrock/claude-sonnet": "aws-bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1-0",
    "bedrock/claude-haiku": "aws-bedrock/us.anthropic.claude-haiku-4-5-20251001-v1-0",
    "bedrock/llama-70b": "aws-bedrock/us.meta.llama3-3-70b-instruct-v1-0",
    "bedrock/nova-lite": "aws-bedrock/us.amazon.nova-lite-v1-0",
}
DEFAULT_MODEL = "bedrock/claude-sonnet"
# Ordered model-level fallback: Claude (Bedrock) -> Llama (Bedrock, different provider family).
FALLBACK_CHAIN = ["bedrock/claude-sonnet", "bedrock/llama-70b"]


def _resolve(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def call_llm(*, stage: str, messages, model: str = DEFAULT_MODEL, **kw):
    """`stage` is mandatory: it tags the trace and selects the gateway routing rule.

    stage is one of: localize, synthesize, evaluate. Routes through the TrueFoundry gateway to
    Bedrock and falls back primary -> next model on any gateway/model error. Never adds retry
    logic beyond the ordered fallback (the stage ladder handles the rest).
    """
    trace = kw.pop("trace", None)
    # Build the attempt order: the requested model first, then the rest of the fallback chain.
    order, seen = [], set()
    for m in [model, *FALLBACK_CHAIN]:
        if m not in seen:
            order.append(m)
            seen.add(m)

    input_hash = hashlib.sha256(json.dumps(messages, default=str).encode()).hexdigest()
    last_exc = None
    force_fail = os.environ.get("OSTEON_FORCE_FAIL")
    for i, alias in enumerate(order):
        resolved = _resolve(alias)
        # SIMULATED, fully reversible failover (demo). When OSTEON_FORCE_FAIL == "gateway" we make
        # the PRIMARY model fail BEFORE the network call, so the AI Gateway reroutes to the next
        # model in FALLBACK_CHAIN (which still really runs). No credentials are touched and nothing
        # is mutated — clearing the env var restores normal routing. The failed attempt is tagged
        # simulated=True (+ reroute_to) on the trace so the dashboard can prove the failover.
        simulated = force_fail == "gateway" and alias == DEFAULT_MODEL
        try:
            if simulated:
                raise openai.OpenAIError(
                    f"simulated: {alias} Bedrock credentials revoked (forced gateway failover, demo)"
                )
            result = client.chat.completions.create(
                model=resolved,
                messages=messages,
                extra_headers={"x-tfy-metadata": json.dumps({"stage": stage})},
                **kw,
            )
            if trace:
                trace.emit(
                    span=f"llm:{stage}",
                    model=resolved,
                    alias=alias,
                    stage=stage,
                    fallback=i > 0,
                    input_hash=input_hash,
                    output_hash=hashlib.sha256(result.model_dump_json().encode()).hexdigest(),
                )
            return result
        except Exception as exc:  # rate limit / outage / bad response -> try the next model
            last_exc = exc
            if trace:
                span = {
                    "span": f"llm:{stage}",
                    "model": resolved,
                    "alias": alias,
                    "fallback": True,
                    "error": str(exc)[:160],
                }
                if simulated:
                    span["simulated"] = True
                    span["reason"] = "forced gateway failover (demo)"
                    span["reroute_to"] = order[i + 1] if i + 1 < len(order) else alias
                trace.emit(**span)
            continue
    raise last_exc
