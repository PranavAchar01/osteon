"""The ONLY entry point to a model.

Routing, fallback, retries, and cooldown all live in gateway/routing.yaml - never here.
"""
import json

from openai import OpenAI

from common.settings import settings

client = OpenAI(api_key=settings.TFY_TOKEN or "MISSING", base_url=settings.TFY_GATEWAY_URL)


def call_llm(*, stage: str, messages, model: str = "bedrock/claude-sonnet", **kw):
    """`stage` is mandatory: it tags the trace and selects the gateway routing rule.

    stage is one of: localize, synthesize, evaluate.
    """
    return client.chat.completions.create(
        model=model,
        messages=messages,
        extra_headers={"x-tfy-metadata": json.dumps({"stage": stage})},
        **kw,
    )
