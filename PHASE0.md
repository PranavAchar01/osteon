# PHASE 0 - Shared Foundation (do this together, once)

Phase 0 is **complete and frozen** in this repo. Every owner runs the steps below and must
see the same green result before touching their split. That is what "everyone on the same
page" means here: identical environment, identical frozen contracts, and a loop that already
runs end-to-end on stubs.

## What Phase 0 ships (already in the repo)
- `common/` - settings, the one gateway client, frozen contracts, error taxonomy, fallback ladder, trace, MCP base.
- `gateway/` - `routing.yaml`, `guardrails.yaml`, `mcp-registry.yaml` (config-as-code).
- `orchestrator.py` - chains A -> B <-> C; runs green on Phase 0 stubs.
- `split_*/engine.py` + `mcp_server.py` - **stubs** that return valid contract objects (you replace the rungs).
- `fixtures/` - one golden JSON for every contract (the canonical data shapes).
- `tests/smoke_test.py` - proves the foundation with no credentials.

## Setup (each person, identical)
1. **Accounts:** TrueFoundry account; add an AWS Bedrock provider account; create a Virtual Account Token; request Bedrock access for Claude / Llama 3 70B / Mistral Large; alias them in the gateway as `bedrock/claude-sonnet`, `bedrock/llama-70b`, `bedrock/mistral-large`.
2. `git clone https://github.com/PranavAchar01/osteon.git && cd osteon`
3. `uv venv && source .venv/bin/activate`
4. `uv pip install -e ".[dev]"`
5. `cp .env.example .env` and fill `TFY_TOKEN`.

## Verify (must pass for everyone)
```bash
pytest -q                  # contracts + ladder->floor + trace + orchestrator all green
python orchestrator.py     # prints passed=true, FoS, trace_id; writes traces/<id>.jsonl
OSTEON_FORCE_FAIL=evaluate python orchestrator.py   # C rung1 fails -> floor -> loop still completes
```
The three commands above need **no credentials**. To confirm the live Bedrock-via-gateway path:
```bash
python -c "from common.llm import call_llm; print(call_llm(stage='smoke', messages=[{'role':'user','content':'ping'}]).choices[0].message.content)"
```

## Apply shared gateway config (once, by one person)
Confirm the exact YAML field names against your deployed gateway version first, then:
```bash
tfy apply -f gateway/routing.yaml
tfy apply -f gateway/guardrails.yaml
tfy apply -f gateway/mcp-registry.yaml
```
Then create the guardrail integrations named in `guardrails.yaml` (e.g. `implant/theta-bounds-check`) in the TrueFoundry registry/UI so the references resolve.

## Definition of "Phase 0 done"
- [ ] Everyone: `pytest -q` green.
- [ ] Everyone: `python orchestrator.py` prints `"passed": true`.
- [ ] Everyone: the forced-failure run still completes (proves the floor).
- [ ] One person: gateway configs applied; guardrail integrations created.
- [ ] `common/` and `gateway/*.yaml` frozen (changes require all three approving).

**Phase 0 frozen -> each owner goes to their split's `SETUP.md`.**
