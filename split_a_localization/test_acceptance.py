
import os
import json
import shutil
import pytest
from pathlib import Path

from common.contracts import CaseSpec
from common.trace import LoopTrace
from split_a_localization.engine import run
from split_a_localization import mcp_server
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent

@pytest.fixture
def sample_case():
    case_path = ROOT / "fixtures" / "example_case.json"
    with open(case_path) as f:
        return CaseSpec(**json.load(f))

def test_split_a_rung2_success(sample_case):
    """Tests that Rung 2 produces a valid PlacementPlan."""
    # We can't easily force rung 2 without mocking rung 1, so we'll rely on the default behavior
    # where the dummy rung 1 passes through for now. Once rung 1 is real, this test will change.
    
    # To test rung 2 directly, we can temporarily alter the ladder
    from split_a_localization import engine
    original_ladder = engine.run
    engine.run = engine.with_fallback([engine._rung2], engine._floor)

    plan = engine.run(sample_case, LoopTrace(sample_case.case_id, stage="localize"))

    assert plan is not None
    assert plan.case_id == sample_case.case_id
    assert plan.fallback_rung == 2
    assert len(plan.anchor_points) > 0
    assert plan.confidence > 0.5

    # Restore original ladder
    engine.run = original_ladder

def test_split_a_fallback_to_rung2(sample_case):
    """Tests that the engine falls back from Rung 1 to Rung 2."""
    os.environ["OSTEON_FORCE_FAIL"] = "localize"
    
    plan = run(sample_case, LoopTrace(sample_case.case_id, stage="localize"))
    
    assert plan is not None
    assert plan.case_id == sample_case.case_id
    assert plan.fallback_rung == 2
    assert plan.confidence > 0.5
    
    del os.environ["OSTEON_FORCE_FAIL"]

from unittest.mock import patch

def test_split_a_fallback_to_floor(sample_case, monkeypatch):
    """Tests that the engine falls back to the floor if all rungs fail."""
    from split_a_localization import engine
    
    # Force both rung1 and rung2 to fail
    def fail_rung(case, trace):
        raise engine.RetryableError("Forced failure")

    monkeypatch.setattr(engine, '_rung1', fail_rung)
    monkeypatch.setattr(engine, '_rung2', fail_rung)

    # Re-create the run function with the patched rungs
    run_with_failures = engine.with_fallback([engine._rung1, engine._rung2], engine._floor)
    plan = run_with_failures(sample_case, LoopTrace(sample_case.case_id, stage="localize"))

    assert plan is not None
    assert plan.case_id == sample_case.case_id
    assert plan.fallback_rung == "floor"
    assert plan.confidence < 0.3

@patch('subprocess.run')
def test_render_markers(mock_run, sample_case):
    """Tests that the render_markers tool produces a PNG file."""
    from split_a_localization import engine
    plan = engine._rung2(sample_case, LoopTrace(sample_case.case_id, stage="localize-test"))
    
    # The plan needs a valid mesh path for the tool to work
    plan.fit_target_surface_path = str(ROOT / "fixtures" / "dummy_bone.stl")
    
    # Simulate a successful blender run
    mock_run.return_value = None

    result = mcp_server.render_markers(plan.model_dump())
    
    assert result is not None
    assert "png_path" in result
    png_path = result["png_path"]
    assert png_path.endswith(".png")
    
    # In a real scenario, we'd check if the file was created.
    # Since we are mocking subprocess, we can't do that.
    # So we'll just check that a path is returned.
    assert png_path is not None
    
    # Clean up the temp file if it was created
    if os.path.exists(png_path):
        os.unlink(png_path)


@pytest.mark.skipif(
    not (shutil.which(mcp_server._blender_bin()) or os.path.exists(mcp_server._blender_bin())),
    reason="blender not installed",
)
def test_render_markers_end_to_end(sample_case):
    """Real Blender render: produces a non-empty PNG. Runs only where blender exists."""
    from split_a_localization import engine
    plan = engine._rung2(sample_case, LoopTrace(sample_case.case_id, stage="localize-e2e"))
    plan.fit_target_surface_path = str(ROOT / "fixtures" / "dummy_bone.stl")

    result = mcp_server.render_markers(plan.model_dump())
    png_path = result["png_path"]
    try:
        assert os.path.exists(png_path), "blender did not write the PNG"
        assert os.path.getsize(png_path) > 0, "blender wrote an empty PNG"
    finally:
        if os.path.exists(png_path):
            os.unlink(png_path)


