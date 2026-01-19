"""
Tests for Rust semantic indexing with CFG/DFG summaries.

Verifies that semantic embeddings include all 5 layers for Rust,
not just Python/TypeScript/JavaScript.

This test would have failed before the fix to _get_extractors() in semantic.py
which only supported Python/TS/JS, leaving 14 other languages without CFG/DFG
summaries in their semantic embeddings.

Run with:
    pytest tests/test_rust_semantic.py -v
"""

import json
import pytest
from pathlib import Path


# Sample Rust code with clear control flow and data flow
RUST_WITH_BRANCHES = """
pub fn classify(x: i32) -> &'static str {
    if x > 10 {
        "big"
    } else if x > 5 {
        "medium"
    } else {
        "small"
    }
}

pub fn process_data(input: &str) -> String {
    let trimmed = input.trim();
    let upper = trimmed.to_uppercase();
    let result = format!("{}!", upper);
    result
}

fn validate_token(token: &str) -> bool {
    if token.is_empty() {
        return false;
    }
    let len = token.len();
    len >= 8
}
"""


@pytest.fixture
def temp_rust_project(tmp_path):
    """Create a temporary Rust project."""
    filepath = tmp_path / "lib.rs"
    filepath.write_text(RUST_WITH_BRANCHES)
    return str(tmp_path)


class TestSemanticCFGSummary:
    """Test that CFG summaries are populated for Rust."""

    def test_cfg_summary_populated(self, temp_rust_project):
        """_get_cfg_summary should return complexity and blocks for Rust."""
        from tldr.semantic import _get_cfg_summary

        file_path = Path(temp_rust_project) / "lib.rs"
        summary = _get_cfg_summary(file_path, "classify", "rust")

        assert summary != "", f"CFG summary should not be empty, got: '{summary}'"
        assert "complexity:" in summary, f"Should include complexity, got: {summary}"
        assert "blocks:" in summary, f"Should include blocks, got: {summary}"

        # classify has 2 if branches, so complexity should be >= 3
        parts = summary.split(",")
        complexity_part = next((p for p in parts if "complexity:" in p), None)
        assert complexity_part is not None, f"No complexity in summary: {summary}"
        complexity = int(complexity_part.split(":")[1].strip())
        assert complexity >= 3, f"classify() should have complexity >= 3, got {complexity}"

    def test_cfg_summary_simple_function(self, temp_rust_project):
        """CFG summary for simple function should have low complexity."""
        from tldr.semantic import _get_cfg_summary

        file_path = Path(temp_rust_project) / "lib.rs"
        summary = _get_cfg_summary(file_path, "process_data", "rust")

        assert summary != "", "CFG summary should not be empty"
        assert "complexity:" in summary


class TestSemanticDFGSummary:
    """Test that DFG summaries are populated for Rust."""

    def test_dfg_summary_populated(self, temp_rust_project):
        """_get_dfg_summary should return vars and def-use chains for Rust."""
        from tldr.semantic import _get_dfg_summary

        file_path = Path(temp_rust_project) / "lib.rs"
        summary = _get_dfg_summary(file_path, "process_data", "rust")

        assert summary != "", f"DFG summary should not be empty, got: '{summary}'"
        assert "vars:" in summary, f"Should include vars count, got: {summary}"
        assert "def-use chains:" in summary, f"Should include def-use chains, got: {summary}"

        # process_data has variables: trimmed, upper, result
        parts = summary.split(",")
        vars_part = next((p for p in parts if "vars:" in p), None)
        assert vars_part is not None, f"No vars in summary: {summary}"
        var_count = int(vars_part.split(":")[1].strip())
        assert var_count >= 2, f"process_data() should have >= 2 vars, got {var_count}"


class TestSemanticIndexIntegration:
    """Test that semantic index includes CFG/DFG for Rust functions.

    This is the key test that would have failed before the fix.
    """

    def test_semantic_index_has_cfg_dfg(self, temp_rust_project):
        """Semantic index metadata should include cfg_summary and dfg_summary for Rust."""
        from tldr.semantic import build_semantic_index

        # Build the semantic index
        build_semantic_index(temp_rust_project, lang="rust", show_progress=False)

        # Check the metadata
        metadata_path = Path(temp_rust_project) / ".tldr" / "cache" / "semantic" / "metadata.json"
        assert metadata_path.exists(), "Semantic index should create metadata.json"

        with open(metadata_path) as f:
            metadata = json.load(f)

        units = metadata.get("units", [])
        assert len(units) > 0, "Should have indexed some functions"

        # Find the classify function
        classify_unit = None
        process_unit = None
        for unit in units:
            if unit.get("name") == "classify":
                classify_unit = unit
            if unit.get("name") == "process_data":
                process_unit = unit

        # Check classify has CFG summary (it has branches)
        assert classify_unit is not None, "classify function should be indexed"
        cfg = classify_unit.get("cfg_summary", "")
        assert cfg != "", f"classify should have cfg_summary, got: {classify_unit}"
        assert "complexity:" in cfg, "cfg_summary should include complexity"

        # Check process_data has DFG summary (it has data flow)
        assert process_unit is not None, "process_data function should be indexed"
        dfg = process_unit.get("dfg_summary", "")
        assert dfg != "", f"process_data should have dfg_summary, got: {process_unit}"
        assert "vars:" in dfg, "dfg_summary should include vars"


class TestSemanticLayerParity:
    """Test that Rust has parity with Python/TypeScript for semantic features."""

    def test_rust_matches_python_format(self, temp_rust_project, tmp_path):
        """Rust CFG/DFG summaries should use same format as Python."""
        from tldr.semantic import _get_cfg_summary

        # Get Python format
        py_file = tmp_path / "test.py"
        py_file.write_text("""
def example(x):
    if x > 10:
        return "big"
    return "small"
""")
        py_cfg = _get_cfg_summary(py_file, "example", "python")

        # Get Rust format
        rs_file = Path(temp_rust_project) / "lib.rs"
        rs_cfg = _get_cfg_summary(rs_file, "classify", "rust")

        # Both should have same format: "complexity:N, blocks:M"
        assert py_cfg.startswith("complexity:"), f"Python format: {py_cfg}"
        assert rs_cfg.startswith("complexity:"), f"Rust format: {rs_cfg}"
        assert ", blocks:" in py_cfg
        assert ", blocks:" in rs_cfg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
