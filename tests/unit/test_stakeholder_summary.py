from pathlib import Path

from llm_perf_opt.profiling.export import write_stakeholder_summary


def test_write_stakeholder_summary_creates_sections(tmp_path: Path) -> None:
    out = tmp_path / "stakeholder_summary.md"
    stats = {
        "aggregates": {
            "prefill_ms": {"mean": 100.0, "std": 1.0},
            "decode_ms": {"mean": 200.0, "std": 2.0},
            "tokens_per_s": {"mean": 10.0, "std": 0.5},
        },
        "mfu_model": 0.01,
        "mfu_stages": {"prefill": 0.02, "decode": 0.03, "vision": 0.0},
        "device": "cuda:0",
        "peak_tflops": 100.0,
    }
    write_stakeholder_summary(
        str(out),
        top_ops=[],
        stage_takeaways={"decode": "Decode dominates."},
        stats=stats,
        top_kernels=[],
    )
    text = out.read_text(encoding="utf-8")
    assert "Stakeholder Summary" in text
    assert "Environment" in text
    assert "Per-Stage Timings" in text or "Prefill" in text
    assert "Top Operators" in text
    assert "Top Kernels" in text

