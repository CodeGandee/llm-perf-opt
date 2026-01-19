from __future__ import annotations

from pathlib import Path

from llm_perf_opt.utils.paths import wan2_1_analytic_dir, wan2_1_report_path, wan2_1_summary_path


def test_wan2_1_paths_are_absolute_and_stable() -> None:
    run_id = "wan2-1-unit-test"
    out_dir = Path(wan2_1_analytic_dir(run_id))
    report_path = Path(wan2_1_report_path(run_id))
    summary_path = Path(wan2_1_summary_path(run_id))

    assert out_dir.is_absolute()
    assert report_path.is_absolute()
    assert summary_path.is_absolute()

    assert str(out_dir).endswith(str(Path("tmp/profile-output") / run_id / "static_analysis" / "wan2_1"))
    assert report_path == out_dir / "report.json"
    assert summary_path == out_dir / "summary.md"
