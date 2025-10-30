from pathlib import Path

from llm_perf_opt.data.models import KernelRecord
from llm_perf_opt.profiling.export import top_n_kernels, write_kernel_markdown


def _k(name: str, total: float, calls: int, mean: float | None = None, device: str = "cuda:0") -> KernelRecord:
    return KernelRecord(
        kernel_name=name,
        device=device,
        total_ms=float(total),
        calls=int(calls),
        mean_ms=float(mean) if mean is not None else 0.0,
    )


def test_top_n_kernels_sort_and_limit():
    records = [
        _k("A", 10.0, 2, 5.0),
        _k("B", 30.0, 3, 10.0),
        _k("C", 20.0, 4, 5.0),
    ]
    top2 = top_n_kernels(records, n=2)
    assert [r.kernel_name for r in top2] == ["B", "C"]
    assert len(top2) == 2


def test_write_kernel_markdown_creates_file(tmp_path: Path):
    records = [
        _k("attn", 12.0, 3, None),  # mean should be derived as 4.0
        _k("gemm", 8.0, 2, 4.0),
    ]
    out_md = tmp_path / "kernels.md"
    write_kernel_markdown(records, str(out_md), top_k=2)
    # mdutils appends .md; our helper strips suffix if present
    assert out_md.exists(), f"Expected file at {out_md}"
    text = out_md.read_text(encoding="utf-8")
    assert "Kernel Summary" in text
    assert "attn" in text and "gemm" in text
    # Derived mean for attn should appear with 6 decimals
    assert "4.000000" in text
