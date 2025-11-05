#!/usr/bin/env bash
EXEC_PYTHON=${PYTHON:-python3}
exec "$EXEC_PYTHON" - "$@" <<'PY'
import argparse
import hashlib
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from llm_perf_opt.profiling.vendor.ncu import build_ncu_cmd


DEFAULT_SECTIONS = [
    "SpeedOfLight",
    "MemoryWorkloadAnalysis",
    "Occupancy",
    "SchedulerStats",
]


class ColorLog:
    def __init__(self) -> None:
        if sys.stdout.isatty() and os.environ.get("NO_COLOR") is None and os.environ.get("TERM", "dumb") != "dumb":
            self.reset = "\033[0m"
            self.bold = "\033[1m"
            self.blue = "\033[34m"
            self.green = "\033[32m"
            self.yellow = "\033[33m"
            self.red = "\033[31m"
            self.cyan = "\033[36m"
        else:
            self.reset = self.bold = self.blue = self.green = self.yellow = self.red = self.cyan = ""

    def _fmt(self, color: str, msg: str) -> str:
        return f"{self.bold}{color}[ncu-profile]{self.reset} {color}{msg}{self.reset}"

    def info(self, msg: str) -> None:
        print(self._fmt(self.blue, msg))

    def ok(self, msg: str) -> None:
        print(self._fmt(self.green, msg))

    def warn(self, msg: str) -> None:
        print(self._fmt(self.yellow, msg), file=sys.stderr)

    def error(self, msg: str) -> None:
        print(self._fmt(self.red, msg), file=sys.stderr)

    def highlight(self, text: str) -> str:
        return f"{self.cyan}{text}{self.reset}"


log = ColorLog()


def load_kernel_config(path: Path, topk: int | None) -> list[dict]:
    try:
        try:
            import ruamel.yaml

            yaml = ruamel.yaml.YAML(typ="safe")  # type: ignore[assignment]
            with path.open("r", encoding="utf-8") as f:
                data = yaml.load(f)
        except ModuleNotFoundError:
            import yaml  # type: ignore

            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
    except Exception as exc:  # pragma: no cover - best effort fallback
        log.error(f"Failed to load kernel config {path}: {exc}")
        sys.exit(2)

    if not isinstance(data, dict) or "kernels" not in data:
        log.error(f"Invalid kernel config: expected 'kernels' list in {path}")
        sys.exit(2)

    kernels = data["kernels"]
    if not isinstance(kernels, list) or not kernels:
        log.error(f"Invalid kernel config: 'kernels' must be a non-empty list in {path}")
        sys.exit(2)

    if topk is not None:
        kernels = kernels[:topk]
        log.info(f"Limiting to top {topk} kernel(s)")

    results: list[dict] = []
    for idx, kernel in enumerate(kernels, start=1):
        if not isinstance(kernel, dict):
            log.error(f"Kernel #{idx} is not a mapping in {path}")
            sys.exit(2)
        if "regex" not in kernel:
            log.error(f"Kernel #{idx} missing 'regex' field in {path}")
            sys.exit(2)
        results.append(kernel)
    return results


def compute_rank_width(count: int) -> int:
    return max(4, len(str(count)))


def md5_hex(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()  # noqa: S324


def export_csv(report_path: Path, sections: list[str]) -> None:
    if not report_path.exists():
        log.warn(f"Report not found, skipping CSV export: {report_path}")
        return

    base = report_path.with_suffix("")
    for section in sections:
        target = base.with_suffix(f".section_{section}.csv")
        result = subprocess.run(
            ["ncu", "--csv", "--section", section, "--import", str(report_path)],
            stdout=target.open("w", encoding="utf-8"),
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode == 0:
            log.ok(f"Section CSV: {target}")
        else:
            log.warn(f"Could not export section: {section}")

    roofline = base.with_suffix(".section_SpeedOfLight_RooflineChart.csv")
    subprocess.run(
        ["ncu", "--csv", "--section", "SpeedOfLight_RooflineChart", "--import", str(report_path)],
        stdout=roofline.open("w", encoding="utf-8"),
        stderr=subprocess.DEVNULL,
        check=False,
    )

    details = base.with_suffix(".details.csv")
    result = subprocess.run(
        ["ncu", "--csv", "--page", "details", "--import", str(report_path)],
        stdout=details.open("w", encoding="utf-8"),
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode == 0:
        log.ok(f"Details CSV: {details}")
    else:
        log.warn("Could not export details CSV")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile CUDA kernels via vendor builder (runner-aligned)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""
        "Example:\n"
        "  ncu-profile-kernels-via-runner.sh --kernel-config top-10.yaml --topk 3 \\\n+          -- python -m llm_perf_opt.runners.llm_profile_runner device=cuda:0",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--kernel-config", type=Path, metavar="<yaml-path>")
    group.add_argument("--kernel-regex", type=str, metavar="<regex>")
    parser.add_argument("--output-dir", type=Path, metavar="<dir>")
    parser.add_argument("--topk", type=int, metavar="<num>")
    parser.add_argument("--extra-sections", nargs="+", default=[], metavar="<section>")
    parser.add_argument("--num-kernel-call-skip", type=int, default=200, metavar="<N>")
    parser.add_argument("--num-kernel-call-profile", type=int, default=1, metavar="<M>")
    parser.add_argument("--force-overwrite", action="store_true")
    parser.add_argument("launch_command", nargs=argparse.REMAINDER, help="-- <cmd> [args]")
    args = parser.parse_args()

    if not args.launch_command:
        parser.error("Launch command is required after -- separator")
    if args.launch_command[0] != "--":
        parser.error("Launch command must be preceded by -- separator")
    args.launch_command = args.launch_command[1:]
    if not args.launch_command:
        parser.error("Launch command is required after -- separator")

    if args.topk is not None and args.kernel_regex:
        parser.error("--topk can only be used with --kernel-config, not with --kernel-regex")

    if args.topk is not None and args.topk <= 0:
        parser.error("--topk must be a positive integer")

    return args


def main() -> int:
    args = parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path("tmp/ncu-profile") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {log.highlight(str(output_dir))}")

    sections = DEFAULT_SECTIONS + list(args.extra_sections)
    log.info(f"Sections: {', '.join(sections)}")
    log.info(f"Launch skip/count: {args.num_kernel_call_skip}/{args.num_kernel_call_profile}")

    if args.kernel_config:
        kernels = load_kernel_config(args.kernel_config, args.topk)
        log.info(f"Kernel config: {log.highlight(str(args.kernel_config))}")
    else:
        kernels = [
            {
                "name": "user_provided",
                "regex": args.kernel_regex,
                "description": "Single kernel profiling",
            }
        ]

    total = len(kernels)
    rank_width = compute_rank_width(total)
    success = 0
    failed = 0

    for idx, kernel in enumerate(kernels, start=1):
        name = str(kernel.get("name", f"kernel_{idx}"))
        regex = str(kernel["regex"])
        desc = kernel.get("description", "")

        log.info("" )
        log.info("=" * 80)
        log.info(f"Profiling kernel {idx}/{total}")
        log.info(f"Name: {name}")
        if desc:
            log.info(f"Description: {desc}")
        log.info(f"Regex: {log.highlight(regex)}")
        log.info("=" * 80)

        kernel_dir = output_dir / f"kernel_{idx:0{rank_width}d}_{md5_hex(name)}"
        kernel_dir.mkdir(parents=True, exist_ok=True)
        output_base = kernel_dir / "ncu"

        cmd = build_ncu_cmd(
            out_base=output_base,
            work_argv=args.launch_command,
            nvtx_expr=None,
            kernel_regex=f"regex:{regex}",
            csv_log=None,
            use_nvtx=False,
            set_name="roofline",
            metrics=None,
            sections=sections,
            target_processes="all",
            force_overwrite=args.force_overwrite,
            kernel_name_base="demangled",
        )

        insert_at = len(cmd) - len(args.launch_command)
        cmd[insert_at:insert_at] = [
            "--launch-skip",
            str(args.num_kernel_call_skip),
            "--launch-count",
            str(args.num_kernel_call_profile),
        ]

        log.info("Running: " + " ".join(cmd))
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            report_path = output_base.with_suffix(".ncu-rep")
            log.ok(f"Profiling complete: {report_path}")
            success += 1
            export_csv(report_path, sections)
        else:
            log.error(f"Profiling failed for kernel: {name}")
            failed += 1

    log.info("" )
    log.info("=" * 80)
    log.ok(f"Profiling complete: {success} succeeded, {failed} failed")
    log.ok(f"Results saved to: {output_dir}")
    log.info("=" * 80)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
PY
