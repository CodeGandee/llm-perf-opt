#!/usr/bin/env python3
"""
Nsight Compute kernel profiler with flexible kernel selection.

This script profiles CUDA kernels using NVIDIA Nsight Compute (ncu) with support
for both single kernel regex patterns and batch profiling from YAML configs.

CLI Usage
---------
python3 ncu-profile-kernels.py [options] -- <launch-command> [launch-args]

Required (one of):
  --kernel-config <yaml-path>  Path to YAML file with kernel names/regex patterns
  --kernel-regex <regex>       Single regex pattern for kernel matching

Options:
  --output-dir <dir>           Directory for profiling results (default: tmp/ncu-profile/<timestamp>)
  --topk <num>                 Profile only top K kernels from YAML (requires --kernel-config)
  --extra-sections <s1> <s2>   Additional ncu sections beyond defaults
  --num-kernel-call-skip <N>   Skip first N kernel invocations (default: 200)
  --num-kernel-call-profile <M> Profile M invocations after skipping (default: 1)
  --force-overwrite            Overwrite existing reports

Launch Command:
  --                           Separator before target application command
  <launch-command>             Command to launch target application
  [launch-args]                Arguments for the target application

Examples
--------
Profile single kernel:
>>> python3 ncu-profile-kernels.py \\
...   --kernel-regex 'internal::gemvx::kernel<.*\\(int\\)7.*>' \\
...   --output-dir tmp/gemvx-profile \\
...   -- python -m llm_perf_opt.runners.llm_profile_runner device=cuda:0

Profile multiple kernels from YAML:
>>> python3 ncu-profile-kernels.py \\
...   --kernel-config top-kernels.yaml \\
...   --extra-sections SourceCounters \\
...   -- python inference.py --model deepseek

Profile only top 3 kernels from YAML:
>>> python3 ncu-profile-kernels.py \\
...   --kernel-config top-kernels.yaml \\
...   --topk 3 \\
...   -- python inference.py

Notes
-----
- Based on NVIDIA best practices from nsight-compute CLI docs
- Uses --launch-skip/--launch-count for sampling (no replay needed)
- Uses --kernel-name-base demangled with regex patterns
- Default sections: SpeedOfLight, MemoryWorkloadAnalysis, Occupancy, SchedulerStats
- Exports section CSVs and details page automatically
- Writes provenance YAML for reproducibility
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import ruamel.yaml


# ============================================================================
# Color logging
# ============================================================================


class ColorLog:
    """Simple colored logging for terminal output."""

    def __init__(self):
        """Initialize color codes based on terminal support."""
        if sys.stdout.isatty() and os.environ.get("TERM", "dumb") != "dumb" and not os.environ.get("NO_COLOR"):
            self.RESET = "\033[0m"
            self.BOLD = "\033[1m"
            self.RED = "\033[31m"
            self.GREEN = "\033[32m"
            self.YELLOW = "\033[33m"
            self.BLUE = "\033[34m"
            self.CYAN = "\033[36m"
        else:
            self.RESET = self.BOLD = self.RED = self.GREEN = self.YELLOW = self.BLUE = self.CYAN = ""

    def info(self, msg: str) -> None:
        """Log info message in blue."""
        print(f"{self.BOLD}{self.BLUE}[ncu-profile]{self.RESET} {msg}")

    def ok(self, msg: str) -> None:
        """Log success message in green."""
        print(f"{self.BOLD}{self.GREEN}[ncu-profile]{self.RESET} {self.GREEN}{msg}{self.RESET}")

    def warn(self, msg: str) -> None:
        """Log warning message in yellow."""
        print(f"{self.BOLD}{self.YELLOW}[ncu-profile]{self.RESET} {self.YELLOW}{msg}{self.RESET}", file=sys.stderr)

    def error(self, msg: str) -> None:
        """Log error message in red."""
        print(f"{self.BOLD}{self.RED}[ncu-profile]{self.RESET} {self.RED}{msg}{self.RESET}", file=sys.stderr)

    def highlight(self, text: str) -> str:
        """Return text highlighted in cyan."""
        return f"{self.CYAN}{text}{self.RESET}"


log = ColorLog()


# ============================================================================
# Kernel config loading
# ============================================================================


def load_kernel_config(yaml_path: Path) -> list[dict[str, Any]]:
    """
    Load kernel configuration from YAML file.

    Parameters
    ----------
    yaml_path : Path
        Path to YAML file with kernel config

    Returns
    -------
    list[dict]
        List of kernel dicts with 'name', 'regex', 'description' keys

    Raises
    ------
    SystemExit
        If file not found or YAML parsing fails
    """
    if not yaml_path.exists():
        log.error(f"Kernel config file not found: {yaml_path}")
        sys.exit(2)

    try:
        yaml = ruamel.yaml.YAML(typ="safe")
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.load(f)

        if not isinstance(data, dict) or "kernels" not in data:
            log.error(f"Invalid kernel config: expected 'kernels' key in {yaml_path}")
            sys.exit(2)

        kernels = data["kernels"]
        if not isinstance(kernels, list) or len(kernels) == 0:
            log.error(f"Invalid kernel config: 'kernels' must be non-empty list in {yaml_path}")
            sys.exit(2)

        # Validate each kernel has required fields
        for i, kernel in enumerate(kernels):
            if not isinstance(kernel, dict):
                log.error(f"Kernel #{i + 1} is not a dict in {yaml_path}")
                sys.exit(2)
            if "name" not in kernel or "regex" not in kernel:
                log.error(f"Kernel #{i + 1} missing 'name' or 'regex' field in {yaml_path}")
                sys.exit(2)

        return kernels

    except Exception as e:
        log.error(f"Failed to parse kernel config YAML: {e}")
        sys.exit(2)


# ============================================================================
# NCU profiling
# ============================================================================


def run_ncu_profile(
    kernel_regex: str,
    output_base: Path,
    launch_cmd: list[str],
    sections: list[str],
    launch_skip: int,
    launch_count: int,
    force_overwrite: bool,
) -> bool:
    """
    Run ncu profiling for a single kernel pattern.

    Parameters
    ----------
    kernel_regex : str
        Regex pattern for kernel name matching
    output_base : Path
        Output file base path (without extension)
    launch_cmd : list[str]
        Launch command and arguments for target application
    sections : list[str]
        NCU sections to collect
    launch_skip : int
        Number of kernel invocations to skip
    launch_count : int
        Number of kernel invocations to profile
    force_overwrite : bool
        Whether to overwrite existing reports

    Returns
    -------
    bool
        True if profiling succeeded, False otherwise
    """
    # Build ncu command
    ncu_args = [
        "ncu",
        "--kernel-name-base",
        "demangled",
        "--kernel-name",
        f"regex:{kernel_regex}",
        "--launch-skip",
        str(launch_skip),
        "--launch-count",
        str(launch_count),
        "-o",
        str(output_base),
    ]

    # Add sections
    for section in sections:
        ncu_args.extend(["--section", section])

    if force_overwrite:
        ncu_args.append("--force-overwrite")

    # Add launch command
    ncu_args.extend(launch_cmd)

    # Log command
    log.info(f"Running: {' '.join(ncu_args)}")

    # Execute
    try:
        result = subprocess.run(ncu_args, check=False)
        return result.returncode == 0
    except FileNotFoundError:
        log.error("ncu command not found. Please ensure Nsight Compute is installed and in PATH.")
        return False
    except Exception as e:
        log.error(f"Failed to run ncu: {e}")
        return False


def export_ncu_csvs(report_path: Path, sections: list[str]) -> None:
    """
    Export section CSVs and details page from ncu report.

    Parameters
    ----------
    report_path : Path
        Path to .ncu-rep file
    sections : list[str]
        Sections to export as CSV
    """
    if not report_path.exists():
        log.warn(f"Report not found, skipping CSV export: {report_path}")
        return

    output_base = report_path.with_suffix("")

    # Export each section
    for section in sections:
        section_csv = output_base.with_suffix(f".section_{section}.csv")
        try:
            result = subprocess.run(
                ["ncu", "--csv", "--section", section, "--import", str(report_path)],
                stdout=open(section_csv, "w"),
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if result.returncode == 0:
                log.ok(f"Section CSV: {section_csv}")
            else:
                log.warn(f"Could not export section: {section}")
        except Exception as e:
            log.warn(f"Failed to export section {section}: {e}")

    # Try SpeedOfLight_RooflineChart (useful for visualization)
    roofline_csv = output_base.with_suffix(".section_SpeedOfLight_RooflineChart.csv")
    try:
        result = subprocess.run(
            ["ncu", "--csv", "--section", "SpeedOfLight_RooflineChart", "--import", str(report_path)],
            stdout=open(roofline_csv, "w"),
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode == 0:
            log.ok(f"Section CSV: {roofline_csv}")
    except Exception:
        pass  # Not critical

    # Export details page
    details_csv = output_base.with_suffix(".details.csv")
    try:
        result = subprocess.run(
            ["ncu", "--csv", "--page", "details", "--import", str(report_path)],
            stdout=open(details_csv, "w"),
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode == 0:
            log.ok(f"Details CSV: {details_csv}")
        else:
            log.warn("Could not export details CSV")
    except Exception as e:
        log.warn(f"Failed to export details CSV: {e}")


def write_provenance(
    output_dir: Path,
    kernel_source: str,
    kernel_regex: str | None,
    kernel_config: Path | None,
    launch_cmd: list[str],
    sections: list[str],
    launch_skip: int,
    launch_count: int,
    topk: int | None = None,
) -> None:
    """
    Write provenance YAML for reproducibility.

    Parameters
    ----------
    output_dir : Path
        Output directory for artifacts
    kernel_source : str
        Source of kernel selection ("single_regex" or "yaml_config")
    kernel_regex : str | None
        Single kernel regex if used
    kernel_config : Path | None
        Path to kernel config YAML if used
    launch_cmd : list[str]
        Launch command and arguments
    sections : list[str]
        NCU sections collected
    launch_skip : int
        Number of invocations skipped
    launch_count : int
        Number of invocations profiled
    topk : int | None
        Number of top kernels to profile (if limited)
    """
    prov_path = output_dir / "command.yaml"

    yaml = ruamel.yaml.YAML()
    yaml.default_flow_style = False

    data = {
        "timestamp": datetime.now().isoformat(),
        "script": Path(__file__).name,
        "version": "v1",
        "output_dir": str(output_dir),
        "kernel_source": kernel_source,
        "launch_skip": launch_skip,
        "launch_count": launch_count,
        "sections": sections,
        "launch_command": launch_cmd,
    }

    if kernel_regex:
        data["kernel_regex"] = kernel_regex
    if kernel_config:
        data["kernel_config"] = str(kernel_config)
    if topk is not None:
        data["topk"] = topk

    # Add tool versions
    try:
        ncu_version = subprocess.run(
            ["ncu", "--version"], capture_output=True, text=True, check=False
        ).stdout.strip()
        data["ncu_version"] = ncu_version
    except Exception:
        data["ncu_version"] = "unknown"

    try:
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        data["python_version"] = python_version
    except Exception:
        data["python_version"] = "unknown"

    with open(prov_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)

    log.ok(f"Provenance: {prov_path}")


# ============================================================================
# Main CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Profile CUDA kernels with Nsight Compute (ncu)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile single kernel
  %(prog)s --kernel-regex 'gemvx::kernel<.*\\(int\\)7.*>' \\
    --output-dir tmp/gemvx \\
    -- python -m llm_perf_opt.runners.llm_profile_runner device=cuda:0

  # Profile multiple kernels from YAML
  %(prog)s --kernel-config top-kernels.yaml \\
    --extra-sections SourceCounters \\
    -- python inference.py --model deepseek

  # Profile with custom sampling
  %(prog)s --kernel-regex 'flash_fwd.*' \\
    --num-kernel-call-skip 500 \\
    --num-kernel-call-profile 10 \\
    -- python benchmark.py

  # Profile only top 5 kernels from YAML
  %(prog)s --kernel-config top-kernels.yaml --topk 5 \\
    -- python inference.py

Default sections: SpeedOfLight, MemoryWorkloadAnalysis, Occupancy, SchedulerStats
        """,
    )

    # Kernel selection (mutually exclusive)
    kernel_group = parser.add_mutually_exclusive_group(required=True)
    kernel_group.add_argument(
        "--kernel-config",
        type=Path,
        metavar="<yaml-path>",
        help="Path to YAML file listing kernel names and regex patterns",
    )
    kernel_group.add_argument(
        "--kernel-regex", type=str, metavar="<regex>", help="Regex pattern of kernel names to profile"
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=Path,
        metavar="<dir>",
        help="Directory to save profiling results (default: tmp/ncu-profile/<timestamp>)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        metavar="<num>",
        help="Profile only top K kernels from YAML (requires --kernel-config)",
    )

    # NCU options
    parser.add_argument(
        "--extra-sections",
        nargs="+",
        metavar="<section>",
        default=[],
        help="Additional NCU sections beyond defaults",
    )
    parser.add_argument(
        "--num-kernel-call-skip",
        type=int,
        default=200,
        metavar="<N>",
        help="Number of initial kernel invocations to skip (default: 200)",
    )
    parser.add_argument(
        "--num-kernel-call-profile",
        type=int,
        default=1,
        metavar="<M>",
        help="Number of kernel invocations to profile after skipping (default: 1)",
    )
    parser.add_argument(
        "--force-overwrite", action="store_true", help="Overwrite existing report files"
    )

    # Launch command separator
    parser.add_argument(
        "launch_command",
        nargs="*",
        help="Launch command and arguments for target application (after --)",
    )

    args = parser.parse_args()

    # Validate launch command
    if not args.launch_command:
        parser.error("Launch command is required after -- separator")

    # Validate topk is only used with kernel-config
    if args.topk is not None:
        if args.kernel_regex:
            parser.error("--topk can only be used with --kernel-config, not with --kernel-regex")
        if args.topk <= 0:
            parser.error("--topk must be a positive integer")

    return args


def main() -> None:
    """Main entry point for ncu-profile-kernels.py."""
    args = parse_args()

    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        repo_root = Path(__file__).parent.parent.parent.parent
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = repo_root / "tmp" / "ncu-profile" / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {log.highlight(str(output_dir))}")

    # Default sections (aligned with v2 bash script)
    default_sections = ["SpeedOfLight", "MemoryWorkloadAnalysis", "Occupancy", "SchedulerStats"]
    sections = default_sections + args.extra_sections

    log.info(f"Sections: {', '.join(sections)}")
    log.info(f"Launch skip/count: {args.num_kernel_call_skip}/{args.num_kernel_call_profile}")

    # Get kernels to profile
    if args.kernel_regex:
        # Single kernel mode
        kernels = [{"name": "user_provided", "regex": args.kernel_regex, "description": "Single kernel profiling"}]
        kernel_source = "single_regex"
        kernel_config_path = None
        log.info(f"Kernel regex: {log.highlight(args.kernel_regex)}")
    else:
        # Batch mode from YAML
        kernel_config_path = args.kernel_config
        kernels = load_kernel_config(kernel_config_path)
        kernel_source = "yaml_config"
        log.info(f"Kernel config: {log.highlight(str(kernel_config_path))}")
        log.info(f"Loaded {len(kernels)} kernel(s) from config")

        # Apply topk filter if specified
        if args.topk is not None:
            original_count = len(kernels)
            kernels = kernels[:args.topk]
            log.info(f"Limiting to top {args.topk} kernel(s) (from {original_count} total)")

    # Write provenance
    write_provenance(
        output_dir=output_dir,
        kernel_source=kernel_source,
        kernel_regex=args.kernel_regex,
        kernel_config=kernel_config_path,
        launch_cmd=args.launch_command,
        sections=sections,
        launch_skip=args.num_kernel_call_skip,
        launch_count=args.num_kernel_call_profile,
        topk=args.topk if args.kernel_config else None,
    )

    # Profile each kernel
    success_count = 0
    failed_count = 0

    for i, kernel in enumerate(kernels):
        kernel_name = kernel.get("name", f"kernel_{i + 1}")
        kernel_regex = kernel["regex"]
        kernel_desc = kernel.get("description", "")

        log.info("")
        log.info(f"{'=' * 80}")
        log.info(f"Profiling kernel {i + 1}/{len(kernels)}")
        log.info(f"Name: {kernel_name}")
        if kernel_desc:
            log.info(f"Description: {kernel_desc}")
        log.info(f"Regex: {log.highlight(kernel_regex)}")
        log.info(f"{'=' * 80}")

        # Create output path for this kernel
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in kernel_name[:100])
        output_base = output_dir / f"kernel_{i + 1:03d}_{safe_name}"

        # Run profiling
        success = run_ncu_profile(
            kernel_regex=kernel_regex,
            output_base=output_base,
            launch_cmd=args.launch_command,
            sections=sections,
            launch_skip=args.num_kernel_call_skip,
            launch_count=args.num_kernel_call_profile,
            force_overwrite=args.force_overwrite,
        )

        if success:
            report_path = output_base.with_suffix(".ncu-rep")
            log.ok(f"Profiling complete: {report_path}")
            success_count += 1

            # Export CSVs
            log.info("Exporting section CSVs...")
            export_ncu_csvs(report_path, sections)
        else:
            log.error(f"Profiling failed for kernel: {kernel_name}")
            failed_count += 1

    # Summary
    log.info("")
    log.info(f"{'=' * 80}")
    log.ok(f"Profiling complete: {success_count} succeeded, {failed_count} failed")
    log.ok(f"Results saved to: {output_dir}")
    log.info(f"{'=' * 80}")

    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
