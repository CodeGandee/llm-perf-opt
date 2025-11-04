#!/usr/bin/env python3
"""
Extract top-K kernels from Nsight Systems CSV and generate YAML for ncu profiling.

This module provides functionality to parse NVIDIA Nsight Systems kernel summary
CSV files and extract the top-K kernels by performance metrics. It generates
YAML configuration files compatible with batch ncu profiling workflows.

Functions
---------
escape_kernel_name_to_regex : str -> str
    Convert kernel name to exact-match regex pattern using re.escape()
main : None -> None
    Main CLI entry point with argument parsing and execution

Examples
--------
Extract all kernels sorted by time percentage (default):

>>> python extract-top-kernels.py summary_cuda_gpu_kern_sum.csv -o all-kernels.yaml

Extract top-3 kernels sorted by time percentage:

>>> python extract-top-kernels.py summary_cuda_gpu_kern_sum.csv -o top-3.yaml --topk 3

Extract top-5 kernels sorted by total time:

>>> python extract-top-kernels.py summary.csv -o top-5.yaml --topk 5 --sort-by time_ns

Notes
-----
- Generates exact-match regex patterns using re.escape() for safety
- Users may need to manually edit patterns to match kernel families
- Output YAML is compatible with ncu-profile-top-kernels.v2.sh
- Based on NVIDIA best practices from nsight-compute CLI documentation
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from io import StringIO
from typing import Dict, List, Any

import ruamel.yaml


def escape_kernel_name_to_regex(name: str) -> str:
    """
    Convert a kernel name to a regex pattern that matches it exactly.

    Uses re.escape() to handle special regex characters, then wraps with ^...$
    for exact matching. This follows NVIDIA best practices for kernel name
    matching in nsight-compute CLI workflows.

    Parameters
    ----------
    name : str
        Demangled kernel name from nsys CSV output

    Returns
    -------
    str
        Escaped regex pattern for exact matching (without 'regex:' prefix)

    Examples
    --------
    >>> escape_kernel_name_to_regex("kernel<int>(T)")
    '^kernel<int>\\(T\\)$'

    Notes
    -----
    Based on NVIDIA best practices from nsight-compute CLI docs:
    https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
    """
    # Escape all regex special characters so they match literally
    escaped = re.escape(name)
    # Wrap with anchors for exact match
    return f"^{escaped}$"


def main() -> None:
    """
    Main CLI entry point for extracting top-K kernels from nsys CSV.

    Parses command line arguments, reads the CSV file, sorts kernels by
    specified metric, and generates a YAML configuration file for batch
    ncu profiling workflows.

    Raises
    ------
    SystemExit
        Exit code 1 if CSV file not found or parsing errors occur
        Exit code 2 if required columns are missing from CSV

    Examples
    --------
    Extract top-3 kernels (default behavior):

    >>> main()  # when called with args: summary.csv -o top-3.yaml

    Notes
    -----
    This function handles all CLI argument parsing, file I/O, CSV processing,
    and YAML generation. It provides comprehensive error handling and user
    feedback throughout the extraction process.
    """
    parser = argparse.ArgumentParser(
        description="Extract top-K kernels from nsys CSV and create YAML config for ncu profiling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all kernels and display on console (default)
  %(prog)s summary_cuda_gpu_kern_sum.csv

  # Extract all kernels to file
  %(prog)s summary_cuda_gpu_kern_sum.csv -o all-kernels.yaml

  # Extract top-3 kernels to file
  %(prog)s summary_cuda_gpu_kern_sum.csv -o top-3-kernels.yaml --topk 3

  # Extract top-5 kernels sorted by total time (ns)
  %(prog)s summary.csv -o top-5.yaml --topk 5 --sort-by time_ns

The generated YAML can be used with ncu-profile-top-kernels.v2.sh for batch profiling.
        """,
    )
    parser.add_argument("csv_filepath", help="Path to nsys summary CSV (e.g., summary_cuda_gpu_kern_sum.csv)")
    parser.add_argument("-o", "--output", help="Output YAML file path (default: print to console)")
    parser.add_argument("--topk", type=int, default=None, help="Number of top kernels to extract (default: all kernels)")
    parser.add_argument(
        "--sort-by",
        choices=["time_ns", "time_pct"],
        default="time_pct",
        help="Sort kernels by 'time_ns' (Total Time) or 'time_pct' (Time %%) (default: time_pct)",
    )

    args = parser.parse_args()

    # Validate input
    csv_path = Path(args.csv_filepath)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # Read CSV
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    if not rows:
        print("Error: CSV file is empty or has no data rows", file=sys.stderr)
        sys.exit(1)

    # Detect column names (case-insensitive partial match)
    columns = rows[0].keys()

    # Find Name column
    name_col = None
    for col in columns:
        if "name" in col.lower():
            name_col = col
            break
    if not name_col:
        print(f"Error: Could not find 'Name' column in CSV. Available columns: {list(columns)}", file=sys.stderr)
        sys.exit(1)

    # Find sorting column
    if args.sort_by == "time_ns":
        sort_col = None
        for col in columns:
            if "total" in col.lower() and "time" in col.lower() and "ns" in col.lower():
                sort_col = col
                break
        if not sort_col:
            print(f"Error: Could not find 'Total Time (ns)' column. Available: {list(columns)}", file=sys.stderr)
            sys.exit(1)
    else:  # time_pct
        sort_col = None
        for col in columns:
            if "time" in col.lower() and "%" in col:
                sort_col = col
                break
        if not sort_col:
            print(f"Error: Could not find 'Time (%)' column. Available: {list(columns)}", file=sys.stderr)
            sys.exit(1)

    # Find Instances/Invocations column
    instances_col = None
    for col in columns:
        if col.lower() in ["instances", "invocations", "count", "calls"]:
            instances_col = col
            break

    # Sort rows
    def parse_numeric(val: str) -> float:
        """
        Parse numeric value from CSV, handling commas and empty strings.

        Parameters
        ----------
        val : str
            String value from CSV cell that should represent a number

        Returns
        -------
        float
            Parsed numeric value, or 0.0 if empty/invalid

        Notes
        -----
        Handles common CSV formatting like comma separators in large numbers.
        Returns 0.0 for empty or whitespace-only strings to enable sorting.
        """
        if not val or val.strip() == "":
            return 0.0
        return float(val.replace(",", ""))

    try:
        rows.sort(key=lambda r: parse_numeric(r[sort_col]), reverse=True)
    except (ValueError, KeyError) as e:
        print(f"Error sorting by column '{sort_col}': {e}", file=sys.stderr)
        sys.exit(1)

    # Extract top-K (or all if topk is None)
    if args.topk is None:
        top_kernels = rows
        extracted_count = len(rows)
        extraction_desc = f"all {extracted_count}"
    else:
        top_kernels = rows[: args.topk]
        extracted_count = len(top_kernels)
        extraction_desc = f"top-{extracted_count}"

        if len(top_kernels) < args.topk:
            print(f"Warning: Only found {len(top_kernels)} kernels, requested top-{args.topk}", file=sys.stderr)

    # Build YAML data structure
    yaml_data: Dict[str, List[Dict[str, Any]]] = {
        "kernels": []
    }

    for i, row in enumerate(top_kernels, 1):
        name = row.get(name_col, "").strip()
        if not name:
            print(f"Warning: Row {i} has no kernel name, skipping", file=sys.stderr)
            continue

        # Get metrics for description
        time_pct = row.get("Time (%)", "?")
        instances = row.get(instances_col, "?") if instances_col else "?"

        # Generate exact-match regex pattern
        regex_pattern = escape_kernel_name_to_regex(name)

        kernel_entry = {
            "name": name,
            "regex": regex_pattern,
            "description": f"Kernel #{i} - {time_pct}% of GPU time, {instances} calls"
        }
        yaml_data["kernels"].append(kernel_entry)

    # Generate YAML with ruamel.yaml
    yaml = ruamel.yaml.YAML()
    yaml.preserve_quotes = True
    yaml.width = 1000000  # Set extremely large width to prevent any wrapping
    yaml.default_flow_style = False
    yaml.map_indent = 2
    yaml.sequence_indent = 4

    # Create header comments
    header_lines = [
        "# Top kernels extracted from Nsight Systems profiling",
        f"# Source: {csv_path.name}",
        f"# Sorted by: {args.sort_by} (column: {sort_col})",
        "# Format: each kernel has full name and regex pattern",
        "# The regex pattern is used for ncu --kernel-name filtering",
        "#",
        "# NOTE: The regex patterns below use re.escape() for exact matching.",
        "# You may want to manually edit these patterns to match kernel families",
        "# instead of exact instances. For example:",
        "#   - Exact: 'internal::gemvx::kernel<...>(int)7...' (matches one variant)",
        "#   - Family: '.*internal::gemvx::kernel<.*\\(int\\)7.*>' (matches all (int)7 variants)",
        "",
    ]

    # Generate YAML content
    yaml_buffer = StringIO()
    yaml.dump(yaml_data, yaml_buffer)
    yaml_content = "\n".join(header_lines) + yaml_buffer.getvalue()

    # Output handling: file or console
    if args.output:
        # Write to file
        try:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(yaml_content)

            print(f"✓ Extracted {extraction_desc} kernels to: {output_path}", file=sys.stderr)
            print(f"  Source: {csv_path}", file=sys.stderr)
            print(f"  Sorted by: {sort_col}", file=sys.stderr)
            print("\nNext steps:", file=sys.stderr)
            print(f"  1. Review {output_path} and adjust regex patterns if needed", file=sys.stderr)
            print(f"  2. Run: ./scripts/ncu/examples/ncu-profile-top-kernels.v2.sh --yaml {output_path}", file=sys.stderr)

        except Exception as e:
            print(f"Error writing YAML: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Display on console as plain text
        # Print info to stderr so YAML goes to stdout for redirection
        print(f"# Extracted {extraction_desc} kernels from {csv_path} (sorted by {sort_col})", file=sys.stderr)

        # Print YAML content to stdout
        print(yaml_content)


if __name__ == "__main__":
    main()
