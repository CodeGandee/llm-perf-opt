# Plan – Implement Option A for DeepSeek-OCR Call Graph Grouping

## Goal

Turn the TorchLens-derived dynamic call graph for DeepSeek-OCR (`tmp/dsocr-torchlens-callgraph/dsocr-call-graph-torchlens.json`) into a higher-level, human-readable description that:

- Groups stacked modules into `for N` constructs (depth-wise repetition, e.g., transformer layers, SAM blocks).
- Groups repeated calls of the same module node at one graph level into `parfor N` constructs (fan-out / looped reuse under the same parent).
- Produces an output format stable enough for downstream analytic performance models and documentation.

We will implement this purely as **post-processing on the TorchLens JSON** (Option A from `plan-dsocr-callgraph-grouping.md`), with DeepSeek-OCR-specific knowledge encoded as configuration + naming rules. No new runtime hooks beyond the existing TorchLens trace are required.

## Inputs and Existing Artifacts

- Primary input: `tmp/dsocr-torchlens-callgraph/dsocr-call-graph-torchlens.json` containing:
  - `call_counts`: module → number of forward passes.
  - `edges`: `"parent->child"` → dynamic edge count.
  - `op_call_counts`: module → number of ops whose innermost module is this module.
  - `module_children`: parent → list of child modules (static hierarchy).
- Optional visual aid: `tmp/dsocr-torchlens-callgraph/dsocr-call-graph-torchlens.dot` / `.svg` for manual inspection.
- Model knowledge:
  - Stage-specific prefixes already used elsewhere (e.g., `sam_model`, `vision_model`, `projector`, `model.layers.*`).
  - Static analyzer mapping in `DeepseekOCRStaticAnalyzer` (`m_stage_module_map`) to keep naming consistent with other reports.

## Deliverable

- New analytical script: `scripts/analytical/dsocr_callgraph_group_modules.py`.
- New Pixi task: `dsocr-callgraph-group` invoking the script.
- Output artifacts under `tmp/dsocr-torchlens-callgraph/`, e.g.:
  - `dsocr-call-graph-grouped.json` – machine-readable grouped representation.
  - `dsocr-call-graph-grouped.md` – human-readable summary with `for` / `parfor` lines.

We will initially target DeepSeek-OCR specifically but keep the design modular enough to add other models later if needed.

## Design Overview

### 1. Data Model for Grouped Output

Define a small, explicit schema for grouped constructs:

- `GroupKind = "for" | "parfor"`.
- `GroupNode` (JSON object):
  - `name`: canonical module family or node name (e.g., `sam_model.blocks`, `sam_model.blocks.0.mlp`).
  - `kind`: `"for"` or `"parfor"`.
  - `count`: integer `N`.
  - `indices`: optional list or range descriptor for depth indices (e.g., `[0, 1, 2, 3]` or `"0..3"`).
  - `stage`: optional stage label (e.g., `vision`, `sam`, `llm`) for downstream filtering.
  - `parent`: optional parent module name (for `parfor`).
  - `annotations`: optional list of free-form strings (e.g., `"global+local views"`, `"crops"`).

The top-level JSON will have:

- `groups`: list of `GroupNode`.
- `metadata`: version, timestamp, TorchLens source path, model name (if derivable).

The `.md` summary will present the same info as readable lines, for example:

- `sam_model.blocks[0..11] for 12  # SAM transformer blocks`
- `sam_model.blocks.0.mlp parfor 4 under sam_model.blocks.0  # per-crop MLP reuse`

### 2. Configuration for DeepSeek-OCR Module Families

Introduce a small dictionary within the new script (or a separate config module if it grows) describing:

- Stage names and module prefixes:
  - `"sam"` → `["sam_model", "sam_model.blocks", "sam_model.blocks.{i}"]`
  - `"vision"` → `["vision_model", "clip_model", "vision_model.encoder.layers.{i}"]` (if present)
  - `"llm"` → `["model.layers.{i}", "model.embed_tokens", "model.norm"]`
- For each prefix, flags controlling which group types to emit:
  - `depth_for_enabled`: bool (e.g., true for `sam_model.blocks.*`, `model.layers.*`).
  - `parfor_enabled`: bool (e.g., true for modules we expect to be reused many times under a parent).

We will:

- Use simple pattern matching on module names, focusing on numeric indices (`blocks.0`, `blocks.1`, …) to detect family members.
- Avoid heavy regex; keep it readable and explicit for DeepSeek-OCR.

### 3. Heuristics for `for N` (Depth-Wise Stacks)

For each module family that has numeric indices (e.g., `sam_model.blocks.{i}`, `model.layers.{i}`):

1. Enumerate children under their common parent using `module_children`:
   - E.g., for parent `sam_model`, collect all children with names like `sam_model.blocks.0`, `sam_model.blocks.1`, …
2. Extract integer indices and sort them:
   - Build contiguous ranges, e.g., `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]`.
3. For each contiguous range, emit:
   - `kind = "for"`, `count = len(range)`, `indices` = `"start..end"`.
4. Optionally enrich with:
   - `stage` = `sam` or `llm`.
   - `annotations` with known architectural facts (e.g., `"SAM transformer blocks"`).

This is purely static, derived from the hierarchy and naming, and does not rely on dynamic call counts.

### 4. Heuristics for `parfor N` (Sibling Reuse at a Level)

For each module node `M` that we care about (based on prefixes and stages):

1. Find its parents using the `edges` map:
   - Treat `"parent->child"` edges as dynamic transitions.
   - For each edge where `child == M`, note `edge_call_counts[(parent, child)]`.
2. For each `(parent, M)` pair:
   - Let `N_edge = edge_call_counts[(parent, M)]`.
   - Obtain `module_call_counts[M]` if needed as a sanity check.
3. Decide when to emit a `parfor`:
   - If `N_edge > 1`, emit:
     - `name = M`, `kind = "parfor"`, `count = N_edge`, `parent = parent`.
   - Optionally annotate:
     - If we know from model semantics that `M` processes crops, sequences, or tokens, add a comment (e.g., `"approx crop count"`).
4. To avoid noise:
   - We may set a minimum threshold (e.g., `N_edge >= 2`) and ignore trivial `parfor 1`.
   - Optionally restrict to specific families (e.g., `sam_model.blocks.*`, `model.layers.*.mlp`).

This aligns with the semantic definition: repeated calls of the same module node under the same parent correspond to `parfor N`, regardless of true parallelism.

### 5. Script Structure

`scripts/analytical/dsocr_callgraph_group_modules.py` will have:

- `load_callgraph(path: Path) -> dict[str, Any]`:
  - Load the JSON and perform basic validation of expected keys.
- `parse_edges(edges_dict) -> dict[tuple[str, str], int]`:
  - Convert `"parent->child"` strings back into `(parent, child)` tuples.
- `discover_indexed_families(module_children: dict[str, list[str]]) -> dict[str, list[int]]`:
  - Utility to find modules that share a base prefix with numeric tail segments.
- `build_for_groups(...) -> list[GroupNode]`:
  - Implement the depth-wise logic from Section 3 using configuration + `module_children`.
- `build_parfor_groups(...) -> list[GroupNode]`:
  - Implement the sibling-reuse logic from Section 4 using `edges` + `call_counts`.
- `write_grouped_json(...)` and `write_grouped_markdown(...)`:
  - Write `dsocr-call-graph-grouped.json` and `dsocr-call-graph-grouped.md`.
- `main()`:
  - CLI arguments (e.g., `--input-json`, `--output-dir`, `--verbose`) with defaults set to the standard TorchLens call graph paths.

We will keep the script stateless and idempotent: rerunning it with the same input JSON should always produce the same outputs.

### 6. Pixi Task and Integration

- Add a Pixi task in `pyproject.toml`:
  - `dsocr-callgraph-group = { cmd = "python scripts/analytical/dsocr_callgraph_group_modules.py" }`.
- Optionally define a helper task that chains generation and grouping:
  - `dsocr-callgraph-all = { depends-on = ["dsocr-torchlens-callgraph"], cmd = "python scripts/analytical/dsocr_callgraph_group_modules.py" }`.
- Update any relevant docs (if needed later) to mention the grouped output files and how to interpret them.

### 7. Validation Strategy

After implementing the script:

- Run `pixi run dsocr-torchlens-callgraph` followed by `pixi run dsocr-callgraph-group`.
- Manually inspect:
  - That `for` counts for `sam_model.blocks` and `model.layers` match known architecture (e.g., 12 SAM blocks, 32 LLM layers).
  - That `parfor` counts on key modules are plausible relative to what we expect from:
    - Number of crops or image patches.
    - Sequence lengths used in `dsocr_torchlens_callgraph.py`.
- Cross-check a few representative modules by hand from the DOT/JSON to ensure:
  - `parfor N` matches `edge_call_counts[(parent, child)]`.
  - `for N` ranges align with contiguous indices in `module_children`.

If we discover ambiguous patterns or noisy `parfor` groups, we’ll refine:

- The per-prefix configuration (which modules we summarize).
- Minimum thresholds for `N`.
- Optional blacklist of modules that should never be grouped.

### 8. Future Extensions (Optional, Not in Initial Implementation)

- Add a small CLI flag or config file to switch between:
  - Raw grouping (only `for` / `parfor` semantics).
  - Stage-aware grouping that collapses stages into single lines (e.g., `vision_stage parfor N`).
- Introduce support for non-indexed module families (e.g., naming patterns without numeric suffixes) using explicit lists in config.
- If needed, integrate with a later Option B/C implementation that may supply additional metadata (e.g., explicit “crop count” or “token count”) to replace some of the heuristics while reusing the same grouping output format.

