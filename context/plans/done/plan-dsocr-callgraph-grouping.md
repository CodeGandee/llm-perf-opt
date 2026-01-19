# Plan – Grouping DeepSeek-OCR Dynamic Call Graph into `for N` / `parfor N`

## HEADER
- **Status**: Done
- **Completed**: 2026-01-19

## Goal

Take the TorchLens-derived dynamic call graph for DeepSeek-OCR (stored in `tmp/dsocr-torchlens-callgraph/dsocr-call-graph-torchlens.{json,dot}`) and aggregate repeated module calls into higher-level constructs using *graph-structural* semantics:

- `XYZModule for N` — **stacked** modules along depth (e.g., `blocks.0..blocks.N-1` or `layers.0..layers.N-1`).
- `XYZModule parfor N` — **repeated calls of the same module node at a single graph level** under the same parent, regardless of whether execution was actually parallel or serial (e.g., a loop that calls `sam_model.blocks.0` 100 times is treated as `parfor 100`).

In other words, `parfor` means “N sibling calls at the same call-graph level to the same module”, not necessarily true hardware/software parallelism. The output should be a model-specific, human-readable description that’s stable enough to feed analytic performance models.

## Current State (What TorchLens Gives Us)

The JSON produced by `dsocr_torchlens_callgraph.py` contains:

- `call_counts`: module name → number of passes (`history.module_num_passes`).
- `edges`: flattened `"{parent}->{child}"` → edge frequency (built from `containing_modules_origin_nested` across layers).
- `op_call_counts`: module name → number of ops whose innermost module is this module.
- `module_children`: parent → list of child modules (static module hierarchy).

Observations from current DeepSeek-OCR trace:

- Many modules (e.g., `sam_model.blocks.i.*`) have `call_counts == 2`, reflecting global + local views.
- Edge weights from `sam_model` to blocks (`label="128"`, `120`, etc.) indicate many dynamic transitions through the same modules.
- The graph is rich enough to see structure, but it does not directly label which repeats are “sequential” vs “parallel”.

## Options for Grouping Strategy (Revised Semantics)

### Option A – Pure Post-Processing of TorchLens JSON/DOT (No New Hooks)

**Idea**

Implement a dedicated grouping script that:

1. Loads `dsocr-call-graph-torchlens.json`.
2. Uses `call_counts`, `edges`, and `module_children` together with DeepSeek-OCR-specific knowledge to infer:
   - Sequential loops (`for N`): e.g., repeated passes through transformer blocks in the LLM stack.
   - Parallel fan-out (`parfor N`): e.g., per-crop processing in `sam_model` / `vision_model`.
3. Emits a compact representation, e.g.:
   - `sam_model.blocks.0 for 1 parfor 4` (4 crops, single layer position).
   - `model.layers[i] for 32` (32 decoder layers, 1 pass each).

**How to Distinguish `for` vs `parfor` with Graph Semantics**

- `for N` (stacked depth):
  - Count how many *distinct* modules of the same “family” appear along a depth-wise chain:
    - E.g., `sam_model.blocks.0`, `sam_model.blocks.1`, …, `sam_model.blocks.L-1` → `sam_model.blocks for L`.
    - Similarly for `model.layers.0..31`, `sam_model.blocks.*.mlp`, etc.
  - This is essentially “how many times does this kind of module appear as unique nodes along the forward path?”.

- `parfor N` (repeated calls at one level):
  - For a given module name (e.g., `sam_model.blocks.0.mlp`), examine:
    - Its `call_counts` (TorchLens module passes).
    - Edge multiplicities from its parent(s) in `edges`.
  - If the same module node is invoked N times from the same parent (same module address), report:
    - `sam_model.blocks.0.mlp parfor N`.
  - This explicitly covers loops: even if the module is called serially in Python (e.g., a for loop), it still appears as multiple passes at the same call-graph level and is treated as `parfor N`.

- Use **stage knowledge** and shapes only to help interpret the numbers (e.g., N ≈ number of crops for vision vs N ≈ sequence length for LLM), not to define parallelism.

**Pros**

- No further monkeypatching or changes to vendor code.
- Reusable for repeated runs (only requires JSON).
- Easy to make deterministic, model-specific rules.

**Cons**

- Relies on heuristics and model-specific knowledge (but that is acceptable here).
- TorchLens’s notion of “passes” is coarse; some patterns may be ambiguous without additional instrumentation.

### Option B – Attach Custom Hooks on Top of TorchLens (Dynamic Classification)

**Idea**

Add a separate runtime tracer (forward hooks) for selected modules (e.g., LLM blocks, vision blocks) that:

- Logs a per-step “lane ID” (e.g., crop index, sequence position, or image index).
- Tracks call stacks with logical “iteration counters”.

Then post-process:

- Calls where we see repeated invocations of the same module (same address) with varying lane IDs but the same parent → `parfor N` for that module.
- Chains of different module indices (e.g., `layers.0..31`) still define `for N` depth stacks.

**Pros**

- More robust, less heuristic; explicitly captures parallelism.
- Can be extended to more complex stages (prefill vs decode).

**Cons**

- Additional runtime complexity, especially when combined with TorchLens.
- Requires careful hook registration and teardown to avoid interfering with profiling.

### Option C – Monkeypatch DeepSeek-OCR to Emit Loop Metadata

**Idea**

Inject small, model-specific helpers into the vendor model at runtime (without modifying files on disk), for example:

- Wrap `sam_model` / `vision_model` calls with functions that:
  - Tag outputs with metadata (e.g., number of crops).
  - Emit a light-weight log of loop structure.
- Wrap LLM block iteration loops to log “iteration IDs”.

Then merge this metadata with TorchLens’ graph to compute exact `for` / `parfor` groupings.

**Pros**

- Very precise grouping; we know exactly which loops are over crops vs tokens vs layers.
- Still avoids editing the HF cache on disk (only runtime monkeypatch).

**Cons**

- More invasive than pure post-processing; fragile if vendor changes code layout.
- Harder to maintain across model versions.

## Recommended Approach

**Short Term (within this repo)**: **Option A – Pure Post-Processing on TorchLens JSON + Stage Knowledge**, with a small amount of DeepSeek-OCR-specific logic and graph-based `for`/`parfor` semantics.

Rationale:

- We already have a reliable TorchLens trace and static stage mapping.
- The call graph is deterministic for this model and config; heuristic rules can be codified clearly.
- We avoid extra runtime hooks/monkeypatch beyond the minimal `quick_gelu` patch already needed for TorchLens.
- Easy to iterate on the grouping rules as we refine analytic models.

**Concrete Plan for Option A**

1. **Define grouping rules and configuration**
   - Add a small config structure (Python dict) mapping:
     - Stage → module prefix (reuse `DeepseekOCRStaticAnalyzer.m_stage_module_map`).
     - For each family of modules, whether we want to:
       - Summarize depth (`for N` over indices), and/or
       - Summarize repeated passes (`parfor N` over dynamic calls per node).
   - For example:
     - `sam_model.blocks.*`: report `for L` over block indices plus `parfor N` for per-block repeated passes.
     - `model.layers.*`: report `for L` over layer indices; `parfor N` if we observe multiple passes of the same `layers.i` node.

2. **Implement a grouping script**
   - New script: `scripts/analytical/dsocr_callgraph_group_modules.py`.
   - Responsibilities:
     - Load `tmp/dsocr-torchlens-callgraph/dsocr-call-graph-torchlens.json`.
     - For each module family and module node:
       - Identify its stage and prefix.
       - Use `module_children` and naming patterns (`blocks.i`, `layers.j`) to compute depth-wise `for N`.
       - Use `call_counts` and outgoing edges from parents to compute node-local `parfor M` (how many times this node is used at that graph level).
     - Produce a summary JSON / Markdown with grouped lines like:
       - `sam_model.blocks[0..3].mlp for 4` (four depth-wise blocks).
       - `sam_model.blocks.0.mlp parfor 4` (same module node used four times from its parent).
       - `model.layers[0..31] for 32`.

3. **Integrate into Pixi and docs**
   - Add a Pixi task, e.g.:
     - `dsocr-callgraph-group = { cmd = "python scripts/analytical/dsocr_callgraph_group_modules.py" }`.
   - Document:
     - The interpretation of `for` vs `parfor`.
     - How this maps back to the original TorchLens graph (for debugging).

4. **Validation**
   - Cross-check grouped output against:
     - Static analysis reports (per-stage FLOPs/activations).
     - Known model architecture (e.g., number of SAM blocks, transformer layers).
   - Optionally sample a few TorchLens paths to confirm grouping choices.

**Medium Term (Optional Enhancements)**

- If heuristics prove insufficient, add a minimal hook-based tracer for a few key modules (Option B) to record “lane IDs” for crops vs tokens, then feed that into the same grouping logic.
- If we ever need exact loop bounds for a different DeepSeek-OCR variant, add per-model config files that define the grouping patterns instead of hard-coding them in the script.
