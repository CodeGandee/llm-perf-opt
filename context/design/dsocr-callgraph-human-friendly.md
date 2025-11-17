# DeepSeek-OCR Call Graph – Human-Friendly Contract

This document defines the **desired shape and semantics** of the DeepSeek-OCR call graph we want to generate and visualize. It is the contract that downstream tooling and visualizations should follow.

The intent is to:

- Expose the **PyTorch module structure** (class + runtime name).
- Summarize **dynamic repetition** using `for N` / `parfor N` labels.
- Attach **runtime metadata per node** (constructor config that affects tensor sizes).
- Attach **input/output tensor shape + dtype information per node** so we can estimate
  total tensor sizes and memory footprint.
- Focus purely on **module-to-module calls** (no explicit orchestrator/manager nodes).

## 1. Node Identity and Labels

### 1.1 Every node is a `nn.Module`

- Each graph node corresponds to a **single PyTorch `torch.nn.Module` instance** or a family of such instances.
- The node label must always include:
  - The **module class name** (e.g., `DeepseekOCRBlock`, `SAMBlock`, `Linear`).
  - The **runtime name** (hierarchical attribute path inside the model), e.g.:
    - `sam_model.blocks.0.attn`
    - `vision_model.transformer.layers.3.mlp.fc1`
    - `layers.7.mlp.experts.12`

**Preferred label format** (single instance):

- `<ClassName> @ <runtime-name>`
  - Example: `SAMBlock @ sam_model.blocks.3`
  - Example: `Linear @ sam_model.blocks.0.mlp.lin1`

### 1.2 Multi-instance nodes and patterns

When there are multiple runtime instances of the **same module class** that we want to group, the node should:

- Still show the **class name**.
- Show the runtime name as a **pattern** that matches all grouped instances.

Examples (patterns are conceptual – regex-like, not necessarily literal regex):

- `SAMBlock @ sam_model.blocks.[0-11]`
  - Represents blocks `sam_model.blocks.0` through `.11`.
- `MoEExpert @ layers.3.mlp.experts.[0-63]`
- `SelfAttention @ vision_model.transformer.layers.*.self_attn`

We can use:

- Bracket ranges: `[0-11]`, `[0,1,2,3]`, `[7-63]`.
- Wildcards: `*` to denote “any index at this position”.

The exact syntax is less important than **stability and clarity**; tools should treat this as an opaque label with clear human meaning.

### 1.3 Per‑node runtime metadata (constructor/config)

In addition to the human‑readable label, the call graph **must carry structured
metadata for each node** capturing the module’s configuration at runtime. This
is primarily used for tensor‑size and memory estimation.

For each module node (single instance or pattern family) we require:

- **Constructor / config parameters** that influence tensor shapes, for example:
  - `hidden_size`, `embed_dim`, `num_heads`, `intermediate_size`
  - `num_layers`, `num_blocks`, `num_experts`, `top_k`
  - `kernel_size`, `stride`, `padding`, `dilation`
  - Any other dimensions that determine activation sizes or parameter counts.
- These should be stored as a **JSON‑serializable dict** (e.g. `constructor_params`)
  keyed by the module’s runtime name (or pattern) in the callgraph JSON.
- The node label itself stays compact (`Class @ runtime`); tools should rely on
  the metadata for detailed analysis rather than overloading the label.

## 2. Repetition Semantics: `parfor N` vs `for N`

We distinguish between two kinds of dynamic repetition:

- **`parfor N`** – repeated calls of the **same module instance** within one caller.
- **`for N`** – a stack / sequence of **distinct module instances** of the same type, called in series (output of one feeds into the next).

### 2.1 `parfor N` – repeated calls to the same instance

Definition:

- For a given module instance `M` (or pattern node), if its `.forward()` is invoked **N times from the same caller** within a single logical run, we represent this as:
  - `<ClassName> @ <runtime-or-pattern> parfor N`

Graphically:

- There is an **edge** from the caller node `C` to the module node `M` with label:
  - `parfor N`

Examples:

- A single attention block reused across crops:
  - Node: `SAMBlock @ sam_model.blocks.0`
  - Edge: `sam_model -> SAMBlock @ sam_model.blocks.0` labeled `parfor 128`
  - If grouped: `SAMBlock @ sam_model.blocks.[0-11].attn parfor 1160`

Semantics:

- `parfor` does **not** mean actual hardware parallelism; it means “**N sibling invocations at the same call-graph level of the same module instance**”.

### 2.2 `for N` – stacked modules in series

Definition:

- When we have **N distinct module instances** of the same conceptual kind (same class, different indices) that form a **serial chain** (output of index `i` feeds into input of index `i+1`), we represent this as:
  - `<ClassName> @ <pattern> for N`

Graphically:

- There is an edge from the **parent manager or container** module to this grouped node with label:
  - `for N`

Examples:

- Transformer blocks stacked in a vision encoder:
  - Node: `TransformerBlock @ vision_model.transformer.layers.[0-23] for 24`
  - Edge: `vision_model.transformer -> TransformerBlock @ vision_model.transformer.layers.[0-23]` labeled `for 24`
- SAM image encoder blocks:
  - Node: `SAMBlock @ sam_model.blocks.[0-11] for 12`
  - Edge: `sam_model -> SAMBlock @ sam_model.blocks.[0-11]` labeled `for 12`

Semantics:

- `for N` means “**N distinct layers in depth order**” – a stack, not a loop over the same object.

## 3. Edge Semantics

### 3.1 Direction and labels

- Edges are **directed**, from **caller** module to **callee** module:
- Edge labels:
  - `for N` – caller feeds into a **stack of N distinct module instances**.
  - `parfor N` – caller invokes the **same module instance N times**.
  - Optionally unlabelled or `once` for single calls we do not aggregate.

### 3.2 Granularity and grouping

- Grouping along indices:
  - Per-instance module nodes (`sam_model.blocks.0`, `sam_model.blocks.1`, …) **can be collapsed** into a pattern node (`sam_model.blocks.[0-11]`) when:
    - All instances share the same class.
    - They form a clean contiguous index range or a meaningful pattern.
  - Per-instance `parfor` edges (e.g., each block’s `attn`) can be **summed** into a single `parfor` count on the pattern node (e.g., `sam_model.blocks.*.attn parfor N`).

- The callgraph used for high-level visualization should prefer:
  - Pattern nodes with `for N` / `parfor N` annotations over a sea of per-index nodes, unless debugging a specific index.

### 3.3 Class-based grouping from call graph structure (visualization)

For the **visualized** graph (DOT/SVG/Mermaid), we further group *instances of the same module class* based on the dynamic call graph:

1. **Same-parent, same-class siblings ⇒ `parfor N`**
   - If a parent module `P` has multiple child modules `C₁, C₂, …, C_N` such that:
     - All `Cᵢ` have the **same module class** (e.g., multiple `DeepseekV2MLP` experts under one MoE gate).
     - All `Cᵢ` are direct children of **the same parent** `P` in the call graph.
   - Then, for visualization:
     - We render **one grouped child node** representing this family.
     - The edge `P → grouped_child` is labeled `parfor N`, indicating that `P` invokes *N same-class siblings* at that call level.
   - The underlying JSON callgraph may still keep per-instance nodes; the grouping is a **view** for readability.

2. **Same-class serial chains ⇒ `for N`**
   - If we observe a serial chain of modules `M₁ → M₂ → … → M_N` such that:
     - Each `Mᵢ` is a **distinct module instance**.
     - All `Mᵢ` share the **same module class**.
     - The **output** of `Mᵢ` feeds into the **input** of `Mᵢ₊₁` (a depth-wise stack, not siblings).
   - Then, for visualization:
     - We represent this as a **single grouped stack node** with a label like `<ClassName> @ <pattern> for N`.
     - The **parent→stack** edge (from the manager/container of the chain) is labeled `for N`.
   - This captures cases where stacked layers are best viewed as a single conceptual depth-wise block.

These class-based grouping rules are applied **on top of** index-based grouping (3.2) by inspecting the **call graph structure** (parents and dynamic edges). They affect only the rendered graph; analytic consumers can still rely on the full per-instance JSON if needed.

## 4. Summary of Preferences

1. **Node = `nn.Module`**  
   - Every node is a PyTorch `nn.Module`. Its label always includes both **class name** and **runtime name**.

2. **Patterns for multi-instance families**  
   - When a module appears as many instances, its runtime name is rendered as a **pattern** (range or wildcard) rather than listing all indices.

3. **Repeated calls ⇒ `parfor N`**  
   - If a module’s `.forward()` is called multiple times from the same caller, the node (and/or edge) carries a label like `<ClassName> @ <pattern> parfor N`.

4. **Stacks of distinct modules ⇒ `for N`**  
   - When modules of the same kind are stacked in depth and called in serial (`out[i]` → `in[i+1]`), they are represented as `<ClassName> @ <pattern> for N`, with an edge from their container labeled `for N`.

5. **Same-class grouping for visualization**  
   - For readability, visualizations:
     - Group **same-class siblings under the same parent** into a single node and annotate the parent→group edge with `parfor N`.
     - Group **same-class modules in a serial chain** into a single stack node and annotate the parent→stack edge with `for N`.
   - These groupings are derived from the **dynamic edge structure**; they do not require module names to share a particular index pattern, though index-based patterns may be used in the grouped label.

6. **Runtime config + I/O tensor metadata per node**  
   - For every node we also maintain:
     - A `constructor_params` (or similar) dict capturing module config that
       affects tensor dimensions.
     - A description of **input tensors** (per call or representative call),
       including **shape** (e.g., `[B, T, C]` or `[N, C, H, W]`) and **dtype**.
     - A description of **output tensors** with the same information.
   - This metadata does not need to appear in the visual label, but it must be
     present in the underlying callgraph JSON so downstream tools can estimate:
     - Per‑module activation sizes.
     - Aggregate tensor sizes along paths or for the full model.

These rules should guide both the grouping logic (how we post-process the TorchLens call graph) and the visualization (DOT/SVG/Mermaid) so that the resulting callgraph is both faithful to runtime behavior and readable for humans, while also supporting tensor‑size and memory analysis. Tools that manipulate or consume the callgraph should treat this document as the ground truth for expected structure, labels, and required metadata.
