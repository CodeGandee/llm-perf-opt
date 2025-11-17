# Plan – DeepSeek-OCR Call Graph Grouping via Runtime Monkeypatching (Option C)

## Context and Goal

We already have a TorchLens-derived dynamic call graph for DeepSeek-OCR written to:

- `tmp/dsocr-torchlens-callgraph/dsocr-call-graph-torchlens.json`
- `tmp/dsocr-torchlens-callgraph/dsocr-call-graph-torchlens.dot`

This graph shows many repeated calls to the same modules (e.g., `sam_model.blocks.i.*`, `model.layers.j.*`), but does not explicitly label whether those repeats are depth-wise (`for N`) or same-level (`parfor N` in our graph sense).

Option C is to **monkeypatch the DeepSeek-OCR model implementation at runtime** so that, during the TorchLens-traced forward pass, we also emit structured metadata about loop structure and “lane” usage. This metadata will let us classify modules as:

- `XYZModule for N` – N distinct modules of the same family along depth (e.g., `blocks.0..blocks.N-1`).
- `XYZModule parfor N` – N uses of the **same module node at the same call-graph level**, even if the Python execution was serial (e.g., a loop that calls `sam_model.blocks.0` 100 times is summarized as `parfor 100`).

The goal of this plan is to specify *which functions we’d patch*, *how*, and *what metadata we’d emit*, so we can judge complexity before coding.

---

## High-Level Strategy

1. **Keep TorchLens as the primary dynamic tracer** for module addresses, parents, and passes.
2. **Add a lightweight DeepSeek-OCR–specific runtime tracer** that logs loop structure and “lanes”:
   - Lanes: crops / images / sequence spans.
   - Loops: per-block repetition, per-layer iteration, decode steps (if/when we add them).
3. **Implement the tracer as monkeypatch wrappers** around specific vendor functions:
   - No modification to files under `models/deepseek-ocr` or Hugging Face cache on disk.
   - Patches live in our analytical scripts (e.g., under `scripts/analytical/`), activated only for tracing runs.
4. **Combine TorchLens graph + loop metadata** to produce compact `for` / `parfor` summaries.

---

## Candidate Patch Points in DeepSeek-OCR

These are based on the HF-vendored code under:

- `models/deepseek-ocr/modeling_deepseekocr.py` (vendor copy)
- HF cache: `~/.cache/huggingface/modules/transformers_modules/<hash>/modeling_deepseekocr.py` and `deepencoder.py`

### 1. `DeepseekOCRModel.forward` (Vision + Projector + Text Integration)

File: HF cache `modeling_deepseekocr.py` (mirrored in `models/deepseek-ocr/modeling_deepseekocr.py`).

Key responsibilities:

- Uses `sam_model` and `vision_model` to produce global and local visual features.
- Applies `projector` to produce embeddings for both global and cropped views.
- Interleaves vision features into text token embeddings via `inputs_embeds[idx].masked_scatter_(...)`.

Structure we care about:

- Loop over `(image, crop_shape)`:

  ```python
  for image, crop_shape in zip(images, images_spatial_crop):
      images_in_this_batch = []
      patches = image[0]
      image_ori = image[1]
      ...
      global_features = ...
      local_features = ...
      images_in_this_batch.append(global_local_features)
      ...
      images_in_this_batch = torch.cat(images_in_this_batch, dim=0)
      inputs_embeds[idx].masked_scatter_(images_seq_mask[idx].unsqueeze(-1).cuda(),
                                         images_in_this_batch)
  ```

**Patch idea**:

- Wrap `DeepseekOCRModel.forward` with a function that:
  - Assigns a *vision stage ID* (e.g., sample, crop index).
  - Logs:
    - How many image tuples were processed (`len(images)`).
    - How many crops per image (`w_crop * h_crop` from `images_spatial_crop`).
  - For each `(image, crop_shape)` iteration, append a record:

    ```json
    {
      "stage": "vision",
      "module": "sam_model.blocks",
      "image_index": i,
      "num_crops": w_crop * h_crop
    }
    ```

- We don’t change the body; we just wrap around it and inspect arguments (`images`, `images_spatial_crop`).

### 2. `DeepseekV2Model` / `DeepseekV2ForCausalLM.forward` (LLM Stack)

File: HF cache `modeling_deepseekv2.py`.

Key loop:

```python
for decoder_layer in self.layers:
    layer_outputs = decoder_layer(...)
    ...
```

There may be variants (e.g., `for i, layer in enumerate(self.layers)`), but structurally it’s the standard decoder stack.

**Patch idea**:

- Monkeypatch the **model class that owns `self.layers`** to wrap its forward:

  - Before the loop, log how many layers there are: `num_layers = len(self.layers)`.
  - Within the loop, log:

    ```json
    {
      "stage": "llm",
      "module_family": "model.layers",
      "layer_index": i
    }
    ```

- This gives us:
  - `for num_layers` along depth.
  - Potential `parfor` if we ever see multiple passes through the same `layers[i]` node (e.g., decode steps).

### 3. Optional: Decode Loop (Prefill vs Decode)

If we want to separately summarize decode, we can patch the **generation loop** if DeepSeek-OCR exposes one in its HF module (often inside `generate` or a helper).

Given current focus on one-shot forward with TorchLens, we can defer this until we need decode grouping.

---

## Monkeypatch Design

We’ll implement a small runtime tracer module (e.g., `scripts/analytical/dsocr_callgraph_runtime_patch.py`) with:

### 1. A Global Logger Structure

Python-side structure (pseudocode):

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class DsocrCallgraphLog:
    events: list[dict[str, Any]] = field(default_factory=list)

LOG = DsocrCallgraphLog()
```

We’ll append structured events to `LOG.events` during the patched forwards and write them to:

- `tmp/dsocr-torchlens-callgraph/dsocr-callgraph-runtime-metadata.json`

### 2. Patch Helpers

#### `patch_deepseek_ocr_model_forward()`

- Locate the HF module via the session’s model:

  ```python
  model = session.m_model  # DeepseekOCRForCausalLM
  model_mod = __import__(model.__class__.__module__, fromlist=["*"])
  DeepseekOCRModel = getattr(model_mod, "DeepseekOCRModel")
  ```

- Wrap `DeepseekOCRModel.forward`:

  ```python
  orig_forward = DeepseekOCRModel.forward

  def wrapped_forward(self, *args, **kwargs):
      images = kwargs.get("images", None)
      images_spatial_crop = kwargs.get("images_spatial_crop", None)
      # infer metadata about images / crops if present
      LOG.events.append({
          "kind": "vision_inputs",
          "module": "DeepseekOCRModel",
          "num_images": len(images) if images is not None else 0,
          "images_spatial_crop": (
              images_spatial_crop.detach().cpu().tolist()
              if images_spatial_crop is not None and hasattr(images_spatial_crop, "detach")
              else None
          ),
      })
      return orig_forward(self, *args, **kwargs)
  ```

- Assign back:

  ```python
  DeepseekOCRModel._orig_forward_for_callgraph = orig_forward
  DeepseekOCRModel.forward = wrapped_forward
  ```

This is safe for tracing runs and reversible if needed.

#### `patch_deepseek_v2_layers_forward()`

- Identify the LLM base model class that owns `self.layers`. In HF DeepSeekV2, this is typically something like `DeepseekV2Model` or the decoder class.
- Option 1 (simpler): patch `DeepseekV2ForCausalLM.forward` in `modeling_deepseekv2.py`:

  ```python
  DeepseekV2ForCausalLM = getattr(modeling_deepseekv2_mod, "DeepseekV2ForCausalLM")
  orig_forward = DeepseekV2ForCausalLM.forward

  def wrapped_forward(self, *args, **kwargs):
      num_layers = len(self.model.layers)
      LOG.events.append({
          "kind": "llm_stack",
          "module": "model.layers",
          "num_layers": num_layers,
      })
      return orig_forward(self, *args, **kwargs)
  ```

- Option 2 (more granular): patch the decoder block class itself (e.g., `DeepseekV2DecoderLayer`), and record `layer_index` using a naming convention or attributes. This is more invasive and may be deferred unless we need per-layer dynamic call counts beyond what TorchLens already provides.

Given TorchLens already distinguishes individual `model.layers.i` modules and gives `module_num_passes`, **Option 1** (logging just `num_layers`) is sufficient to distinguish `for num_layers` vs `parfor` uses.

### 3. Patch Activation and Teardown

- In the tracing script (`scripts/analytical/dsocr_torchlens_callgraph.py`), after building `DeepSeekOCRSession`, we will:

  ```python
  from scripts.analytical import dsocr_callgraph_runtime_patch as rt

  rt.apply_runtime_patches(session.m_model)
  ```

  where `apply_runtime_patches` calls:

  - `patch_deepseek_quick_gelu()` (already needed for TorchLens).
  - `patch_deepseek_ocr_model_forward()`.
  - `patch_deepseek_v2_layers_forward()` (or equivalent).

- After `tl.log_forward_pass(...)` completes, we will:

  ```python
  rt.write_runtime_log(out_root / "dsocr-callgraph-runtime-metadata.json")
  ```

  with a simple `json.dump(rt.LOG.events, ...)`.

We can optionally add an unpatch function if we want to reuse the process for non-tracing runs.

---

## Combining TorchLens Graph with Runtime Metadata

With this patching in place, we’ll have:

1. **TorchLens graph**:
   - Module addresses (`sam_model.blocks.0.mlp`, `model.layers.3`, etc.).
   - `module_num_passes` per module.
   - Parent→child edge counts (used for `parfor` at graph level).

2. **Runtime metadata**:
   - Vision inputs:
     - Number of images.
     - Number of crops per image (from `images_spatial_crop`).
   - LLM stack:
     - Number of layers (`len(self.model.layers)`).

We will combine these in the grouping script (from Option A plan) by:

- **Depth `for N`**:
  - Use `num_layers` from runtime metadata and static names (`model.layers.0..N-1`) to summarize `model.layers for N`.
  - Similarly, if we want to summarize `sam_model.blocks` stack, count distinct `sam_model.blocks.i` modules from TorchLens graph and label `sam_model.blocks for L`.

- **Same-level `parfor N`**:
  - Use TorchLens `module_num_passes` and edge counts to see how many times the same module node (e.g., `sam_model.blocks.0.mlp`) is used from the same parent.
  - Check that totals are consistent with runtime metadata (e.g., `num_images * num_crops`).

This gives a robust, model-specific way to map from the raw dynamic trace to `for` / `parfor` summaries, with only light monkeypatching of runtime behavior and no persistent changes to vendor code.

---

## Complexity Assessment

**Scope of changes:**

- New runtime patch module under `scripts/analytical/` (no changes to `src/` or vendor directories).
- 2–3 monkeypatched functions:
  - `DeepseekOCRModel.forward`.
  - `DeepseekV2ForCausalLM.forward` (or its base model).
  - (Already present) `quick_gelu` patch.
- One additional JSON artifact:
  - `tmp/dsocr-torchlens-callgraph/dsocr-callgraph-runtime-metadata.json`.

**Risk and fragility:**

- Tight coupling to DeepSeek-OCR’s current HF implementation (function names and module paths).
- If vendor updates `modeling_deepseekocr.py` or `modeling_deepseekv2.py`, patch functions may need to be adjusted.
- However, since patches live in analytical tooling and do not alter production pipelines, this is acceptable for research/profiling use.

Overall complexity: **moderate but contained**, with clear, isolated patch points and minimal behavioral impact outside tracing runs.

