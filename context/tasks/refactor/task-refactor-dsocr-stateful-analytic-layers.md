**What to Refactor**
- Files (DeepSeek-OCR analytic layers and consumers):
  - `extern/modelmeter/models/deepseek_ocr/layers/llama/llama_flash_attention2.py`
  - `extern/modelmeter/models/deepseek_ocr/layers/decoder/deepseek_v2_decoder_layer.py`
  - `extern/modelmeter/models/deepseek_ocr/layers/core/deepseek_ocr_model.py`
  - `extern/modelmeter/models/deepseek_ocr/layers/stage_cost.py`
  - `extern/modelmeter/models/deepseek_ocr/HOLISTIC_ANALYSIS.md`
  - `src/llm_perf_opt/runners/dsocr_analyzer.py`
- Scope:
  - Make key analytic layers **explicitly stateful** with respect to:
    - stage mode (`prefill` vs `decode`), and
    - KV-cache meta state (context length, decode progress).
  - Add small **member functions to mutate state** (e.g., `set_prefill_state`, `set_decode_state`, `step_decode`) rather than recreating analytic layers for different stages.
  - Update holistic docs and analyzer wiring so prefill/decode simulations and hooked runs use a **single evolving analytic model**, mirroring the real DeepSeek-OCR decoder / KV-cache API.

**Why Refactor**
- Current analytic layers are mostly **shape-only and stateless**:
  - `LlamaFlashAttention2` has internal `m_seq_len_q` / `m_seq_len_kv` for FLOPs but still drives I/O and activation memory from a single `m_seq_len`. There is no explicit notion of "current stage" or KV length.
  - `DeepseekV2DecoderLayer` currently implements decode modeling by **cloning a new decoder layer** and manually overriding the attention primitive’s internal fields instead of tracking stage/KV state on the original instance.
  - `DeepseekOCRModel` aggregates a vision stack and a repeated decoder layer, but has no concept of **run state** (`prefill` vs `decode`, decode step counter, effective KV length); any stage helpers today recompute shapes rather than reading a shared state.
- This causes friction for:
  - **Separate-stage stats**:
    - Prefill vs decode FLOPs/IO/memory are computed by changing `seq_len` or instantiating separate objects, not by evolving a shared model state.
    - It’s easy to drift between prefill/decode conventions in different call sites.
  - **Simulated runs** (see `HOLISTIC_ANALYSIS.md`):
    - The doc describes simulated prefill + K-token decode runs and KV growth, but the code snippets rely on re-instantiating layers with different `seq_len` values instead of using a stateful model.
  - **Hooked runs**:
    - Real DeepSeek-OCR decoder layers keep `past_key_values` and layer instances alive across decode steps.
    - The analytic model would be easier to wire as a **shadow profiler** if it had similar stateful semantics (one analytic layer per HF layer, shared across steps, with mutable KV meta).
- Aligning the analytic model with the real DeepSeek-OCR implementation:
  - Reduces impedance in `DeepseekOCRStaticAnalyzer` and future hook-based tooling.
  - Makes the holistic analysis notes more **directly executable** rather than recipe-only.

**Key Questions and Answers**
1) **How do we distinguish prefill vs decode analytically, and support both modes?**
   - Prefill is modeled as a **full-sequence pass over the initial context**, controlled by the root model:
     - The root analytic model exposes `start_prefill(context_len, batch_size, kv_cache=None) -> SyntheticKVCache`.
     - Internally this:
       - sets `DeepseekV2DecoderLayer.m_stage = "prefill"`, `m_context_len = S_prefill`, `m_decode_len = 0`, and configures `LlamaFlashAttention2` with `S_q = S_kv = S_prefill`, and
       - attaches a `SyntheticKVCache` (either user-provided or created from config) so decoder layers can update KV state as they run, mirroring `DynamicCache` in the real model.
     - Pseudocode (prefill with managed KV cache):
       ```python
       # 1) Construct analytic model.
       analytic_model = DeepseekOCRModel.from_layers(
           vision_layer=vision_stack,          # SAM + CLIP + projector
           decoder_layer=decoder_layer,        # DeepseekV2DecoderLayer
           num_decoder_layers=num_layers,
       )

       # 2) Run analytic prefill. Caller passes workload args (context_len,
       #    batch_size) and may optionally pass a user-created SyntheticKVCache
       #    or None. The root model will:
       #    - reuse its existing cache if kv_cache is None and it already has one,
       #    - otherwise create or adopt the given kv_cache, and
       #    - keep the resulting cache in an internal member for future decode.
       kv_cache: SyntheticKVCache = analytic_model.start_prefill(
           context_len=S_prefill,              # matches HF prefill sequence
           batch_size=B,
           kv_cache=None,
       )
       # Internally, start_prefill might look like:
       #
       #   def start_prefill(
       #       self,
       #       *,
       #       context_len: int,
       #       batch_size: int,
       #       kv_cache: SyntheticKVCache | None = None,
       #   ) -> SyntheticKVCache:
       #       # Configure workload state
       #       self.m_context_len = context_len
       #       self.m_decode_len = 0
       #       self.m_batch_size = batch_size
       #
       #       # Acquire or create KV cache
       #       if kv_cache is None:
       #           kv_cache = self.m_kv_cache or SyntheticKVCache.from_config(self._config)
       #       self.m_kv_cache = kv_cache
       #
       #       # Run decoder prefill, letting layers call
       #       # kv_cache.get_usable_length(layer_idx) and kv_cache.update(...)
       #       return kv_cache

       flops_tc = analytic_model.forward_tensor_core_flops() or 0.0
       flops_cuda = analytic_model.forward_cuda_core_flops() or 0.0
       flops_prefill_tflops = flops_tc + flops_cuda
       io_prefill_tb = analytic_model.forward_cal_io() or 0.0
       weights_prefill_gb = analytic_model.forward_memory_weight() or 0.0
       acts_prefill_gb = analytic_model.forward_memory_activation() or 0.0
       kv_prefill_gb = analytic_model.forward_memory_kvcache() or 0.0
       ```
     - In this mode, `analytic_model.operation_mode == "prefill"`, and all standard
       `BaseLayer` stats (`forward_*`, `backward_*`) are interpreted as:
       “the cost of running prefill at the current `(context_len, batch_size)`.”
   - Decode (per token) is modeled as a **single-token query over an evolving KV cache**, again controlled by the root model:
     - The root model exposes:
       - `start_decode(kv_cache: SyntheticKVCache | None) -> SyntheticKVCache`, which:
         - switches mode to `"decode"`,
         - attaches the provided cache or reuses the internal one from prefill, and
         - configures decoder layers so that **current stats** (from `forward_*`) represent the cost of decoding the **next** token from the current KV state; and
       - `decode_one_token() -> SyntheticKVCache`, which:
         - requires that a KV cache is already attached (otherwise raises),
         - advances internal decode state (including KV lengths via `SyntheticKVCache.update(...)`), and
         - leaves stats aligned with the cost of decoding the *subsequent* token.
     - The decoder layer internally tracks how many decode tokens are already cached via `m_decode_len`:
       - At a given decode step, attention sees `S_q = 1` and `S_kv = S_prefill + K_cached`, where `K_cached == m_decode_len` mirrors `Cache.get_usable_length(...)` in the HF implementation.
     - Pseudocode (per-token decode with state evolution):
       ```python
        # After prefill:
        kv_cache = analytic_model.start_decode(kv_cache=None)
        # decoder_layer.m_stage = "decode"
        # decoder_layer.m_context_len = S_prefill
        # decoder_layer.m_decode_len = 0  # K_cached

        for step in range(K):  # simulate K decode tokens
            # 1) At this step, effective KV length is:
            #    S_kv = S_prefill + decoder_layer.m_decode_len
            #    which mirrors Cache.get_usable_length() in HF.

            # 2) Stats for the *next* token are given directly by forward_*.
            flops_tc = analytic_model.forward_tensor_core_flops() or 0.0
            flops_cuda = analytic_model.forward_cuda_core_flops() or 0.0
            flops_tflops = flops_tc + flops_cuda
            io_tb = analytic_model.forward_cal_io() or 0.0
            kv_gb = analytic_model.forward_memory_kvcache() or 0.0
            record_decode_step(
                step=step,
                flops_tflops=flops_tflops,
                io_tb=io_tb,
                kv_gb=kv_gb,
            )

            # 3) Conceptually, HF now appends this token to the cache via Cache.update(...).
            #    In the analytic model we mirror that via decode_one_token(), which bumps
            #    both the internal decode counters and the SyntheticKVCache state.
            kv_cache = analytic_model.decode_one_token()
        ```
   - The refactor makes both modes **first-class and stateful** by:
     - Adding explicit stage and shape mutators (`set_prefill_shape`, `set_decode_shape`, `set_prefill_state`, `set_decode_state`, `step_decode`) on attention/decoder layers.
     - Giving the root model a single, explicit mode switch, visible via `operation_mode`:
       - `start_prefill(...)` → `operation_mode == "prefill"`, and stats answer “what is the cost of prefill at the current `(context_len, batch_size)`?”
       - `start_decode(...)` → `operation_mode == "decode"`, and stats answer “what is the cost of decoding one token from the current KV state?”
       - `decode_one_token()` → advance KV/decode state so subsequent stats (still in `"decode"` mode) answer “cost of decoding the next token after this one,” without requiring callers to manage lengths.

2) **How do we model KV-cache size from prefill and its growth during decode? Where does the “fake kv_cache” live?**
   - KV-cache size is modeled via a shared `SyntheticKVCache` meta object in `layers/stage_cost.py`, treated as the analytic analogue of HF’s `DynamicCache`:
     - It owns per-layer logical lengths and size calculations; analytic layers never expect callers to set `context_len` / `decode_len` directly.
     - Typical usage pattern:
       - The root analytic model constructs a single `SyntheticKVCache` (or accepts one) at the start of a run.
       - Each decoder layer:
         - calls `kv_cache.get_usable_length(layer_idx)` to obtain the current KV length (mirroring `Cache.get_usable_length`), and
         - calls `kv_cache.update(layer_idx, num_new_tokens=...)` after “writing” new tokens.
       - `forward_memory_kvcache()` on attention/decoder layers can internally query `kv_cache.size_gb(layer_idx)` to expose KV size to callers.
   - Prefill KV-cache handling:
     - During analytic prefill, the model passes `kv_cache` down to decoder layers and they grow their per-layer KV lengths purely from shapes (batch size, `S_prefill`) and `update(...)` calls; callers only create the cache, not its lengths.
     - Pseudocode (prefill KV growth, conceptual):
       ```python
       kv_cache = SyntheticKVCache(batch=B, num_kv_heads=num_kv_heads, head_dim=head_dim)

       # Analytic prefill: each decoder layer sees the same kv_cache.
       for layer_idx, dec_layer in enumerate(decoder_layers):
           kv_len_before = kv_cache.get_usable_length(layer_idx)  # typically 0 on first prefill
           dec_layer.run_prefill_step(kv_cache=kv_cache, layer_idx=layer_idx, seq_len=S_prefill)
           # inside run_prefill_step, the layer calls kv_cache.update(layer_idx, num_new_tokens=S_prefill)

       # After prefill, per-layer KV sizes are given by kv_cache.size_gb(layer_idx).
       kv_gb_layer0 = kv_cache.size_gb(layer_idx=0)
       ```
   - Decode KV-cache handling:
     - During analytic decode, we reuse the same `kv_cache` object; each step:
       - layers call `get_usable_length(layer_idx)` to see `S_prefill + K_cached` for that layer, and
       - after “processing” the new token(s), call `update(layer_idx, num_new_tokens=...)` to reflect additional decode tokens in the cache.
     - Pseudocode (KV growth across decode steps):
       ```python
       # Assume kv_cache has already been populated by prefill.
       for step in range(K):
           for layer_idx, dec_layer in enumerate(decoder_layers):
               kv_len_before = kv_cache.get_usable_length(layer_idx)  # S_prefill + K_cached

               # Attention for this layer uses:
               # - S_q = 1 (decode token)
               # - S_kv = kv_len_before
               dec_layer.run_decode_step(kv_cache=kv_cache, layer_idx=layer_idx, num_new_tokens=1)
               # inside run_decode_step, the layer calls kv_cache.update(layer_idx, num_new_tokens=1)

           # At any time we can inspect KV size:
           kv_gb_layer0 = kv_cache.size_gb(layer_idx=0)
       ```
   - The “fake kv_cache” lives solely in the analytic layer stack:
     - As a `SyntheticKVCache` instance owned by the analytic model (or created by the caller once and passed in).
     - As `kv_gb` values reported by `forward_memory_kvcache()` on attention/decoder/root-model analytics, which internally consult the shared `SyntheticKVCache`.
     - It carries only scalar shape/size information and is independent of HF’s real `Cache` / `DynamicCache` objects.

3) **How do we verify analytic prefill/decode FLOP counts against the real model (like `.verify_by_impl()`)?**
   - Prefill verification:
     - Use the real `DeepseekOCRForCausalLM` (or `DeepseekV2ForCausalLM`) with `use_cache=True` and no `past_key_values` to run a prefill forward at `S_prefill` (including image tokens).
     - Wrap the call in `torch.utils.flop_counter.FlopCounterMode` to measure prefill FLOPs.
     - Build a corresponding analytic model:
       - Configure it with the same `hidden_size`, head counts, MoE config, and `S_prefill` via `start_prefill(...)`.
       - Call `forward_tensor_core_flops()` / `forward_cuda_core_flops()` on the analytic model in `"prefill"` mode to obtain theoretical prefill FLOPs and compare to the measured FLOPs within a tolerance.
      - Pseudocode (prefill verification):
        ```python
        # Real model prefill
        from torch.utils.flop_counter import FlopCounterMode
        from models.deepseek_ocr.modeling_deepseekocr import DeepseekOCRForCausalLM

        hf_model = DeepseekOCRForCausalLM.from_pretrained(model_id).cuda().eval()
        with FlopCounterMode(mods=hf_model, display=False) as counter:
            out = hf_model(
                input_ids=input_ids.cuda(),
                attention_mask=attn_mask.cuda(),
                images=images,
                images_seq_mask=images_seq_mask.cuda(),
                images_spatial_crop=images_spatial_crop.cuda(),
                use_cache=True,
                return_dict=True,
            )
        flops_prefill_measured = counter.get_total_flops()  # int

        # Analytic prefill
        analytic_model = build_analytic_dsocr_model_from_config(hf_model.config)
        analytic_model.start_prefill(context_len=S_prefill, batch_size=B, kv_cache=None)
        assert analytic_model.operation_mode == "prefill"
        flops_tc = analytic_model.forward_tensor_core_flops() or 0.0
        flops_cuda = analytic_model.forward_cuda_core_flops() or 0.0
        flops_prefill_theoretical = (flops_tc + flops_cuda) * 1.0e12

        rel_diff = abs(flops_prefill_theoretical - flops_prefill_measured) / max(
            flops_prefill_measured, 1.0,
        )
        assert rel_diff <= 0.05
        ```
   - Decode verification:
     - Run one or more decode steps using the HF generation-style flow illustrated in `about-deepseek-ocr-kv-cache.md`:
      - Start from `past_key_values` produced by prefill, and at each step:
        - Call `model.prepare_inputs_for_generation(...)` to trim `input_ids` and prepare `cache_position`.
        - Use `FlopCounterMode` around the decode forward to measure per-step FLOPs.
        - Use `Cache.get_usable_length(...)` (or equivalent) to recover the effective KV length `S_prefill + K_cached` at that step.
     - On the analytic side, mirror the same sequential flow using `start_decode(...)` and `decode_one_token()`:
       - After analytic prefill, call `start_decode(kv_cache=None)` to enter `"decode"` mode using the prefill KV cache.
       - For each real decode step:
         - In `"decode"` mode, call `forward_tensor_core_flops()` / `forward_cuda_core_flops()` on the root model to obtain the theoretical cost of the **next** token.
         - Compare these FLOPs to the measured per-step FLOPs from HF.
         - Call `decode_one_token()` once to advance analytic KV/decode state, mirroring `Cache.update(...)` in the real model.
     - Pseudocode (multi-step decode verification):
        ```python
        # Real model: prefill has already run, we have past_kv and attn_mask.

        # Analytic model: mirror prefill first.
        analytic_model = build_analytic_dsocr_model_from_config(hf_model.config)
        analytic_model.start_prefill(context_len=S_prefill, batch_size=B, kv_cache=None)

        # Switch analytic model to decode mode, reusing its internal KV cache.
        analytic_model.start_decode(kv_cache=None)
        assert analytic_model.operation_mode == "decode"

        for step in range(K):
            # 1) Real model: measure FLOPs for this decode step.
            prepared = hf_model.prepare_inputs_for_generation(
                next_ids,
                past_key_values=past_kv,
                attention_mask=attn_mask,
                use_cache=True,
            )
            with FlopCounterMode(mods=hf_model, display=False) as counter:
                out = hf_model(**prepared, return_dict=True)
            flops_decode_measured = counter.get_total_flops()

            # 2) Analytic model: in decode mode, forward_* stats describe cost
            #    of decoding the *next* token from current KV state.
            flops_tc = analytic_model.forward_tensor_core_flops() or 0.0
            flops_cuda = analytic_model.forward_cuda_core_flops() or 0.0
            flops_decode_theoretical = (flops_tc + flops_cuda) * 1.0e12

            rel_diff = abs(flops_decode_theoretical - flops_decode_measured) / max(
                flops_decode_measured, 1.0,
            )
            assert rel_diff <= 0.05

            # 3) Advance both real and analytic KV state to the next step.
            past_kv = out.past_key_values
            attn_mask = update_attention_mask(attn_mask, out)  # sketch
            next_ids = get_next_ids_from_logits(out.logits)

            analytic_model.decode_one_token()
        ```
   - Implementation-wise, this can be done in a dedicated script (e.g., `run_verify_prefill_decode.py`) that:
     - Mirrors the existing per-layer `.verify_by_impl()` style: given an HF implementation and device, it constructs analytic layers, configures their state, and asserts relative differences stay within a small tolerance.
     - Optionally exposes thin wrappers like `verify_prefill_flops_by_impl(...)` / `verify_decode_flops_by_impl(...)` on the analytic modules, reusing this logic.

**How to Refactor**
1) Baseline: identify required state
   - Attention primitive (`LlamaFlashAttention2`):
     - Query length `S_q` and context / KV length `S_kv`.
     - Stage mode: `prefill` vs `decode` (affects I/O modeling).
   - Decoder layer (`DeepseekV2DecoderLayer`):
     - Stage mode: `prefill` vs `decode`.
     - Prefill context length `S_prefill`, current decode length `K`, and effective KV length `S_prefill + K`.
   - Root model (`DeepseekOCRModel`):
     - Shared **workload state** for the run:
       - `batch_size`, `context_len` (`S_prefill`), `decode_len` (`K`), `num_decoder_layers`, and `current_stage`.
   - KV meta (`stage_cost.py`):
     - Reuse `SyntheticKVCache` as the canonical representation for KV size at:
       - end of prefill, and
       - after `K` decode tokens.

2) Make `LlamaFlashAttention2` stateful for query/context lengths
   - Before (simplified):
     ```python
     class LlamaFlashAttention2(BaseLayer):
         def __init__(self, *, seq_len: int, hidden_size: int, num_heads: int, ...):
             if seq_len <= 0:
                 raise ValueError("seq_len must be positive")
             ...
             self.m_seq_len: int = seq_len
             self.m_seq_len_q: int = seq_len
             self.m_seq_len_kv: int = seq_len
             self.m_hidden_size: int = hidden_size
             self.m_num_heads: int = num_heads
             self.m_batch_size: int = batch_size
     ```
   - After: add stage + shape mutators:
     ```python
     class LlamaFlashAttention2(BaseLayer):
         def __init__(self, *, seq_len: int, hidden_size: int, num_heads: int, ...):
             ...
             self.m_seq_len = seq_len
             self.m_seq_len_q = seq_len
             self.m_seq_len_kv = seq_len
             self.m_stage: str = "prefill"  # "prefill" | "decode"

         def set_prefill_shape(self, *, seq_len: int, batch_size: int | None = None) -> None:
             if seq_len <= 0:
                 raise ValueError("seq_len must be positive")
             if batch_size is not None:
                 if batch_size <= 0:
                     raise ValueError("batch_size must be positive")
                 self.m_batch_size = batch_size
             self.m_stage = "prefill"
             self.m_seq_len = seq_len
             self.m_seq_len_q = seq_len
             self.m_seq_len_kv = seq_len

         def set_decode_shape(self, *, context_len: int, batch_size: int | None = None) -> None:
             if context_len <= 0:
                 raise ValueError("context_len must be positive")
             if batch_size is not None:
                 if batch_size <= 0:
                     raise ValueError("batch_size must be positive")
                 self.m_batch_size = batch_size
             self.m_stage = "decode"
             # One-token query over cached context.
             self.m_seq_len_q = 1
             self.m_seq_len_kv = context_len
             # Keep m_seq_len as the context length for legacy I/O formulas.
             self.m_seq_len = context_len
     ```
   - Update I/O and memory to respect stage:
     - FLOPs already use `m_seq_len_q` / `m_seq_len_kv` and need minimal changes.
     - I/O and activation memory should treat prefill vs decode differently:
       ```python
       def forward_cal_io(self) -> float:
           b = float(self.m_batch_size)
           d_model = float(self.m_hidden_size)
           bytes_per_val = 2.0

           if self.m_stage == "prefill":
               s = float(self.m_seq_len_q)  # == seq_len_kv
           else:  # decode
               # Per-token I/O is dominated by the query path.
               s = float(self.m_seq_len_q)  # ~1

           input_bytes = b * s * d_model * bytes_per_val
           qkv_bytes = b * s * (3.0 * d_model) * bytes_per_val
           output_bytes = b * s * d_model * bytes_per_val
           total_bytes = input_bytes + qkv_bytes + output_bytes
           return (total_bytes * 8.0) / 1.0e12
       ```
     - KV memory should use `SyntheticKVCache` semantics with `context_len = m_seq_len_kv`:
       ```python
       def forward_memory_kvcache(self) -> float:
           head_dim = int(self.m_hidden_size / self.m_num_heads)
           kv_meta = SyntheticKVCache(
               batch=self.m_batch_size,
               num_kv_heads=self.m_num_key_value_heads,
               head_dim=head_dim,
               context_len=int(self.m_seq_len_kv),
               decode_len=0 if self.m_stage == "prefill" else 1,
           )
           return kv_meta.size_gb()
       ```
   - Backward compatibility:
     - If callers never call `set_prefill_shape` / `set_decode_shape`, the constructor still configures a symmetric `seq_len_q = seq_len_kv = seq_len`, matching current behavior.

3) Make `DeepseekV2DecoderLayer` manage stage and KV state instead of cloning
   - Today, decode helpers clone a new “unit layer” and override internal fields.
   - After refactor, the decoder becomes stateful and we rely on that state in combination with generic `forward_*` methods:
     ```python
     class DeepseekV2DecoderLayer(BaseLayer):
         def __init__(..., seq_len: int, ...):
             ...
             self.m_stage: str = "prefill"
             self.m_context_len: int = seq_len      # S_prefill (prefill KV length)
             # decode_len tracks how many decode tokens are already cached,
             # mirroring Cache.get_usable_length(...) semantics in the HF model.
             self.m_decode_len: int = 0             # K_cached (decode steps completed)
             self._sync_sub_layers_prefill()

         def _sync_sub_layers_prefill(self) -> None:
             self.m_input_layernorm.set_seq_len(self.m_context_len)
             self.m_post_attention_layernorm.set_seq_len(self.m_context_len)
             self.m_self_attn.set_prefill_shape(
                 seq_len=self.m_context_len,
                 batch_size=self.m_batch_size,
             )
             self.m_mlp.set_seq_len(self.m_context_len)

         def _sync_sub_layers_decode(self) -> None:
             # Effective KV length seen by the *next* decode step is:
             #   S_kv = S_prefill + K_cached,
             # where K_cached == self.m_decode_len reflects tokens already in the cache.
             context_len = self.m_context_len + self.m_decode_len
             self.m_self_attn.set_decode_shape(
                 context_len=context_len,
                 batch_size=self.m_batch_size,
             )
             # Norms/MLP can continue to use single-token seq_len if desired.

         def set_prefill_state(self, *, context_len: int, batch_size: int | None = None) -> None:
             self.m_stage = "prefill"
             self.m_context_len = context_len
             self.m_decode_len = 0
             if batch_size is not None:
                 self.m_batch_size = batch_size
             self._sync_sub_layers_prefill()

         def set_decode_state(self, *, context_len: int, decode_len: int, batch_size: int | None = None) -> None:
             self.m_stage = "decode"
             self.m_context_len = context_len
             self.m_decode_len = decode_len
             if batch_size is not None:
                 self.m_batch_size = batch_size
             self._sync_sub_layers_decode()

         def step_decode(self, *, batch_size: int | None = None) -> None:
             """Mark that one more decode token has been cached and resync.

             Callers should:
             - compute per-token stats using the *current* state
               (m_decode_len == number of tokens already in the cache), then
             - call step_decode() after conceptually appending the new token.
             This mirrors the HF flow where Cache.get_usable_length() is
             queried before Cache.update(...) during generation.
             """
             self.set_decode_state(
                 context_len=self.m_context_len,
                 decode_len=self.m_decode_len + 1,
                 batch_size=batch_size,
             )
     ```
   - We keep the decoder class API minimal:
     - It exposes state mutators (`set_prefill_state`, `set_decode_state`, `step_decode`) plus the standard `BaseLayer` analytics (`forward_tensor_core_flops`, `forward_cal_io`, `forward_memory_*`).
     - Callers are expected to:
       1) set the desired stage/KV state, then
       2) call the standard analytics methods and aggregate numbers in whatever shape they need (e.g., dicts, small dataclasses) in their own code, rather than relying on class-level helpers.

4) Add workload / stage control APIs on `DeepseekOCRModel`
   - Introduce a shared notion of run state and stage-aware entrypoints:
     ```python
     class DeepseekOCRModel(BaseLayer):
         def __init__(self) -> None:
             ...
             self.m_stage: str = "prefill"  # "prefill" | "decode"
             self.m_context_len: int = 0
             self.m_decode_len: int = 0
             self.m_batch_size: int = 1
             self.m_kv_cache: SyntheticKVCache | None = None

         @property
         def operation_mode(self) -> str:
             """Return the current analytic operation mode.

             - 'prefill': forward_* stats describe the cost of running prefill
               at the configured context_len/batch_size.
             - 'decode': forward_* stats describe the cost of decoding one
               additional token from the current KV state.
             """
             return self.m_stage

         def start_prefill(
             self,
             *,
             context_len: int,
             batch_size: int,
             kv_cache: SyntheticKVCache | None = None,
         ) -> SyntheticKVCache:
             """Switch to prefill mode and run analytic prefill once.

             After this call, forward_* stats answer: 'cost of prefill at
             the given context_len and batch_size'. The returned kv_cache
             reflects the KV state after prefill and is stored on the model.
             """
             if context_len <= 0 or batch_size <= 0:
                 raise ValueError("context_len and batch_size must be positive")

             self.m_stage = "prefill"
             self.m_context_len = context_len
             self.m_decode_len = 0
             self.m_batch_size = batch_size

             if kv_cache is None:
                 kv_cache = self.m_kv_cache or SyntheticKVCache.from_config(self._config)
             self.m_kv_cache = kv_cache

             if self.m_decoder_layer is not None and hasattr(self.m_decoder_layer, "set_prefill_state"):
                 self.m_decoder_layer.set_prefill_state(
                     context_len=context_len,
                     batch_size=batch_size,
                 )
             # Decoder prefill logic (no real tensors): layers will consult and
             # update self.m_kv_cache as needed.
             return kv_cache

         def start_decode(self, kv_cache: SyntheticKVCache | None = None) -> SyntheticKVCache:
             """Switch to decode mode.

             After this call, forward_* stats answer: 'cost of decoding the
             next token from the current KV state'.
             """
             self.m_stage = "decode"
             # Reuse prefill context length and batch size.
             if self.m_context_len <= 0:
                 raise ValueError("start_decode requires a valid prefill context")

             if kv_cache is not None:
                 self.m_kv_cache = kv_cache
             if self.m_kv_cache is None:
                 raise ValueError("start_decode requires an existing KV cache")

             # Ensure decoder is in decode mode with the current context_len.
             if self.m_decoder_layer is not None and hasattr(self.m_decoder_layer, "set_decode_state"):
                 self.m_decoder_layer.set_decode_state(
                     context_len=self.m_context_len,
                     decode_len=self.m_decode_len,
                     batch_size=self.m_batch_size,
                 )
             return self.m_kv_cache

         def decode_one_token(self) -> SyntheticKVCache:
             """Advance decode state by one token.

             Requires an attached KV cache. After this call, forward_* stats
             answer the cost of decoding the *next* token from the updated
             KV state.
             """
             if self.m_stage != "decode":
                 raise ValueError("decode_one_token requires decode mode; call start_decode first")
             if self.m_kv_cache is None:
                 raise ValueError("decode_one_token requires an attached KV cache")

             # Advance logical decode length.
             self.m_decode_len += 1
             if self.m_decoder_layer is not None and hasattr(self.m_decoder_layer, "step_decode"):
                 self.m_decoder_layer.step_decode(batch_size=self.m_batch_size)

             # Decoder layers are expected to update self.m_kv_cache as part of
             # their analytic decode step (e.g., via SyntheticKVCache.update).
             return self.m_kv_cache
     ```
   - Use the new model-level state for simulated runs:
     ```python
     def simulate_run(self, *, decode_tokens: int) -> dict[str, float]:
         # Prefill stage.
         self.start_prefill(context_len=S_prefill, batch_size=1, kv_cache=None)
         flops_prefill = (self.forward_tensor_core_flops() or 0.0) + (
             self.forward_cuda_core_flops() or 0.0
         )

         # Decode stage: reuse the same model but advance decode_len.
         self.start_decode(kv_cache=None)
         flops_decode = 0.0
         for _ in range(decode_tokens):
             # Stats correspond to the next token from current KV state.
             flops_tc = self.forward_tensor_core_flops() or 0.0
             flops_cuda = self.forward_cuda_core_flops() or 0.0
             flops_decode += flops_tc + flops_cuda
             self.decode_one_token()

         return {
             "prefill_flops_tflops": flops_prefill,
             "decode_flops_tflops": flops_decode,
         }
     ```

5) Update `HOLISTIC_ANALYSIS.md` simulated run to use stateful APIs
     - Replace the current “Option A / Option B” pseudo-code that:
       - re-instantiates `DeepseekV2DecoderLayer` for prefill vs decode, and
       - treats KV size via manual `seq_len` arithmetic,
       with examples that:
       - construct one `DeepseekOCRModel`.
       - call `start_prefill(context_len=..., batch_size=..., kv_cache=...)` and use `forward_*` on the model to derive prefill stats.
       - call `start_decode(kv_cache=...)`, then in a loop:
         - use `forward_*` to get per-token decode stats for the current KV state, and
         - call `decode_one_token()` to advance KV/decode state.
   - Document how `SyntheticKVCache` is used inside `DeepseekV2DecoderLayer` / `LlamaFlashAttention2` and `DeepseekOCRModel` (for example, as an internal helper to implement `forward_memory_kvcache`) rather than manually reconstructing KV formulas in the note.

6) Wire stateful analytics into `DeepseekOCRStaticAnalyzer` (gradual migration)
   - In `src/llm_perf_opt/runners/dsocr_analyzer.py`:
     - After constructing the analytic vision and decoder layers, build a `DeepseekOCRModel` and configure a workload:
       ```python
       analytic_model = DeepseekOCRModel.from_layers(
           vision_stack,
           decoder_layer,
           num_decoder_layers=spec.num_decoder_layers,
       )
       analytic_model.start_prefill(
           context_len=profile.seq_len,
           batch_size=1,
           kv_cache=None,
       )
       ```
     - When populating prefill/decode entries in `AnalyticModelReport`, compute fields like `total_flops_tflops`, `total_io_tb`, `weights_gb`, `activations_gb`, and `kv_gb` directly from `forward_*` calls instead of introducing new class-level aggregation types.
   - Keep the existing path (using `forward_*` directly) as a fallback during migration; once confidence is high, de-duplicate to a single stateful path.

7) Testing strategy
   - Extend `tests/unit/deepseek_ocr/test_analytic_layers_scaling.py`:
     - Add tests that:
       - Compare FLOPs/I/O/memory for:
         - a freshly constructed `LlamaFlashAttention2(seq_len=S)` vs `set_prefill_shape(seq_len=S)` on the same instance.
         - `DeepseekV2DecoderLayer` prefill stats before/after calling `set_prefill_state` (should be consistent).
       - Verify decode behavior:
         - For a decoder layer configured in decode mode, FLOPs from `forward_tensor_core_flops()` / `forward_cuda_core_flops()` match the previous clone-based implementation within a small relative tolerance.
         - KV-cache size from `forward_memory_kvcache()` grows as expected with `decode_len`, and matches `SyntheticKVCache(...).size_gb()` for the same `(context_len, decode_len)`.
   - (Optional) Add a small integration test for `DeepseekOCRModel.simulate_run` to confirm:
     - Prefill flops from `simulate_run` match `estimate_prefill_cost().flops_tflops`.
     - Decode flops are approximately `decode_tokens * per_token_flops`.

**Impact Analysis**
- Functional impact:
  - Enables a **single analytic model instance** to model:
    - Prefill (full context) and
    - Decode (per-token over evolving KV cache),
    by updating internal state instead of re-instantiating layers.
  - Makes simulated runs and hooked runs described in `HOLISTIC_ANALYSIS.md` easier to implement and reason about.
- Compatibility:
  - Constructor signatures for `LlamaFlashAttention2`, `DeepseekV2DecoderLayer`, and `DeepseekOCRModel` remain unchanged.
  - Existing callers that never use the new `set_*` / `step_*` methods will continue to observe the same behavior (subject to minor I/O modeling fixes if we update `forward_cal_io()` to use `S_q`).
- Risks:
  - **State management bugs**: failing to keep `m_context_len`, `m_decode_len`, `m_seq_len_q`, and `m_seq_len_kv` in sync can lead to inconsistent stats.
    - Mitigation: centralize sync logic in small helpers (`_sync_sub_layers_prefill`, `_sync_sub_layers_decode`), and add unit tests that assert invariants (e.g., `m_seq_len_q == m_seq_len_kv` in prefill mode).
  - **Reentrancy / concurrency**: stateful analytic layers are not thread-safe if reused across concurrent runs.
    - Mitigation: document analytic models as single-run objects; construct separate instances per run / session in analyzer code.
  - **Migration complexity**: analyzer and docs need to be updated to prefer stateful helpers.
    - Mitigation: migrate in steps (keep old path as fallback), and compare outputs from both paths during development to catch regressions.

**Expected Outcome**
- A DeepSeek-OCR analytic stack that:
  - Mirrors the real model’s **prefill + decode** flow at the level of layers and KV-cache evolution.
  - Exposes **stateful APIs** (`set_prefill_shape`, `set_decode_shape`, `set_prefill_state`, `set_decode_state`, `step_decode`, `configure_workload`, `simulate_run`) that allow:
    - separate-stage stats (prefill vs per-token decode), and
    - KV-aware simulated runs of arbitrary length `(S_prefill, K)`.
  - Integrates naturally with `DeepseekOCRStaticAnalyzer` and future hook-based profilers.
- Downstream tools can:
  - Compute theoretical prefill/decode FLOPs, I/O, and memory with explicit stage boundaries.
  - Model KV-cache growth over decode and incorporate it into capacity planning and MFU analyses without duplicating formulas.

**Implementation Summary**
- `extern/modelmeter/models/deepseek_ocr/layers/stage_cost.py`
  - Extended `SyntheticKVCache` to behave more like HF `DynamicCache` for analytic purposes:
    - Added `context_len=0` default so callers can construct a cache before prefill.
    - Added `get_usable_length(layer_idx)` and `update(layer_idx, num_new_tokens)` to model KV growth (`S_prefill + K`) while keeping a shared length across layers.
    - Kept `size_bytes()` / `size_gb()` as the single source of truth for KV size; `estimate_kv_cache_size_gb(...)` remains a thin convenience wrapper.

- `extern/modelmeter/models/deepseek_ocr/layers/llama/llama_flash_attention2.py`
  - Made the attention primitive explicitly stageful:
    - New field `m_stage: "prefill" | "decode"` and property `operation_mode`.
    - New mutators:
      - `set_prefill_shape(seq_len, batch_size=None)` → sets `m_seq_len_q = m_seq_len_kv = seq_len` and marks stage as `"prefill"`.
      - `set_decode_shape(context_len, batch_size=None)` → sets `m_seq_len_q = 1`, `m_seq_len_kv = context_len`, `m_seq_len = context_len`, and marks stage as `"decode"`.
  - Updated `forward_cal_io()` to use the stage-aware query length (`m_seq_len_q`) instead of always using a single `m_seq_len`, so per-token decode I/O is smaller than full-context prefill I/O while FLOP formulas continue to use `m_seq_len_q` / `m_seq_len_kv`.

- `extern/modelmeter/models/deepseek_ocr/layers/decoder/deepseek_v2_decoder_layer.py`
  - Added decoder- and KV-state fields:
    - `m_stage: "prefill" | "decode"`, `m_context_len`, `m_decode_len`, and `m_kv_cache: Optional[SyntheticKVCache]`.
    - Property `operation_stage` exposes the current stage.
  - Introduced internal sync helpers:
    - `_sync_sub_layers_prefill()` configures RMSNorm, attention (`set_prefill_shape`), and MLP/MoE to use `seq_len = m_context_len` and `batch_size = m_batch_size`.
    - `_sync_sub_layers_decode()` configures RMSNorm and MLP/MoE for single-token `seq_len = 1` and attention via `set_decode_shape(context_len=m_context_len + m_decode_len, batch_size=m_batch_size)`.
  - Added stage mutators that replace the earlier clone-based decode helpers:
    - `set_prefill_state(context_len, batch_size=None, kv_cache=None)`:
      - Sets `m_stage="prefill"`, `m_context_len=context_len`, `m_decode_len=0`, updates `m_kv_cache`, and calls `_sync_sub_layers_prefill()`.
      - If a `SyntheticKVCache` is attached, aligns its `context_len`/`decode_len` with the new state.
    - `set_decode_state(context_len, decode_len, batch_size=None, kv_cache=None)`:
      - Sets `m_stage="decode"`, updates `m_context_len`, `m_decode_len`, `m_kv_cache`, and calls `_sync_sub_layers_decode()`, keeping the cache in sync.
    - `step_decode(batch_size=None)`:
      - Requires `"decode"` stage, optionally adjusts batch size.
      - Calls `m_kv_cache.update(layer_idx=0, num_new_tokens=1)` when a cache is attached, or increments `m_decode_len` directly, then re-syncs sublayers via `_sync_sub_layers_decode()`.
  - Removed `estimate_prefill_cost()` and `estimate_decode_cost_per_token(...)` so callers follow the single pattern:
    - Set stage and KV state via the mutators, then call the standard `forward_*` analytics.

- `extern/modelmeter/models/deepseek_ocr/layers/core/deepseek_ocr_model.py`
  - Extended root model state:
    - Added `m_stage: "prefill" | "decode"`, `m_context_len`, `m_decode_len`, `m_batch_size`, and `m_kv_cache: Optional[SyntheticKVCache]`.
    - Exposed `operation_mode` property to indicate the current analytic mode to callers.
  - Implemented stage/kv-control APIs on the root model:
    - `start_prefill(context_len, batch_size, kv_cache=None) -> SyntheticKVCache`:
      - Validates inputs and requires a configured decoder.
      - Constructs or adopts a `SyntheticKVCache` using decoder dimensions (`num_key_value_heads`, `hidden_size / num_heads`), sets `context_len`/`decode_len`, and stores it in `m_kv_cache`.
      - Updates model-level state (`m_stage="prefill"`, `m_context_len`, `m_decode_len=0`, `m_batch_size`) and calls `decoder_layer.set_prefill_state(..., kv_cache=m_kv_cache)`.
    - `start_decode(kv_cache=None) -> SyntheticKVCache`:
      - Requires a valid prefill context and decoder.
      - Adopts a provided cache or reuses `m_kv_cache`, and mirrors `decode_len` from it.
      - Sets `m_stage="decode"` and calls `decoder_layer.set_decode_state(context_len=m_context_len, decode_len=m_decode_len, batch_size=m_batch_size, kv_cache=m_kv_cache)`.
    - `decode_one_token() -> SyntheticKVCache`:
      - Requires `"decode"` mode and an attached cache.
      - Delegates step advancement to `decoder_layer.step_decode(batch_size=m_batch_size)`, then synchronizes `m_decode_len` from `m_kv_cache.decode_len`.
  - Removed StageCost-based helpers (`estimate_prefill_cost`, `estimate_decode_cost_per_token`) in favor of the stateful mode + `forward_*` pattern.

- `src/llm_perf_opt/runners/dsocr_analyzer.py`
  - After building the analytic model via `DeepseekOCRModel.from_layers(...)`, the analyzer now:
    - Calls `model_layer.start_prefill(context_len=workload.seq_len, batch_size=1, kv_cache=None)` to ensure that:
      - Model-level `forward_*` stats correspond to a prefill workload, and
      - Decoder and attention are in a consistent prefill state before metrics are collected.
    - Wraps this in a `try/except` and logs a warning if prefill configuration fails, so existing behavior remains robust.
  - The rest of `_build_module_nodes_and_metrics` still uses `forward_*` metrics on the configured layers, but those now reflect the stateful prefill semantics.

- `tests/unit/deepseek_ocr/test_analytic_layers_scaling.py`
  - Added stateful behavior tests on top of existing scaling checks:
    - `test_llama_flash_attention_prefill_vs_decode_modes()`:
      - Verifies that in decode mode (`set_decode_shape`), attention I/O and FLOPs per token are non-negative and do not exceed the prefill values at the same context length.
    - `test_decoder_layer_prefill_vs_decode_modes()`:
      - Configures a decoder layer for prefill and decode (with a `SyntheticKVCache`) and asserts that decode-mode FLOPs/I/O/activation memory are non-negative and less than or equal to prefill.
    - `test_decoder_layer_step_decode_updates_kv_cache()`:
      - Asserts that `step_decode()` advances `SyntheticKVCache.total_len()` by one token, confirming that analytic KV state evolves across decode steps.

**References**
- Code and docs
  - `extern/modelmeter/models/deepseek_ocr/layers/llama/llama_flash_attention2.py`
  - `extern/modelmeter/models/deepseek_ocr/layers/decoder/deepseek_v2_decoder_layer.py`
  - `extern/modelmeter/models/deepseek_ocr/layers/core/deepseek_ocr_model.py`
  - `extern/modelmeter/models/deepseek_ocr/layers/stage_cost.py`
  - `extern/modelmeter/models/deepseek_ocr/HOLISTIC_ANALYSIS.md`
  - `extern/modelmeter/models/deepseek_ocr/LAYER_IO_ESTIMATION_GUIDE.md`
  - `src/llm_perf_opt/runners/dsocr_analyzer.py`
- Third-party libraries (Context7 IDs)
  - `/pytorch/pytorch`
  - `/huggingface/transformers`
