# Code Review — Stage 2 Deep Profiling (NVTX gating for Nsight Systems)

Scope
- Goal: Review NVTX-gated Nsight Systems/Compute pipeline for Stage 2 and assess the open issue: nvtx_capture=decode sometimes yields “No reports were generated” while prefill works.
- Files reviewed:
  - src/llm_perf_opt/runners/deep_profile_runner.py:118–139
  - src/llm_perf_opt/profiling/vendor/nsys.py:11–74, 119–129
  - src/llm_perf_opt/profiling/vendor/ncu.py:18–84
  - src/llm_perf_opt/profiling/nsys_stats.py (parsers vs vendor helpers)
  - src/llm_perf_opt/profiling/nvtx_utils.py (range labels)
  - src/llm_perf_opt/runners/dsocr_session.py (NVTX ranges)
  - conf/profiling/nsys/nsys.default.yaml:1–8
  - conf/runner/stage2.yaml:37–40

Summary
- NVTX ranges are present and consistent: the session emits default-domain ranges "prefill" and "decode".
- Nsight Systems command builder correctly applies `--capture-range=nvtx`, `--nvtx-capture=<expr>`, and injects `--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0` when gating is on (good alignment with NVIDIA guidance).
- Nsight Compute command builder uses `--nvtx --nvtx-include <expr>` (default `decode*`), optionally seeds `--kernel-name` from NSYS stats (good).
- A likely root cause for “No reports were generated” when `nvtx_capture=decode` is a bad default path in the runner combined with NVTX gating semantics: the fallback value “range” is invalid and can silently miss capture.

Detailed Findings

1) Bad nvtx_capture fallback (‘range’) in runner
- Reference: src/llm_perf_opt/runners/deep_profile_runner.py:125
  - `nvtx_capture=str(getattr(getattr(cfg, "nsys", {}), "nvtx_capture", "range")) if gating_nvtx_nsys else "none",`
- Conf default is `prefill` (conf/profiling/nsys/nsys.default.yaml:8). If users don’t provide cfg.nsys.nvtx_capture, runner falls back to the literal string "range" rather than a valid NVTX label (e.g., "prefill" or "decode").
- Effect: `nsys profile --capture-range=nvtx --nvtx-capture=range ...` will only start when a range named exactly "range" opens — which never happens — so Nsight Systems produces “No reports were generated.” This is consistent with the reported symptoms for “decode sometimes produces no report.”

2) Nsight Systems gating flags are otherwise correct
- Reference: src/llm_perf_opt/profiling/vendor/nsys.py:65–72
  - Adds `--capture-range=nvtx` when enabled and `--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0` (per NVIDIA forum guidance) and `--nvtx-capture=<expr>`.
- This aligns with: NVIDIA Docs (User Guide → CLI Profile Options) and NVIDIA forum (using capture-range=nvtx).

3) NVTX range names and domains
- Reference: src/llm_perf_opt/profiling/nvtx_utils.py
  - Default-domain ranges "prefill" and "decode" exist; Stage 2 domain labels (LLM@prefill, LLM@decode_all) also exist but the session currently uses default-domain helpers via dsocr_session.
- Reference: src/llm_perf_opt/runners/dsocr_session.py (uses prefill_range/decode_range)
- This matches the capture expressions documented in conf.

4) Nsight Compute configuration and stats use
- Reference: src/llm_perf_opt/profiling/vendor/ncu.py:69–76
  - `--nvtx --nvtx-include <expr>` is applied when gating is enabled; kernel regex from NSYS stats is a reasonable refinement.

5) Duplicate NSYS stats helpers
- Reference: src/llm_perf_opt/profiling/nsys_stats.py vs vendor/nsys.py
  - Both define stats/export helpers; the runner uses the vendor versions for commands but relies on nsys_stats.py for CSV parsing. This can be okay if role separation is documented to avoid drift.

Impact and Risk
- The single largest correctness risk is the fallback "range" in the runner, which can fully suppress NSYS reports under NVTX gating, mimicking the reported issue. All else (env var, capture flags, NVTX labels) is correct and aligns with NVIDIA guidance. The fallback can also hide problems intermittently, depending on config presence.

Recommendations (non-breaking)
1) Fix invalid default for nvtx_capture in runner
- Change fallback from "range" to a valid label consistent with defaults, e.g. "prefill".
  - src/llm_perf_opt/runners/deep_profile_runner.py:125
  - Rationale: Avoid silent miss and align with conf nsight default.

2) Offer optional NVTX domain filters for robustness
- Add optional pass-through for `--nvtx-domain-include=default` and/or `--nvtx-domain-exclude` in vendor/nsys.py and wire via cfg to reduce matching ambiguity across libraries.
  - Users can set it to `default` since ranges come from Python default domain.

3) Emit profiler argv for reproducibility
- Record constructed NSYS and NCU commands into artifacts (e.g., `nsys/cmd.txt`, `ncu/cmd.txt`) right before subprocess calls. This greatly simplifies field triage for gating issues.

4) Document gating troubleshooting in conf
- In conf/profiling/nsys/nsys.default.yaml, add comments:
  - To verify capture: set `gating_nvtx=false` (capture everything).
  - Try alternative expressions: `nvtx_capture=prefill` vs `decode`.
  - Optionally set `nvtx_domain_include=default` when supported.
  - Consider `--capture-range-end=repeat:1` to capture only first matching range in multi-iteration flows.

5) Minor hygiene
- Clarify in nsys_stats.py that command builders live in vendor/nsys.py to avoid divergence; keep only CSV parsing here.

Validation Ideas
- Manual runs (GPU + Nsight required):
  - `+nsys.gating_nvtx=true +nsys.nvtx_capture=prefill` → expect a report.
  - `+nsys.gating_nvtx=true +nsys.nvtx_capture=decode` → expect a report if decode executes and emits NVTX; otherwise no report (by design).
  - `+nsys.gating_nvtx=false` → expect a report (no gating risk).
- With domain include (when added): `+nsys.nvtx_domain_include=default`.
- Optionally cap to first match with `--capture-range-end=repeat:1`.

Code References (exact)
- deep_profile_runner fallback causing silent miss:
  - src/llm_perf_opt/runners/deep_profile_runner.py:125
    - `nvtx_capture=... "range"` (should be a valid label such as "prefill").
- Nsight Systems env var + capture flags (correct):
  - src/llm_perf_opt/profiling/vendor/nsys.py:67–72
- Conf default (consistent with "prefill"):
  - conf/profiling/nsys/nsys.default.yaml:8

Online-Examples (how-to/usage)
- Nsight Systems NVTX gating and capture end
  - NVIDIA Docs — User Guide (CLI options; `--capture-range`, `--nvtx-capture`, `--capture-range-end`)
    - https://docs.nvidia.com/nsight-systems/UserGuide/index.html
  - HackMD quickstart with NVTX gating + capture-range-end
    - https://hackmd.io/@epicurehack/rkcPP59xyg
  - NVIDIA forum — `--capture-range=nvtx` and NSYS_NVTX_PROFILER_REGISTER_ONLY
    - https://forums.developer.nvidia.com/t/using-capture-range-nvtx/254091

API-Doc
- NVTX Python API (push_range/pop_range, domains)
  - https://nvidia.github.io/NVTX/python/reference.html
- Nsight Compute CLI — NVTX include filtering
  - https://docs.nvidia.com/nsight-compute/2022.2/NsightComputeCli/index.html#nvtx-filtering

Proposed Next Steps (non-code)
- Confirm the fallback issue by inspecting produced NSYS argv under default config. If `--nvtx-capture=range` appears, the fix is clear.
- After correcting fallback to "prefill", re-test the original setup where `decode` previously failed; if still failing, evaluate:
  - Does the decode path actually execute and push NVTX on this run?
  - Try adding `--nvtx-domain-include=default` once wired.
  - Try `--capture-range-end=repeat:1` to capture only the first decode instance.
  - Temporarily set `gating_nvtx=false` to ensure CUPTI capture is functioning.

Appendix — Why decode gating can be fragile even with correct defaults
- NVTX gate starts only when the named range opens. If the decode loop is conditional (zero tokens, early exit) the range may never open.
- Name matching is exact; domains must match if specified. Default domain vs custom domain matters.
- Registered-string behavior: although we pass `NSYS_NVTX_PROFILER_REGISTER_ONLY=0`, per NVIDIA guidance, some environments still favor registered strings; domain include can reduce surprises.

