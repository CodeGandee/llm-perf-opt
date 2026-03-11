## 1. Device-Parameterized Sweep Foundation

- [ ] 1.1 Add a thin Wan2.1-local device resolver under `extern/modelmeter/models/` that selects `ngu800p`, `dv100`, `dv200`, or `dv300`, reuses the shared device fields directly, and derives only the still-missing metadata fields.
- [ ] 1.2 Update the Wan2.1 concurrency sweep entrypoint to accept a device selector and reject unsupported device names with a clear error.
- [ ] 1.3 Refactor the sweep output layout and metadata so each run is written under a device-scoped directory and records the normalized device assumptions used for the run.
- [ ] 1.4 Preserve the current DP batch/device semantics in the generalized sweep and verify that `effective_gpus=min(batch_size, device_num)` is still reported correctly.
- [ ] 1.5 Verify that all implementation changes for this change remain within `extern/modelmeter/models/` and that shared modules outside that subtree are used read-only.

## 2. Sweep Execution and Baseline Compatibility

- [ ] 2.1 Run the generalized sweep on `ngu800p` and confirm the resulting NGU curves and summary metrics remain consistent with the current baseline workflow.
- [ ] 2.2 Run the generalized sweep on `dv100`, `dv200`, and `dv300` and verify that each produces complete `results.json`, `results.csv`, and `summary.md` artifacts.

## 3. Generic Reporting and Comparison

- [ ] 3.1 Refactor the Wan2.1 sweep figure generator to read device metadata from results and emit device-specific filenames, labels, and titles instead of NGU-only naming.
- [ ] 3.2 Add compatibility checks that reject consolidated DV stakeholder-report inputs when workload structure or precision assumptions do not match.
- [ ] 3.3 Implement a consolidated `stakeholder-report.en.md` generator that covers `dv100`, `dv200`, and `dv300` in separate sections with NGU-style content.

## 4. Documentation and Artifact Publishing

- [ ] 4.1 Update the Wan2.1 reports documentation to describe the device-selectable sweep workflow, artifact locations, and supported device profiles.
- [ ] 4.2 Generate the first DV stakeholder artifact set for `dv100`, `dv200`, and `dv300`, including one consolidated `stakeholder-report.en.md`, and record the expected report/figure outputs for future reruns.
