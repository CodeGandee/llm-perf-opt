nvtx-based ncu profiling, extend current ncu profiling facility to let to support nvtx markers, so that only those nvtx-marked regions are profiled by ncu, and the reporting should aggregate the results by nvtx ranges. 

Specifically, given nvtx-marked regions in the code, the ncu profiling should be able to:
- report the ncu metrics/sections for each region, including nested regions
- report ncu metrics/sections for selected kernels within each region
- metrics/sections must be configurable by user, through hydra configs (similar to current ncu profiling facility)

Note:
- this is an extension of current ncu profiling facility, which already supports kernel-level profiling and reporting, so we must retain the overall structure of current ncu profiling facility and usage pattern.