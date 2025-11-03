OmniDocBench Dataset

This directory links the OmniDocBench dataset into the workspace following the
project’s dataset layout. The `source-data` symlink points to the external
storage location.

- Root: `datasets/omnidocbench/`
- Source data: `datasets/omnidocbench/source-data -> /data2/datasets/OmniDocBench`
- Variants (optional): create subfolders (e.g., `subset-1k/`) alongside `source-data/`

Do not commit large data files; keep them outside the repository and access
them via symlinks.

First-level Directory Tree (source-data)

```
source-data/
├── .cache/
├── images/
├── pdfs/
├── README.md
├── README_EN.md
├── metafile.yaml
└── OmniDocBench.json
```
