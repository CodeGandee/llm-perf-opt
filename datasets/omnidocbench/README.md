OmniDocBench Dataset

This directory links the OmniDocBench dataset into the workspace following the
project’s dataset layout. The `source-data` symlink points to the external
storage location.

- Root: `datasets/omnidocbench/`
- Source data: `datasets/omnidocbench/source-data -> /workspace/datasets/OpenDataLab___OmniDocBench`
- Variants (optional): create subfolders (e.g., `subset-1k/`) alongside `source-data/`

Do not commit large data files; keep them outside the repository and access
them via symlinks.

Bootstrapping
- Preferred: `datasets/omnidocbench/bootstrap.sh --yes`
  - Config: `datasets/omnidocbench/bootstrap.yaml`
  - Prompts to extract `images.zip`/`pdfs.zip` into the target directory.
- Or run workspace bootstrap: `./bootstrap.sh --yes`
- Manual symlink: `ln -s /workspace/datasets/OpenDataLab___OmniDocBench datasets/omnidocbench/source-data`

Expected Directory Tree (source-data)

```
source-data/
├── .cache/
├── images/           # or images.zip (extract if missing)
├── pdfs/             # or pdfs.zip (extract if missing)
├── README.md
├── README_EN.md
├── metafile.yaml
└── OmniDocBench.json
```

Notes
- If only `images.zip`/`pdfs.zip` are present, extract them so fallback globs
  like `images/*.png` work out-of-the-box.
