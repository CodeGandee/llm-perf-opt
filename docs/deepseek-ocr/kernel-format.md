DeepSeek‑OCR Kernel YAML — Format & Generation
=============================================

Purpose
- Provide a portable, human‑readable index of the most impactful GPU kernels seen when profiling DeepSeek‑OCR with Nsight Systems.
- Each kernel entry includes a raw function name, a friendly name, source library, and shape metadata that captures both operator‑level semantics and tile‑level computation where applicable.

Where the file lives
- Path: `docs/deepseek-ocr/kernels.yaml`

How to generate (from fresh Nsight data)
1) Collect profiling data (per‑stage + combined):
   - Recommended (RTX 5090 env):
     - `pixi install -e rtx5090`
     - `pixi run -e rtx5090 nsys-dsocr-all -- --gpu=0`
   - This writes a run directory like `tmp/nsys-profile/<timestamp>/` with:
     - Per‑stage CSVs: `per-stage/<stage>/nsys/summary_cuda_gpu_kern_sum.csv` for stages `{prefill, decode, sam, clip, projector}`
     - Combined CSV: `all-stage/nsys/summary_cuda_gpu_kern_sum.csv`

2) Build/update `kernels.yaml` from the latest run:
   - One‑liner (run at repo root, uses latest `tmp/nsys-profile/*` and writes `docs/deepseek-ocr/kernels.yaml`):
   - `pixi run -e rtx5090 python - <<'PY'
import csv, pathlib, json, re
from collections import defaultdict

base = sorted([p for p in pathlib.Path('tmp/nsys-profile').glob('*') if p.is_dir()])[-1]
stages = ['prefill','decode','sam','clip','projector']
csfiles = []
for s in stages:
    p = base/ 'per-stage'/ s / 'nsys' / 'summary_cuda_gpu_kern_sum.csv'
    if p.exists(): csfiles.append(p)
comb = base / 'all-stage' / 'nsys' / 'summary_cuda_gpu_kern_sum.csv'
if comb.exists(): csfiles.append(comb)

rows = []
for path in csfiles:
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        rs = [row for row in r if row.get('Name') and row.get('Total Time (ns)', '').isdigit()]
        rs.sort(key=lambda x: int(x['Total Time (ns)']), reverse=True)
        rows.extend(rs[:20])

# Unique by Name, keep aggregate time for sort
agg = defaultdict(int)
for r in rows:
    agg[r['Name']] += int(r['Total Time (ns)'])
ordered = sorted(agg.keys(), key=lambda k: agg[k], reverse=True)

# Helpers to infer metadata (same rules described below)

def cutlass_tile_params(raw:str):
    m = re.search(r'_(\d+)x(\d+)_([0-9]+)x([0-9]+)', raw)
    if m:
        return {'tile':[int(m.group(1)), int(m.group(2))], 'k_tile':[int(m.group(3)), int(m.group(4))]}
    m2 = re.search(r'_(\d+)x(\d+)(?:_|>)', raw)
    if m2:
        return {'tile':[int(m2.group(1)), int(m2.group(2))]}
    return None

def tiled_gemm(params):
    if not params: return None
    M,N = params.get('tile',[None,None])
    kt = params.get('k_tile')
    K = kt[0] if kt else None
    if None in (M,N,K): return None
    return [[f'({M}, {K})', f'({K}, {N})', '->', f'({M}, {N})']]

def infer(row_name:str):
    raw = row_name
    lower = raw.lower()
    friendly = 'CUDA kernel'
    source = 'CUDA'
    generic = [['(…)', '->', '(…)']]
    fixed = None
    tiled = None
    desc = 'CUDA device kernel.'

    if 'cublas' in lower or 'gemvx' in lower:
        source = 'cuBLAS'
        if 'gemv' in lower:
            friendly = 'cuBLAS GEMV (matrix-vector)'
            generic = [['(M, K)', '(K,)', '->', '(M,)']]
        else:
            friendly = 'cuBLAS GEMM'
            generic = [['(M, K)', '(K, N)', '->', '(M, N)']]
        desc = 'General-purpose GEMV/GEMM; tile not encoded in name.'
        return friendly, source, generic, fixed, tiled, desc

    if 'cutlass' in raw and ('gemm' in lower or 'wmma' in lower):
        source = 'CUTLASS'
        friendly = 'CUTLASS GEMM (Tensor Core)'
        generic = [['(M, K)', '(K, N)', '->', '(M, N)']]
        params = cutlass_tile_params(raw)
        tiled = tiled_gemm(params)
        desc = 'Tensor Core GEMM; tile sizes present in name (tiled shows one tile-level multiply).'
        return friendly, source, generic, fixed, tiled, desc

    if 'flash::flash_fwd_splitkv_kernel' in raw:
        source = 'FlashAttention'
        friendly = 'FlashAttention Forward (split-KV)'
        generic = [["Q:(B,H,T,D)", "K:(B,H,S,D)", "V:(B,H,S,D)", "->", "O:(B,H,T,D)"]]
        m = re.search(r'Flash_fwd_kernel_traits<\(int\)(\d+),\s*\(int\)(\d+),\s*\(int\)(\d+)', raw)
        if m:
            Ttile, D, Stile = [int(m.group(1)), int(m.group(2)), int(m.group(3))]
            fixed = [[f'Q:(B,H,T,{D})', f'K:(B,H,S,{D})', f'V:(B,H,S,{D})', '->', f'O:(B,H,T,{D})']]
            tiled = [[f'Q_tile:({Ttile}, {D})', f'K_tile:({Stile}, {D})', f'V_tile:({Stile}, {D})', '->', f'O_tile:({Ttile}, {D})']]
        desc = 'IO-aware fused attention; traits encode tile sizes.'
        return friendly, source, generic, fixed, tiled, desc

    if 'pytorchmemeffattention' in raw or 'fmha_cutlass' in lower:
        source = 'PyTorch MemEffAttention'
        friendly = 'Memory-Efficient Attention (CUTLASS)'
        generic = [["Q:(B,H,T,D)", "K:(B,H,S,D)", "V:(B,H,S,D)", "->", "O:(B,H,T,D)"]]
        m = re.search(r'aligned_(\d+)x\1', raw)
        if m:
            D = int(m.group(1))
            fixed = [[f'Q:(B,H,T,{D})', f'K:(B,H,S,{D})', f'V:(B,H,S,{D})', '->', f'O:(B,H,T,{D})']]
            tiled = [[f'Q_tile:({D}, {D})', f'K_tile:({D}, {D})', f'V_tile:({D}, {D})', '->', f'O_tile:({D}, {D})']]
        desc = 'Memory-efficient attention; alignment hints reflect per-head dim.'
        return friendly, source, generic, fixed, tiled, desc

    if 'at::native::' in raw:
        source = 'PyTorch ATen'
        if 'silu_kernel' in raw:
            friendly = 'ATen SiLU (vectorized)'
            generic = [['(N,)', '->', '(N,)']]
        elif 'CUDAFunctor_add' in raw:
            friendly = 'ATen elementwise add'
            generic = [['(N,)', '(N,)', '->', '(N,)']]
        elif 'MulFunctor' in raw:
            friendly = 'ATen elementwise multiply'
            generic = [['(N,)', '(N,)', '->', '(N,)']]
        elif 'CatArrayBatchedCopy' in raw:
            friendly = 'ATen cat batched copy (vectorized)'
            generic = [['(B, …)', '(B, …)', '->', '(B, …)']]
        else:
            friendly = 'ATen kernel'
            generic = [['(…)', '->', '(…)']]
        desc = 'TensorIterator/vectorized kernel.'
        return friendly, source, generic, fixed, tiled, desc

    if 'cudnn' in lower and 'fprop' in lower and 'nhwc' in lower:
        source = 'cuDNN'
        friendly = 'cuDNN conv fprop (NHWC)'
        generic = [['X:(N,H,W,C)', 'W:(KH,KW,C,OC)', '->', 'Y:(N,H_out,W_out,OC)']]
        desc = 'Convolution forward pass.'
        return friendly, source, generic, fixed, tiled, desc

    return friendly, source, generic, fixed, tiled, desc

entries = []
for nm in ordered:
    fr, src, gen, fix, tiled, desc = infer(nm)
    entries.append({'raw_name': nm, 'friendly_name': fr, 'source_lib': src,
                    'data_shape': {'generic': gen, 'fixed': fix, 'tiled': tiled},
                    'description': desc})

out = pathlib.Path('docs/deepseek-ocr/kernels.yaml')
lines=['# Auto-generated kernel summary (heuristic). Edit as needed.']
for e in entries:
    def esc(s):
        if isinstance(s, str) and any(ch in s for ch in [':','{','}','[',']','#','&','*','!','|','>','\'','\"','%','@','`']):
            return json.dumps(s)
        return s
    lines.append('- raw_name: ' + esc(e['raw_name']))
    lines.append('  friendly_name: ' + esc(e['friendly_name']))
    lines.append('  source_lib: ' + esc(e['source_lib']))
    lines.append('  data_shape:')
    lines.append('    generic:')
    for tup in e['data_shape']['generic']:
        items = ', '.join(esc(x) for x in tup)
        lines.append(f'      - [{items}]')
    if e['data_shape']['fixed'] is None:
        lines.append('    fixed: null')
    else:
        lines.append('    fixed:')
        for tup in e['data_shape']['fixed']:
            items = ', '.join(esc(x) for x in tup)
            lines.append(f'      - [{items}]')
    if e['data_shape']['tiled'] is None:
        lines.append('    tiled: null')
    else:
        lines.append('    tiled:')
        for tup in e['data_shape']['tiled']:
            items = ', '.join(esc(x) for x in tup)
            lines.append(f'      - [{items}]')
    lines.append('  description: ' + esc(e['description']))

out.write_text('\n'.join(lines)+"\n", encoding='utf-8')
print('Wrote', out)
PY`

That generator applies the same heuristics described below and rewrites `kernels.yaml` using the latest profiling run.

YAML schema and keywords
- The file is a YAML list; each item is a single kernel entry with the following keys:
  - `raw_name` (string)
    - The demangled function name as emitted in Nsight Systems CSV (quoted when needed).
  - `friendly_name` (string)
    - Human‑readable label (e.g., “CUTLASS GEMM (Tensor Core)”, “FlashAttention Forward”).
  - `source_lib` (enum)
    - One of: `cuBLAS`, `CUTLASS`, `FlashAttention`, `PyTorch MemEffAttention`, `PyTorch ATen`, `cuDNN`, `Triton`, `CUDA`.
  - `data_shape` (mapping)
    - `generic` (list of tuple‑like entries)
      - Operator‑level semantics using NumPy‑style shapes, for example GEMM:
        - `[(M, K), (K, N), ->, (M, N)]`
      - Attention uses symbolic Q/K/V/O with batch/head/time dims, e.g.:
        - `["Q:(B,H,T,D)", "K:(B,H,S,D)", "V:(B,H,S,D)", "->", "O:(B,H,T,D)"]`
    - `fixed` (list or null)
      - Concrete shapes when the variant enforces a dimension (e.g., per‑head `D=64` inferred from `aligned_64x64` or FlashAttention traits). Null when not applicable.
    - `tiled` (list or null)
      - A single tile‑level computation in NumPy style. Examples:
        - CUTLASS GEMM name contains `256x128_32x3` → `[(256, 32), (32, 128), ->, (256, 128)]`
        - FlashAttention traits contain `Ttile=128, D=64, Stile=128` → `["Q_tile:(128, 64)", "K_tile:(128, 64)", "V_tile:(128, 64)", "->", "O_tile:(128, 64)"]`
      - Note: tile sizes are kernel configuration, not global tensor dimensions.
  - `description` (string)
    - Short explanation and any clarifying notes (e.g., that tile sizes shown are not global tensor shapes).

Heuristics used for mapping
- cuBLAS (GEMV/GEMM): generic shapes only; tiles are not encoded in names.
- CUTLASS GEMM/WMMA: names embed tile info like `MxN_KxStages`; rendered as a single tile multiply in `data_shape.tiled`.
- FlashAttention (`flash::flash_fwd_splitkv_kernel`): traits encode tile sizes and head dim; `fixed` and `tiled` populated accordingly.
- PyTorch MemEffAttention (`fmha_cutlassF_*aligned_64x64*`): `D=64` fixed; `tiled` approximated with `(D, D)` slices.
- PyTorch ATen and cuDNN kernels: generic shapes; no fixed/tiled unless clearly encoded.

Tips
- After generation, review entries for clarity and adjust `friendly_name`/`description` if you have more precise context.
- If you publish results, include the Nsight run directory pointer (e.g., `tmp/nsys-profile/<ts>/`) for reproducibility.

