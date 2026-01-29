from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from gpu_idle_check import format_snapshots, resolve_physical_gpu_ids, wait_for_idle
from metrics_schema import FlopEstimates, RunMetrics, VramPeak


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_env_file(path: Path) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ (no overrides)."""

    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip("'").strip('"')
        if not k:
            continue
        os.environ.setdefault(k, v)


def _parse_size(value: str) -> tuple[int, int]:
    # Wan uses (width, height).
    if "*" not in value:
        raise ValueError(f"Invalid --size {value!r}; expected like '1280*720'")
    w_s, h_s = value.split("*", 1)
    return int(w_s), int(h_s)


def _word_count(text: str) -> int:
    # Simple, deterministic "word" heuristic good enough for the >=200 requirement.
    return len([w for w in text.replace("\n", " ").split(" ") if w.strip()])


def _default_long_prompt() -> str:
    # >= 200 words. Keep it single-line so it is easy to copy/record.
    return (
        "Create a cinematic, high detail, text-to-video scene that tells a clear story from beginning to end. "
        "The setting is a busy modern coastal city at sunrise after a night rain, with reflective streets, soft fog, "
        "and warm light spilling between tall buildings. The main subject is a lone cyclist wearing a bright rain "
        "jacket and a small backpack, riding through an intersection while traffic lights change and pedestrians wait. "
        "The camera starts with a wide establishing shot, then smoothly tracks closer, matching the cyclist's speed, "
        "showing the bike chain, wet asphalt, and water droplets on the frame. Add believable motion blur and depth of "
        "field, but keep the subject sharp. Include subtle environmental animation: steam rising from manholes, leaves "
        "fluttering, distant buses pulling away, and a flock of birds crossing the sky. Maintain consistent character "
        "appearance and clothing across the entire clip. The mood is calm but energetic, like a commercial for urban "
        "mobility. Use realistic lighting, physically plausible reflections, and stable geometry. Avoid flicker, avoid "
        "sudden cuts, avoid artifacts, avoid unreadable text, and avoid strange deformations. The cyclist reaches the "
        "waterfront, slows down, and looks toward the horizon as the sun breaks through clouds. End on a gentle "
        "close-up of the cyclist's face and the glowing skyline, with smooth camera easing and natural color grading."
    )


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    repo_root = _repo_root()

    parser = argparse.ArgumentParser(description="Wan2.1 runtime profiling (T2V-14B)")
    parser.add_argument("--ckpt-dir", type=str, default=str(repo_root / "models" / "wan2.1-t2v-14b" / "source-data"))
    parser.add_argument("--task", type=str, default="t2v-14B")
    parser.add_argument("--size", type=str, default="1280*720")
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=10, help="Diffusion steps to profile (default: 10).")
    parser.add_argument("--estimate-steps", type=int, default=50, help="Extrapolation target (default: 50).")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--collect-flops",
        action="store_true",
        help="Collect FLOP estimates for text encoder, 1 diffusion step (cond+uncond model forward), and VAE decode, and extrapolate diffusion to profile/estimate steps. Expensive.",
    )
    parser.add_argument("--prompt-file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device-id", type=int, default=0, help="Logical CUDA device id inside the process (default: 0).")
    parser.add_argument("--sample-solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    parser.add_argument("--sample-shift", type=float, default=5.0)
    parser.add_argument("--guide-scale", type=float, default=5.0)
    parser.add_argument("--offload-model", action="store_true", help="Mirror upstream default CPU offload behavior.")
    parser.add_argument("--cuda-visible-devices", type=str, default=None, help="Override CUDA_VISIBLE_DEVICES (e.g. '6,7').")
    parser.add_argument("--wait-idle-s", type=float, default=0.0, help="Wait up to N seconds for target GPUs to become idle.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output dir. Default: tmp/wan21-profiling-<time>-fp16/",
    )

    args = parser.parse_args(argv)

    # Load .env first (no overrides), then allow explicit CLI override.
    _load_env_file(repo_root / ".env")
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    # Pre-run GPU idleness check (before importing torch / initializing CUDA context).
    physical_gpu_ids = resolve_physical_gpu_ids(default=[])
    if physical_gpu_ids:
        try:
            snaps = wait_for_idle(
                physical_gpu_ids,
                max_util_gpu_pct=1.0,
                max_mem_used_pct=1.0,
                timeout_s=float(args.wait_idle_s),
                poll_s=5.0,
            )
            print("GPU idle check: PASS:", format_snapshots(snaps))
        except Exception as exc:
            print(f"GPU idle check: FAIL: {exc}", file=sys.stderr)
            return 2
    else:
        print("GPU idle check: SKIP (no physical GPU ids resolved)", file=sys.stderr)

    # Lazy imports after env is set.
    import math
    import gc

    import torch
    import torch.cuda.amp as amp

    # Make upstream Wan2.1 code importable.
    wan_repo = repo_root / "extern" / "github" / "wan2.1"
    sys.path.insert(0, str(wan_repo.resolve()))
    import wan  # type: ignore
    from wan.configs import SIZE_CONFIGS, WAN_CONFIGS  # type: ignore
    from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps  # type: ignore
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # type: ignore

    ckpt_dir = Path(args.ckpt_dir).resolve()
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint dir: {ckpt_dir}")

    prompt = _default_long_prompt()
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text()
    prompt_wc = _word_count(prompt)
    if prompt_wc < 200:
        raise ValueError(f"Prompt must be >= 200 words; got {prompt_wc}")

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else (repo_root / "tmp" / f"wan21-profiling-{datetime.now().strftime('%Y%m%d-%H%M%S')}-fp16")
    )
    out_dir = out_dir.resolve()
    (out_dir / "scripts").mkdir(parents=True, exist_ok=True)
    (out_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (out_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)

    _write_text(out_dir / "inputs" / "prompt.txt", prompt)
    _write_json(
        out_dir / "reports" / "run_args.json",
        {
            "argv": sys.argv,
            "env": {k: os.environ.get(k) for k in ["CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "NVIDIA_VISIBLE_DEVICES"]},
            "params": vars(args),
        },
    )

    size = _parse_size(args.size)
    if args.size in SIZE_CONFIGS:
        # Keep the upstream canonical size mapping (width, height) for known presets.
        size = tuple(int(x) for x in SIZE_CONFIGS[args.size])

    cfg = copy.deepcopy(WAN_CONFIGS[args.task])
    # FP16-only profiling (diffusion-focused). Upstream may still force some ops to fp32 for stability.
    cfg.param_dtype = torch.float16
    cfg.t5_dtype = torch.float16

    # Build pipeline once and reuse for warmup + measured runs.
    print("Loading WanT2V pipeline (this can take a while)...")
    wan_t2v = wan.WanT2V(
        config=cfg,
        checkpoint_dir=str(ckpt_dir),
        device_id=int(args.device_id),
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    )

    device = wan_t2v.device

    def _sync() -> None:
        torch.cuda.synchronize(device)

    def _reset_peaks() -> None:
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)

    def _get_peaks() -> list[VramPeak]:
        peaks: list[VramPeak] = []
        for i in range(torch.cuda.device_count()):
            peaks.append(
                VramPeak(
                    device=f"cuda:{i}",
                    max_memory_allocated_bytes=int(torch.cuda.max_memory_allocated(i)),
                    max_memory_reserved_bytes=int(torch.cuda.max_memory_reserved(i)),
                )
            )
        return peaks

    def _build_target_shape_and_seq_len(*, frame_num: int, size_wh: tuple[int, int]) -> tuple[tuple[int, int, int, int], int]:
        # Mirror upstream shape math.
        f = int(frame_num)
        target_shape = (
            int(wan_t2v.vae.model.z_dim),
            int((f - 1) // wan_t2v.vae_stride[0] + 1),
            int(size_wh[1] // wan_t2v.vae_stride[1]),
            int(size_wh[0] // wan_t2v.vae_stride[2]),
        )
        seq_len = math.ceil(
            (target_shape[2] * target_shape[3])
            / (wan_t2v.patch_size[1] * wan_t2v.patch_size[2])
            * target_shape[1]
            / wan_t2v.sp_size
        ) * wan_t2v.sp_size
        return target_shape, int(seq_len)

    target_shape, seq_len = _build_target_shape_and_seq_len(frame_num=args.frames, size_wh=size)
    arg_c: dict[str, Any]
    arg_null: dict[str, Any]

    # Encode prompt once (excluded from diffusion timings).
    print("Encoding prompt (excluded from diffusion step timings)...")
    wan_t2v.text_encoder.model.to(device)
    _sync()
    t0 = time.perf_counter()
    context = wan_t2v.text_encoder([prompt], device)
    context_null = wan_t2v.text_encoder([cfg.sample_neg_prompt], device)
    _sync()
    text_encode_s = float(time.perf_counter() - t0)
    if args.offload_model:
        wan_t2v.text_encoder.model.cpu()
    arg_c = {"context": context, "seq_len": seq_len}
    arg_null = {"context": context_null, "seq_len": seq_len}

    def diffusion_only_profile(*, seed: int, sampling_steps: int, record_step_events: bool) -> tuple[torch.Tensor, list[float]]:
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(int(seed))

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=device,
                generator=seed_g,
            )
        ]

        step_ms: list[float] = []

        with amp.autocast(dtype=torch.float16), torch.no_grad():
            if args.sample_solver == "unipc":
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=wan_t2v.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                )
                sample_scheduler.set_timesteps(int(sampling_steps), device=device, shift=float(args.sample_shift))
                timesteps = sample_scheduler.timesteps
            else:
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=wan_t2v.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                )
                sampling_sigmas = get_sampling_sigmas(int(sampling_steps), float(args.sample_shift))
                timesteps, _ = retrieve_timesteps(sample_scheduler, device=device, sigmas=sampling_sigmas)

            latents = noise
            step_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

            _sync()
            wan_t2v.model.to(device)

            for t in timesteps:
                step_start = torch.cuda.Event(enable_timing=True)
                step_end = torch.cuda.Event(enable_timing=True)
                if record_step_events:
                    step_start.record()

                timestep = torch.stack([t])
                noise_pred_cond = wan_t2v.model(latents, t=timestep, **arg_c)[0]
                noise_pred_uncond = wan_t2v.model(latents, t=timestep, **arg_null)[0]
                noise_pred = noise_pred_uncond + float(args.guide_scale) * (noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g,
                )[0]
                latents = [temp_x0.squeeze(0)]

                if record_step_events:
                    step_end.record()
                    step_events.append((step_start, step_end))

            _sync()
            if record_step_events:
                for s_ev, e_ev in step_events:
                    step_ms.append(float(s_ev.elapsed_time(e_ev)))

            out_latent = latents[0]

        # Cleanup (outside timings).
        del noise, latents
        del sample_scheduler
        if args.offload_model:
            wan_t2v.model.cpu()
            torch.cuda.empty_cache()
            gc.collect()
            _sync()

        return out_latent, step_ms

    # Warmup (excluded from timing metrics by requirement).
    print(f"Warmup: diffusion-only run with steps={args.warmup_steps} (not counted)...")
    _reset_peaks()
    _ = diffusion_only_profile(seed=args.seed, sampling_steps=int(args.warmup_steps), record_step_events=False)
    print("Warmup: done.")

    estimate_method = "sum(step_ms) * (estimate_steps / profile_steps)"
    runs: list[RunMetrics] = []
    print(f"Profiling: diffusion-only {args.runs} runs with steps={args.steps} (estimate {args.estimate_steps})...")
    for i in range(int(args.runs)):
        cur_seed = int(args.seed) + i + 1
        _reset_peaks()
        latent, step_ms = diffusion_only_profile(seed=cur_seed, sampling_steps=int(args.steps), record_step_events=True)
        peaks = _get_peaks()

        vae_decode_s: float | None = None
        output_video_shape: list[int] | None = None
        if i == 0:
            try:
                print("Timing VAE decode once (excluded from diffusion step timings)...")
                _sync()
                t0 = time.perf_counter()
                videos = wan_t2v.vae.decode([latent])
                _sync()
                vae_decode_s = float(time.perf_counter() - t0)
                output_video_shape = list(videos[0].shape) if videos else None
                del videos
            except Exception as exc:
                print(f"VAE decode timing failed: {exc}", file=sys.stderr)

        profile_total_ms = float(sum(step_ms))
        estimate_total_ms = (
            profile_total_ms * float(args.estimate_steps) / float(args.steps) if int(args.steps) > 0 else float("nan")
        )
        mean_step_ms = (profile_total_ms / float(len(step_ms))) if step_ms else float("nan")

        runs.append(
            RunMetrics(
                run_index=i,
                seed=int(cur_seed),
                diffusion_profile_steps=int(args.steps),
                diffusion_estimate_steps=int(args.estimate_steps),
                diffusion_estimate_method=estimate_method,
                text_encode_s=float(text_encode_s),
                diffusion_step_ms=step_ms,
                diffusion_step_profile_total_ms=profile_total_ms,
                diffusion_step_estimate_total_ms=estimate_total_ms,
                vae_decode_s=vae_decode_s,
                vram_peaks=peaks,
                output_latent_shape=list(latent.shape),
                output_video_shape=output_video_shape,
            )
        )
        print(
            f"run={i} profile_s={profile_total_ms/1000.0:.3f} est{args.estimate_steps}_s={estimate_total_ms/1000.0:.3f} "
            f"mean_step_ms={mean_step_ms:.3f} peak0_gib={peaks[0].max_memory_allocated_bytes/(1024**3):.2f}"
        )

    flops = FlopEstimates(
        method="torch.utils.flop_counter.FlopCounterMode",
        notes="Counts model-forward FLOPs only via torch.utils.flop_counter.FlopCounterMode. Scheduler/update ops are not counted. Custom CUDA ops may be undercounted.",
        text_encoder_flops=None,
        diffusion_step_flops=None,
        diffusion_total_profile_steps_flops=None,
        diffusion_total_estimate_steps_flops=None,
        vae_decode_flops=None,
        end_to_end_total_profile_steps_flops=None,
        end_to_end_total_estimate_steps_flops=None,
    )
    if args.collect_flops:
        from torch.utils.flop_counter import FlopCounterMode

        notes_extra: list[str] = []

        # Use one representative latent for all FLOP passes (shape matters; values do not).
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(int(args.seed))
        latent = torch.randn(target_shape, device=device, dtype=torch.float32, generator=seed_g)
        latents = [latent]

        text_encoder_flops: int | None = None
        diffusion_step_flops: int | None = None
        vae_decode_flops: int | None = None

        try:
            wan_t2v.text_encoder.model.to(device)
            with torch.no_grad(), FlopCounterMode(display=False) as fcm:
                _ = wan_t2v.text_encoder([prompt], device)
                _ = wan_t2v.text_encoder([cfg.sample_neg_prompt], device)
            text_encoder_flops = int(fcm.get_total_flops())
            if args.offload_model:
                wan_t2v.text_encoder.model.cpu()
                torch.cuda.empty_cache()
                gc.collect()
                _sync()
        except Exception as exc:
            notes_extra.append(f"text_encoder_flops ERROR: {exc}")

        try:
            with amp.autocast(dtype=torch.float16), torch.no_grad():
                if args.sample_solver == "unipc":
                    sample_scheduler = FlowUniPCMultistepScheduler(
                        num_train_timesteps=wan_t2v.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False,
                    )
                    sample_scheduler.set_timesteps(int(args.steps), device=device, shift=float(args.sample_shift))
                    timesteps = sample_scheduler.timesteps
                else:
                    sample_scheduler = FlowDPMSolverMultistepScheduler(
                        num_train_timesteps=wan_t2v.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False,
                    )
                    sampling_sigmas = get_sampling_sigmas(int(args.steps), float(args.sample_shift))
                    timesteps, _ = retrieve_timesteps(sample_scheduler, device=device, sigmas=sampling_sigmas)

                if len(timesteps) <= 0:
                    raise RuntimeError("No timesteps produced for FLOP pass.")

                timestep = torch.stack([timesteps[0]])

                wan_t2v.model.to(device)
                with FlopCounterMode(display=False) as fcm:
                    _ = wan_t2v.model(latents, t=timestep, **arg_c)[0]
                    _ = wan_t2v.model(latents, t=timestep, **arg_null)[0]
                diffusion_step_flops = int(fcm.get_total_flops())
        except Exception as exc:
            notes_extra.append(f"diffusion_step_flops ERROR: {exc}")

        try:
            with torch.no_grad(), FlopCounterMode(display=False) as fcm:
                _ = wan_t2v.vae.decode(latents)
            vae_decode_flops = int(fcm.get_total_flops())
        except Exception as exc:
            notes_extra.append(f"vae_decode_flops ERROR: {exc}")

        diffusion_total_profile_steps_flops = (
            int(diffusion_step_flops * int(args.steps)) if diffusion_step_flops is not None else None
        )
        diffusion_total_estimate_steps_flops = (
            int(diffusion_step_flops * int(args.estimate_steps)) if diffusion_step_flops is not None else None
        )
        end_to_end_total_profile_steps_flops = (
            int(text_encoder_flops + diffusion_total_profile_steps_flops + vae_decode_flops)
            if (
                text_encoder_flops is not None
                and diffusion_total_profile_steps_flops is not None
                and vae_decode_flops is not None
            )
            else None
        )
        end_to_end_total_estimate_steps_flops = (
            int(text_encoder_flops + diffusion_total_estimate_steps_flops + vae_decode_flops)
            if (
                text_encoder_flops is not None
                and diffusion_total_estimate_steps_flops is not None
                and vae_decode_flops is not None
            )
            else None
        )

        flops = FlopEstimates(
            method=flops.method,
            notes=flops.notes if not notes_extra else f"{flops.notes} " + " | ".join(notes_extra),
            text_encoder_flops=text_encoder_flops,
            diffusion_step_flops=diffusion_step_flops,
            diffusion_total_profile_steps_flops=diffusion_total_profile_steps_flops,
            diffusion_total_estimate_steps_flops=diffusion_total_estimate_steps_flops,
            vae_decode_flops=vae_decode_flops,
            end_to_end_total_profile_steps_flops=end_to_end_total_profile_steps_flops,
            end_to_end_total_estimate_steps_flops=end_to_end_total_estimate_steps_flops,
        )

    # Persist artifacts.
    payload: dict[str, Any] = {
        "meta": {
            "created_at": datetime.now().isoformat(),
            "host": os.uname().nodename,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "precision": {
            "requested": "fp16",
            "autocast_dtype": "float16",
        },
        "workload": {
            "task": args.task,
            "size": {"w": int(size[0]), "h": int(size[1])},
            "frames": int(args.frames),
            "diffusion_profile_steps": int(args.steps),
            "diffusion_estimate_steps": int(args.estimate_steps),
            "diffusion_only": True,
            "diffusion_estimate_method": estimate_method,
            "prompt_word_count": int(prompt_wc),
        },
        "warmup": {"steps": int(args.warmup_steps)},
        "runs": [asdict(r) for r in runs],
        "flops": asdict(flops),
    }

    _write_json(out_dir / "reports" / "metrics.json", payload)

    # Small human-readable summary.
    totals_profile_ms = [r.diffusion_step_profile_total_ms for r in runs]
    totals_est_ms = [r.diffusion_step_estimate_total_ms for r in runs]
    mean_profile_s = (sum(totals_profile_ms) / 1000.0 / len(totals_profile_ms)) if totals_profile_ms else float("nan")
    mean_est_s = (sum(totals_est_ms) / 1000.0 / len(totals_est_ms)) if totals_est_ms else float("nan")
    text_s = float(runs[0].text_encode_s) if runs else float("nan")
    vae_s = float(runs[0].vae_decode_s) if (runs and runs[0].vae_decode_s is not None) else float("nan")
    end_to_end_est_s = float(text_s + mean_est_s + vae_s) if all(x == x for x in [text_s, mean_est_s, vae_s]) else float("nan")
    summary = [
        "# Wan2.1 Runtime Profiling Summary",
        "",
        "- Precision: fp16",
        f"- Workload: task={args.task} size={size[0]}x{size[1]} frames={args.frames} diffusion_steps_profiled={args.steps} estimate_steps={args.estimate_steps}",
        f"- Prompt words: {prompt_wc}",
        f"- Runs (excluding warmup): {len(runs)}",
        f"- Text encode time (s): {text_s:.3f}",
        f"- Mean diffusion time profiled (s): {mean_profile_s:.3f}",
        f"- Mean diffusion time estimated (s): {mean_est_s:.3f}",
        f"- VAE decode time (s): {vae_s:.3f}",
        f"- End-to-end time estimated (s): {end_to_end_est_s:.3f}",
        f"- Estimate method: {estimate_method}",
        "",
        "## Per-run diffusion timings",
    ]
    for r in runs:
        profile_s = float(r.diffusion_step_profile_total_ms) / 1000.0
        est_s = float(r.diffusion_step_estimate_total_ms) / 1000.0
        mean_step_ms = (float(r.diffusion_step_profile_total_ms) / len(r.diffusion_step_ms)) if r.diffusion_step_ms else float("nan")
        summary.append(
            f"- run {r.run_index}: profile_s={profile_s:.3f} est{r.diffusion_estimate_steps}_s={est_s:.3f} mean_step_ms={mean_step_ms:.3f}"
        )
    if any(
        v is not None
        for v in [
            flops.text_encoder_flops,
            flops.diffusion_step_flops,
            flops.vae_decode_flops,
            flops.end_to_end_total_estimate_steps_flops,
        ]
    ):
        def _tflops(flops_count: int | None, seconds: float) -> str:
            if flops_count is None or seconds <= 0 or seconds != seconds:
                return "nan"
            return f"{(float(flops_count) / float(seconds) / 1e12):.3f}"

        mean_step_s = (float(sum(runs[0].diffusion_step_ms)) / 1000.0 / len(runs[0].diffusion_step_ms)) if (runs and runs[0].diffusion_step_ms) else float("nan")

        summary += [
            "",
            "## TFLOP/s (estimate)",
            f"- method: {flops.method}",
            f"- notes: {flops.notes}",
            f"- text_encoder_tflops: {_tflops(flops.text_encoder_flops, text_s)}",
            f"- diffusion_step_tflops: {_tflops(flops.diffusion_step_flops, mean_step_s)}",
            f"- vae_decode_tflops: {_tflops(flops.vae_decode_flops, vae_s)}",
            f"- end_to_end_estimate_tflops: {_tflops(flops.end_to_end_total_estimate_steps_flops, end_to_end_est_s)}",
        ]
    _write_text(out_dir / "reports" / "summary.md", "\n".join(summary))

    print(f"Done. Wrote: {out_dir / 'reports' / 'metrics.json'}")
    print(f"Done. Wrote: {out_dir / 'reports' / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
