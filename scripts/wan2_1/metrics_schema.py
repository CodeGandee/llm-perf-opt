from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class VramPeak:
    device: str  # "cuda:0", "cuda:1", ...
    max_memory_allocated_bytes: int
    max_memory_reserved_bytes: int


@dataclass(frozen=True)
class RunMetrics:
    run_index: int
    seed: int
    diffusion_profile_steps: int
    diffusion_estimate_steps: int
    diffusion_estimate_method: str
    text_encode_s: float
    diffusion_step_ms: List[float]
    diffusion_step_profile_total_ms: float
    diffusion_step_estimate_total_ms: float
    vae_decode_s: Optional[float]
    vram_peaks: List[VramPeak]
    output_latent_shape: List[int]
    output_video_shape: Optional[List[int]]


@dataclass(frozen=True)
class FlopEstimates:
    method: str
    notes: str
    text_encoder_flops: Optional[int]
    diffusion_step_flops: Optional[int]
    diffusion_total_profile_steps_flops: Optional[int]
    diffusion_total_estimate_steps_flops: Optional[int]
    vae_decode_flops: Optional[int]
    end_to_end_total_profile_steps_flops: Optional[int]
    end_to_end_total_estimate_steps_flops: Optional[int]


def to_json_dict(obj: Any) -> Dict[str, Any]:
    return asdict(obj)
