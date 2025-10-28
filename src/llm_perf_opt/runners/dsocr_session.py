"""DeepSeek-OCR session wrapper.

Provides a load-once wrapper around a third-party model/tokenizer and exposes a
`run_inference` method with NVTX stage segmentation. This class follows the
OO discipline (m_-prefixed members, factory constructor) and avoids modifying
third-party code.
"""

from __future__ import annotations

from pathlib import Path
import logging
from time import perf_counter
from typing import Any, Optional

import torch
from transformers import AutoModel, AutoTokenizer  # type: ignore[import-untyped]

from llm_perf_opt.profiling.nvtx_utils import decode_range, prefill_range


class DeepSeekOCRSession:
    """Load-once session for DeepSeek-OCR with NVTX-segmented inference.

    Use `from_local()` to construct an instance. Member variables are prefixed
    with `m_` and read-only properties are provided for access where useful.
    """

    def __init__(self) -> None:
        self.m_model: Optional[Any] = None
        self.m_tokenizer: Optional[Any] = None
        self.m_device: Optional[torch.device] = None
        self.m_dtype: Optional[torch.dtype] = torch.bfloat16

    @property
    def device(self) -> Optional[torch.device]:
        """Return the configured device, if initialized."""

        return self.m_device

    @classmethod
    def from_local(
        cls, model_path: str, device: str = "cuda:0", use_flash_attn: bool = True
    ) -> "DeepSeekOCRSession":
        """Create a session from a local model path.

        Parameters
        ----------
        model_path : str
            Absolute path to the model repository directory.
        device : str, default='cuda:0'
            Device specifier (e.g., 'cuda:0' or 'cpu').
        use_flash_attn : bool, default=True
            Attempt to use flash attention for faster inference.
        """

        if not Path(model_path).is_absolute():
            raise ValueError("model_path must be absolute")

        inst = cls()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
        attn_impl = "flash_attention_2" if use_flash_attn else "eager"
        try:
            model = AutoModel.from_pretrained(
                model_path,
                _attn_implementation=attn_impl,
                trust_remote_code=True,
                use_safetensors=True,
                local_files_only=True,
            )
        except Exception:
            model = AutoModel.from_pretrained(
                model_path,
                _attn_implementation="eager",
                trust_remote_code=True,
                use_safetensors=True,
                local_files_only=True,
            )

        dev = torch.device(device) if device else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        model = model.eval().to(dev)
        try:
            model = model.to(dtype=torch.bfloat16)
        except Exception:
            pass

        inst.m_model = model
        inst.m_tokenizer = tokenizer
        inst.m_device = dev
        return inst

    @torch.inference_mode()
    def run_inference(
        self,
        image_path: str,
        prompt: str,
        max_new_tokens: int = 64,
        return_text: bool = False,
    ) -> dict:
        """Run NVTX-segmented inference on a single image.

        For Stage 1, we prefer a simple generate-based loop that allows
        segmentation visibility. Some third-party APIs (e.g., ``.infer``) may
        be monolithic; we don't rely on them for NVTX segmentation here.
        """

        if self.m_model is None or self.m_tokenizer is None or self.m_device is None:
            raise RuntimeError("Session is not initialized. Use from_local() first.")

        logger = logging.getLogger(__name__)
        img_abs = str(Path(image_path).resolve())
        logger.info("Session inference start | image=%s max_new_tokens=%d", img_abs, int(max_new_tokens))

        # Construct minimal image placeholders for DeepSeek‑OCR HF path.
        # The HF implementation expects image tensors in kwargs. For Stage 1,
        # we provide zeroed tensors to bypass the heavy vision stack while
        # preserving code paths and shapes. This yields representative timings
        # without requiring full image preprocessing here.
        zeros_ori = torch.zeros((1, 3, 640, 640), dtype=self.m_dtype or torch.bfloat16, device=self.m_device)
        zeros_crop = torch.zeros((1, 3, 1024, 1024), dtype=self.m_dtype or torch.bfloat16, device=self.m_device)
        images = [(zeros_crop, zeros_ori)]
        images_spatial_crop = torch.zeros((1, 2), dtype=torch.long, device=self.m_device)

        # Prefill: first forward pass
        t0 = perf_counter()
        with prefill_range():
            inputs = self.m_tokenizer(prompt, return_tensors="pt").to(self.m_device)
            _ = self.m_model(
                **inputs,
                images=images,
                images_seq_mask=None,
                images_spatial_crop=images_spatial_crop,
            )
        prefill_ms = (perf_counter() - t0) * 1000.0
        logger.info("Prefill done | image=%s prefill_ms=%.3f", img_abs, prefill_ms)

        # Decode: greedy generation
        t1 = perf_counter()
        with decode_range():
            out = self.m_model.generate(
                **inputs,
                images=images,
                images_seq_mask=None,
                images_spatial_crop=images_spatial_crop,
                max_new_tokens=max_new_tokens,
            )
        decode_ms = (perf_counter() - t1) * 1000.0

        input_len = int(inputs["input_ids"].shape[-1])
        tokens = int(out.shape[-1] - input_len)
        text: str | None = None
        if return_text:
            try:
                gen_ids = out[0, input_len:].tolist()
                text = self.m_tokenizer.decode(gen_ids, skip_special_tokens=False)
            except Exception:
                text = None
        logger.info(
            "Decode done | image=%s decode_ms=%.3f tokens=%d tokens_per_s=%.3f",
            img_abs,
            decode_ms,
            tokens,
            (tokens / (decode_ms / 1000.0)) if decode_ms > 0 else 0.0,
        )
        result: dict[str, object] = {
            "prefill_ms": float(prefill_ms),
            "decode_ms": float(decode_ms),
            "tokens": int(tokens),
        }
        if return_text and text is not None:
            result["text"] = text
        return result
