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
import nvtx  # type: ignore[import-untyped]


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
        self.m_nvtx_hooks: list[Any] = []
        self.m_stage_time_ms: dict[str, float] = {"sam": 0.0, "clip": 0.0, "projector": 0.0}

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
        try:
            inst._install_nvtx_stage_hooks()
        except Exception:
            # Best-effort; NVTX hooks are optional
            pass
        return inst

    def _install_nvtx_stage_hooks(self) -> None:
        """Attach NVTX ranges to key submodules (SAM, CLIP, projector).

        This mirrors the paper's stages and allows profiler views to attribute
        time to these blocks even when the logic is inside the model.
        """

        if self.m_model is None:
            return

        core = getattr(self.m_model, "model", None)
        if core is None:
            return

        def _attach(mod: Any, label: str) -> None:
            if mod is None:
                return
            try:
                import time
                def _pre(_m, _inp):
                    nvtx.push_range(label)
                    _m.__dict__["__nvtx_t0"] = time.perf_counter()

                def _post(_m, _inp, _out):
                    nvtx.pop_range()
                    try:
                        t0 = _m.__dict__.pop("__nvtx_t0", None)
                        if t0 is not None:
                            dt = (time.perf_counter() - float(t0)) * 1000.0
                            self.m_stage_time_ms[label] = self.m_stage_time_ms.get(label, 0.0) + float(dt)
                    except Exception:
                        pass

                h1 = mod.register_forward_pre_hook(_pre)
                h2 = mod.register_forward_hook(_post)
                self.m_nvtx_hooks.extend([h1, h2])
            except Exception:
                pass

        _attach(getattr(core, "sam_model", None), "sam")
        _attach(getattr(core, "vision_model", None), "clip")
        _attach(getattr(core, "projector", None), "projector")

    @torch.inference_mode()
    def run_inference(
        self,
        image_path: str,
        prompt: str,
        max_new_tokens: int = 64,
        return_text: bool = False,
        preprocess: Optional[dict] = None,
        infer: Optional[dict] = None,
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

        # Build inputs: either full preprocessing (default) or placeholder path
        use_pre = True
        base_size = 1024
        image_size = 640
        crop_mode = False
        patch_size = 16
        downsample_ratio = 4
        if preprocess is not None:
            use_pre = bool(preprocess.get("enable", True))
            base_size = int(preprocess.get("base_size", base_size))
            image_size = int(preprocess.get("image_size", image_size))
            crop_mode = bool(preprocess.get("crop_mode", crop_mode))
            patch_size = int(preprocess.get("patch_size", patch_size))
            downsample_ratio = int(preprocess.get("downsample_ratio", downsample_ratio))

        if use_pre:
            # Load and normalize image
            from PIL import Image, ImageOps  # local import to avoid hard dep at import time
            from torchvision import transforms  # type: ignore[import-untyped]

            img = Image.open(img_abs).convert("RGB")
            img = ImageOps.exif_transpose(img)
            mean = (0.5, 0.5, 0.5)
            tfm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=mean),
            ])

            # Global padded view -> BF16 tensor
            global_view = ImageOps.pad(img, (base_size, base_size), color=tuple(int(x * 255) for x in mean))
            images_ori = tfm(global_view).to(self.m_dtype or torch.bfloat16)
            images_ori = images_ori.unsqueeze(0).to(self.m_device)  # [1,3,H,W]

            # Local crops (optional via crop_mode) — align with vendor dynamic_preprocess
            images_spatial_crop = torch.tensor([[1, 1]], dtype=torch.long, device=self.m_device)
            if crop_mode:
                # Re-implement vendor dynamic_preprocess selection
                orig_w, orig_h = img.size
                aspect_ratio = orig_w / float(orig_h)
                # Candidate ratios where blocks in [2..9]
                target_ratios = sorted({
                    (i, j)
                    for n in range(2, 10)
                    for i in range(1, n + 1)
                    for j in range(1, n + 1)
                    if (i * j) <= 9 and (i * j) >= 2
                }, key=lambda x: x[0] * x[1])

                def _closest_ratio(ratio_list: list[tuple[int, int]]) -> tuple[int, int]:
                    best = (1, 1)
                    best_diff = float("inf")
                    area = orig_w * orig_h
                    for (w_r, h_r) in ratio_list:
                        target_ar = w_r / float(h_r)
                        diff = abs(aspect_ratio - target_ar)
                        if diff < best_diff:
                            best = (w_r, h_r)
                            best_diff = diff
                        elif diff == best_diff:
                            if area > 0.5 * image_size * image_size * w_r * h_r:
                                best = (w_r, h_r)
                    return best

                w_crop, h_crop = _closest_ratio(target_ratios)
                target_w = image_size * w_crop
                target_h = image_size * h_crop
                resized = img.resize((target_w, target_h))
                crops = []
                for i in range(w_crop * h_crop):
                    c = i % (target_w // image_size)
                    r = i // (target_w // image_size)
                    box = (c * image_size, r * image_size, (c + 1) * image_size, (r + 1) * image_size)
                    crops.append(resized.crop(box))
                images_spatial_crop = torch.tensor([[w_crop, h_crop]], dtype=torch.long, device=self.m_device)
                crop_tensors = [tfm(c).to(self.m_dtype or torch.bfloat16) for c in crops]
                images_crop = (
                    torch.stack(crop_tensors, dim=0).to(self.m_device)
                    if crop_tensors
                    else torch.zeros((1, 3, base_size, base_size), dtype=self.m_dtype or torch.bfloat16, device=self.m_device)
                )
            else:
                images_crop = torch.zeros((1, 3, base_size, base_size), dtype=self.m_dtype or torch.bfloat16, device=self.m_device)
                w_crop = h_crop = 1
            images = [(images_crop, images_ori)]

            # Build input_ids and images_seq_mask aligned to <image> span
            IMAGE_TOKEN_ID = 128815
            text_splits = prompt.split("<image>")
            tokenized_str: list[int] = []
            mask_list: list[bool] = []

            for text_piece in text_splits[:-1]:
                toks = self.m_tokenizer.encode(text_piece, add_special_tokens=False)
                tokenized_str += toks
                mask_list += [False] * len(toks)

                # Mirror vendor token span sizing
                import math as _math
                num_queries = _math.ceil((image_size // patch_size) / downsample_ratio)
                num_queries_base = _math.ceil((base_size // patch_size) / downsample_ratio)
                # Base tokens (global view)
                tokenized_image = ([IMAGE_TOKEN_ID] * num_queries_base + [IMAGE_TOKEN_ID]) * num_queries_base
                tokenized_image += [IMAGE_TOKEN_ID]
                # Local crops (if any)
                if crop_mode and (w_crop > 1 or h_crop > 1):
                    tokenized_image += (
                        ([IMAGE_TOKEN_ID] * (num_queries * w_crop) + [IMAGE_TOKEN_ID]) * (num_queries * h_crop)
                    )
                tokenized_str += tokenized_image
                mask_list += [True] * len(tokenized_image)

            toks_tail = self.m_tokenizer.encode(text_splits[-1], add_special_tokens=False)
            tokenized_str = [0] + tokenized_str + toks_tail  # BOS=0
            mask_list = [False] + mask_list + [False] * len(toks_tail)

            input_ids = torch.tensor(tokenized_str, dtype=torch.long, device=self.m_device).unsqueeze(0)
            images_seq_mask = torch.tensor(mask_list, dtype=torch.bool, device=self.m_device).unsqueeze(0)
            logger.info(
                "Preprocess | base=%d img=%d crop=%s grid=%dx%d input_len=%d",
                int(base_size),
                int(image_size),
                str(bool(crop_mode)),
                int(w_crop),
                int(h_crop),
                int(input_ids.shape[-1]),
            )
        else:
            # Placeholder tensors (no vision fusion)
            zeros_ori = torch.zeros((1, 3, 640, 640), dtype=self.m_dtype or torch.bfloat16, device=self.m_device)
            zeros_crop = torch.zeros((1, 3, 1024, 1024), dtype=self.m_dtype or torch.bfloat16, device=self.m_device)
            images = [(zeros_crop, zeros_ori)]
            images_spatial_crop = torch.zeros((1, 2), dtype=torch.long, device=self.m_device)
            images_seq_mask = None
            input_ids = self.m_tokenizer(prompt, return_tensors="pt").to(self.m_device)["input_ids"]

        # Prefill: first forward pass
        t0 = perf_counter()
        self._reset_stage_time_accum()
        with prefill_range():
            if use_pre:
                inputs = {"input_ids": input_ids}
            else:
                nvtx.push_range("tokenizer")
                try:
                    inputs = self.m_tokenizer(prompt, return_tensors="pt").to(self.m_device)
                finally:
                    nvtx.pop_range()
            _ = self.m_model(
                **inputs,
                images=images,
                images_seq_mask=images_seq_mask,
                images_spatial_crop=images_spatial_crop,
            )
        prefill_ms = (perf_counter() - t0) * 1000.0
        logger.info("Prefill done | image=%s prefill_ms=%.3f", img_abs, prefill_ms)

        # Decode: greedy generation
        t1 = perf_counter()
        with decode_range():
            gen_kwargs: dict[str, object] = {
                "max_new_tokens": int(max_new_tokens),
                "eos_token_id": getattr(self.m_tokenizer, "eos_token_id", None),
            }
            if infer is not None:
                for k in ("temperature", "no_repeat_ngram_size", "do_sample", "top_p", "top_k"):
                    if k in infer:
                        gen_kwargs[k] = infer[k]
            # Provide attention mask to silence warnings and match vendor behavior
            attention_mask = None
            try:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=self.m_device)
            except Exception:
                attention_mask = None
            out = self.m_model.generate(
                **inputs,
                images=images,
                images_seq_mask=images_seq_mask,
                images_spatial_crop=images_spatial_crop,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
        decode_ms = (perf_counter() - t1) * 1000.0

        input_len = int(inputs["input_ids"].shape[-1])
        tokens = int(out.shape[-1] - input_len)
        vision_ms = float(self.get_vision_time_ms())
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
            "prefill_len": int(input_len),
            "vision_ms": float(vision_ms),
        }
        # Include sub-stage timings if available from NVTX hooks
        try:
            result["sam_ms"] = float(self.m_stage_time_ms.get("sam", 0.0))
            result["clip_ms"] = float(self.m_stage_time_ms.get("clip", 0.0))
            result["projector_ms"] = float(self.m_stage_time_ms.get("projector", 0.0))
        except Exception:
            pass
        if return_text and text is not None:
            result["text"] = text
        return result
    def _reset_stage_time_accum(self) -> None:
        self.m_stage_time_ms = {"sam": 0.0, "clip": 0.0, "projector": 0.0}

    def get_vision_time_ms(self) -> float:
        return float(self.m_stage_time_ms.get("sam", 0.0) + self.m_stage_time_ms.get("clip", 0.0) + self.m_stage_time_ms.get("projector", 0.0))

    def estimate_static_compute(self, image_h: int = 1024, image_w: int = 1024, seq_len: int = 1024) -> dict:
        """Best-effort static compute report using fvcore + analytic formulas.

        Returns params and rough FLOPs estimates for vision/prefill/decode.
        """

        report: dict[str, object] = {"notes": "fvcore used where feasible; some blocks analytic"}
        try:
            total = 0
            trainable = 0
            if self.m_model is not None:
                for p in self.m_model.parameters():
                    n = int(p.numel())
                    total += n
                    if p.requires_grad:
                        trainable += n
            report["params"] = {"total": int(total), "trainable": int(trainable)}
        except Exception:
            report["params"] = {"total": 0, "trainable": 0}

        # Attempt fvcore FLOPs for lm_head projection as a proxy
        try:
            from fvcore.nn import FlopCountAnalysis  # type: ignore[import-untyped]
            import torch
            if self.m_model is not None:
                head = getattr(self.m_model, "lm_head", None)
                hidden_size = int(getattr(getattr(self.m_model, "config", object()), "hidden_size", 1024))
                if head is not None:
                    x = torch.randn(1, 1, hidden_size, device=self.m_device or torch.device("cpu"), dtype=torch.float32)
                    fca = FlopCountAnalysis(head, (x,))
                    flops = float(fca.total())
                else:
                    flops = 0.0
                report["decode"] = {"flops_per_token_proj_only": flops}
        except Exception:
            report.setdefault("decode", {})

        # Add placeholders for vision/prefill; full fvcore across custom modules may not work
        report.setdefault("vision", {"flops": 0.0})
        report.setdefault("prefill", {"flops_per_seq": 0.0, "flops_per_token_avg": 0.0})
        return report
