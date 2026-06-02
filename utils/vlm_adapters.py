from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


_YES_RE = re.compile(r"\b(yes|yeah|yep|true)\b", re.IGNORECASE)
_NO_RE = re.compile(r"\b(no|nope|false)\b", re.IGNORECASE)
_SOLUTION_RE = re.compile(r"<solution>\s*(yes|no)\s*</solution>", re.IGNORECASE)


def _extract_yes_no(text: str) -> str:
    if not text:
        return "no"

    m = _SOLUTION_RE.search(text)
    if m:
        return m.group(1).lower()

    t = text.strip().lower()
    # strong prefixes
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"

    yes = _YES_RE.search(t) is not None
    no = _NO_RE.search(t) is not None
    if yes and not no:
        return "yes"
    if no and not yes:
        return "no"

    # tie-break: last occurrence
    last_yes = t.rfind("yes")
    last_no = t.rfind("no")
    return "yes" if last_yes > last_no else "no"


@dataclass
class VLMGeneration:
    raw_text: str
    predicted_answer: str


class Qwen3VLAdapter:
    """
    Minimal inference adapter for Qwen3-VL models using 🤗 Transformers.

    It produces a binary yes/no answer required by ORIC-Bench.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        dtype: Optional[str] = "bfloat16",
        attn_implementation: Optional[str] = None,
        trust_remote_code: bool = True,
    ):
        from transformers import Qwen3VLForConditionalGeneration  # local import for optional availability

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        torch_dtype: Optional[torch.dtype]
        if dtype in (None, "auto"):
            torch_dtype = None
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "float32":
            torch_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        model_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )

    def generate_yes_no(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ) -> VLMGeneration:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]

        prompt: str
        if hasattr(self.processor, "apply_chat_template"):
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # fallback: plain question
            prompt = question

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature if temperature > 0 else None,
        }
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        # decode only newly generated portion when possible
        if "input_ids" in inputs:
            gen_ids = out[0][inputs["input_ids"].shape[-1] :]
        else:
            gen_ids = out[0]

        text = self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        ans = _extract_yes_no(text)
        return VLMGeneration(raw_text=text, predicted_answer=ans)


class HFGenericVLMAdapter:
    """Generic HF VLM adapter (InternVL, LLaVA, Phi-Vision, GLM-4V, VILA, etc.)."""

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        dtype: Optional[str] = "bfloat16",
        attn_implementation: Optional[str] = None,
        trust_remote_code: bool = True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        torch_dtype: Optional[torch.dtype]
        if dtype in (None, "auto"):
            torch_dtype = None
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "float32":
            torch_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        model_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )

    def generate_yes_no(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ) -> VLMGeneration:
        if hasattr(self.processor, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = question

        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature if temperature > 0 else None,
        }
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        if "input_ids" in inputs:
            gen_ids = out[0][inputs["input_ids"].shape[-1] :]
        else:
            gen_ids = out[0]

        tok = getattr(self.processor, "tokenizer", None) or getattr(self.processor, "_tokenizer", None)
        if tok is None:
            raise RuntimeError("Processor does not expose a tokenizer; cannot decode generations.")

        text = tok.decode(gen_ids, skip_special_tokens=True).strip()
        ans = _extract_yes_no(text)
        return VLMGeneration(raw_text=text, predicted_answer=ans)


def pick_adapter(model_family: str, model_name_or_path: str):
    mf = model_family.lower()
    if mf == "auto":
        name = model_name_or_path.lower()
        if "qwen3-vl" in name or "qwen3_vl" in name:
            mf = "qwen3_vl"
        else:
            mf = "hf_generic"

    if mf == "qwen3_vl":
        return Qwen3VLAdapter(model_name_or_path=model_name_or_path)
    if mf == "hf_generic":
        return HFGenericVLMAdapter(model_name_or_path=model_name_or_path)

    raise ValueError(
        f"Unsupported model_family: {model_family}. "
        "Use auto/qwen3_vl/hf_generic, or set run=false for API/detector entries."
    )

