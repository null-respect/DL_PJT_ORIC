from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image

from utils.repo_setup import add_repo_to_path, ensure_git_repo, register_eve_builder_hooks
from utils.vlm_adapters import VLMGeneration, _build_model_kwargs, _extract_yes_no


def _model_device(model: Any) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    if hasattr(model, "hf_device_map"):
        device_name = next(iter(model.hf_device_map.values()), "cpu")
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _model_dtype(model: Any) -> torch.dtype:
    return next(model.parameters()).dtype


def _resolve_repo_model_name(model_name_or_path: str, *, default: str) -> str:
    name = model_name_or_path.strip("/").split("/")[-1]
    return name or default


class VilaAdapter:
    """VILA inference via NVLabs/VILA repo (llava_llama architecture)."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: Optional[str] = None,
        dtype: Optional[str] = "bfloat16",
        load_in_4bit: bool = False,
    ):
        add_repo_to_path("VILA")
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init

        self._SeparatorStyle = SeparatorStyle
        self._conv_templates = conv_templates
        self._process_images = process_images
        self._tokenizer_image_token = tokenizer_image_token

        disable_torch_init()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_name = get_model_name_from_path(model_name_or_path) or _resolve_repo_model_name(
            model_name_or_path, default="vila"
        )
        load_kwargs: dict[str, Any] = {
            "load_4bit": load_in_4bit,
            "device_map": "auto" if self.device == "cuda" else {"": self.device},
            "device": self.device,
        }
        if not load_in_4bit:
            load_kwargs["torch_dtype"] = torch.bfloat16 if dtype in (None, "bfloat16", "auto") else torch.float16

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_name_or_path,
            model_name,
            None,
            **load_kwargs,
        )
        self.model.eval()
        self.conv_mode = "vicuna_v1"

    def generate_yes_no(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ) -> VLMGeneration:
        conv = self._conv_templates[self.conv_mode].copy()
        qs = f"<image>\n{question}"
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_tensor = self._process_images([image], self.image_processor, self.model.config)
        if isinstance(image_tensor, torch.Tensor) and image_tensor.ndim >= 4:
            image_tensor = image_tensor[0]
        device = _model_device(self.model)
        dtype = _model_dtype(self.model)
        input_ids = self._tokenizer_image_token(
            prompt,
            self.tokenizer,
            return_tensors="pt",
        ).unsqueeze(0)
        input_ids = input_ids.to(device)
        image_tensor = image_tensor.to(device=device, dtype=dtype)

        stop_str = conv.sep if conv.sep_style != self._SeparatorStyle.TWO else conv.sep2
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "use_cache": True,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0),
                **gen_kwargs,
            )

        input_len = input_ids.shape[1]
        text = self.tokenizer.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[0]
        text = text.strip()
        if text.endswith(stop_str):
            text = text[: -len(stop_str)].strip()
        return VLMGeneration(raw_text=text, predicted_answer=_extract_yes_no(text))


class EveAdapter:
    """EVE inference via baaivision/EVE repo (EVEv1)."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: Optional[str] = None,
        dtype: Optional[str] = "bfloat16",
        load_in_4bit: bool = False,
    ):
        ensure_git_repo("EVEv1", "https://github.com/baaivision/EVE.git")
        add_repo_to_path("EVEv1", subdir="EVEv1")
        register_eve_builder_hooks()
        from eve.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from eve.conversation import SeparatorStyle, conv_templates
        from eve.mm_utils import process_images, tokenizer_image_token
        from eve.model.builder import load_pretrained_model
        from eve.utils import disable_torch_init

        self._DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self._IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self._SeparatorStyle = SeparatorStyle
        self._conv_templates = conv_templates
        self._process_images = process_images
        self._tokenizer_image_token = tokenizer_image_token

        disable_torch_init()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_name = _resolve_repo_model_name(model_name_or_path, default="eve")
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_name_or_path,
            None,
            model_name,
            load_4bit=load_in_4bit,
            device_map="auto" if self.device == "cuda" else {"": self.device},
            device=self.device,
        )
        self.model.eval()
        self.conv_mode = "vicuna_v1"

    def generate_yes_no(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ) -> VLMGeneration:
        conv = self._conv_templates[self.conv_mode].copy()
        qs = f"{self._DEFAULT_IMAGE_TOKEN}\n{question}"
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_tensor = self._process_images([image], self.image_processor, None)[0]
        input_ids = self._tokenizer_image_token(
            prompt,
            self.tokenizer,
            self._IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        )
        device = _model_device(self.model)
        input_ids = input_ids.unsqueeze(0).to(device)
        image_tensor = image_tensor.to(device=device, dtype=torch.float16)

        stop_str = conv.sep if conv.sep_style != self._SeparatorStyle.TWO else conv.sep2
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "use_cache": True,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0),
                **gen_kwargs,
            )

        input_len = input_ids.shape[1]
        text = self.tokenizer.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[0]
        text = text.strip()
        if text.endswith(stop_str):
            text = text[: -len(stop_str)].strip()
        return VLMGeneration(raw_text=text, predicted_answer=_extract_yes_no(text))


class JanusAdapter:
    """Janus-Pro inference via deepseek-ai/Janus repo."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: Optional[str] = None,
        dtype: Optional[str] = "bfloat16",
        load_in_4bit: bool = False,
    ):
        add_repo_to_path("Janus")
        from janus.models import VLChatProcessor
        from transformers import AutoModelForCausalLM

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Janus vision stack is unstable with 4-bit weights (mixed bf16/fp16).
        load_in_4bit = False
        model_kwargs = _build_model_kwargs(
            device=self.device,
            dtype="float16",
            attn_implementation=None,
            trust_remote_code=True,
            load_in_4bit=load_in_4bit,
        )
        self.vl_chat_processor = VLChatProcessor.from_pretrained(model_name_or_path)
        self.tokenizer = self.vl_chat_processor.tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )
        self.model = self.model.to(dtype=torch.float16)
        self.model.eval()

    def generate_yes_no(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ) -> VLMGeneration:
        from janus.utils.io import load_pil_images

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp.name)
            image_path = tmp.name

        conversation = [
            {
                "role": "User",
                "content": "<image_placeholder>\n" + question,
                "images": [image_path],
            },
            {"role": "Assistant", "content": ""},
        ]

        device = _model_device(self.model)
        dtype = _model_dtype(self.model)
        pil_images = load_pil_images(conversation)
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
        ).to(device)
        if getattr(prepare_inputs, "pixel_values", None) is not None:
            prepare_inputs.pixel_values = prepare_inputs.pixel_values.to(dtype=dtype)

        with torch.inference_mode():
            inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
            outputs = self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                use_cache=True,
            )

        text = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        Path(image_path).unlink(missing_ok=True)
        return VLMGeneration(raw_text=text.strip(), predicted_answer=_extract_yes_no(text))
