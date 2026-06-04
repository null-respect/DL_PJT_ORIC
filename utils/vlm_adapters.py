from __future__ import annotations

import gc
import inspect
import re
from dataclasses import dataclass
from typing import Any, Optional

import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
)


_YES_RE = re.compile(r"\b(yes|yeah|yep|true)\b", re.IGNORECASE)
_NO_RE = re.compile(r"\b(no|nope|false)\b", re.IGNORECASE)
_SOLUTION_RE = re.compile(r"<solution>\s*(yes|no)\s*</solution>", re.IGNORECASE)

# Models whose tokenizer expects string content (image passed separately to processor).
_STRING_CHAT_MODEL_TYPES = frozenset({"chatglm", "internvl_chat"})


def _extract_yes_no(text: str) -> str:
    if not text:
        return "no"

    m = _SOLUTION_RE.search(text)
    if m:
        return m.group(1).lower()

    t = text.strip().lower()
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

    last_yes = t.rfind("yes")
    last_no = t.rfind("no")
    return "yes" if last_yes > last_no else "no"


def _resolve_dtype(dtype: Optional[str]) -> Optional[torch.dtype]:
    if dtype in (None, "auto"):
        return None
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _get_model_type(model_name_or_path: str, trust_remote_code: bool = True) -> str:
    cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    return getattr(cfg, "model_type", "") or ""


def _cuda_supports_flash_attn() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major >= 8


def _internvl_build_transform(input_size: int):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def _internvl_find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def _internvl_dynamic_preprocess(image: Image.Image, *, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = _internvl_find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def _internvl_pixel_values(image: Image.Image, model: Any, *, input_size=448, max_num=12) -> torch.Tensor:
    transform = _internvl_build_transform(input_size)
    images = _internvl_dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = torch.stack([transform(im) for im in images])
    dtype = next(model.parameters()).dtype
    device = _model_device(model)
    return pixel_values.to(device=device, dtype=dtype)


def _model_device(model: Any) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    if hasattr(model, "hf_device_map"):
        device_name = next(iter(model.hf_device_map.values()), "cpu")
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _force_phi3_eager_attention(model: Any) -> None:
    """Phi-3.5-V remote code defaults to FlashAttention2, which needs Ampere+ GPUs."""
    import importlib

    from transformers.models.clip.modeling_clip import CLIPAttention

    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if not layers:
        return

    mod = importlib.import_module(layers[0].self_attn.__class__.__module__)
    Phi3Attention = mod.Phi3Attention
    config = model.config
    config._attn_implementation = "eager"

    for layer_idx, layer in enumerate(layers):
        if "Flash" not in type(layer.self_attn).__name__:
            continue
        old_attn = layer.self_attn
        new_attn = Phi3Attention(config, layer_idx=layer_idx)
        new_attn.load_state_dict(old_attn.state_dict(), strict=False)
        device = next(old_attn.parameters()).device
        dtype = next(old_attn.parameters()).dtype
        new_attn.to(device=device, dtype=dtype)
        layer.self_attn = new_attn

    vision_embed = getattr(inner, "vision_embed_tokens", None)
    img_processor = getattr(vision_embed, "img_processor", None) if vision_embed else None
    encoder = getattr(getattr(img_processor, "vision_model", None), "encoder", None)
    if encoder is not None:
        clip_config = img_processor.config
        for layer in encoder.layers:
            if type(layer.self_attn).__name__ != "CLIPAttentionFA2":
                continue
            old_attn = layer.self_attn
            new_attn = CLIPAttention(clip_config)
            new_attn.load_state_dict(old_attn.state_dict(), strict=False)
            device = next(old_attn.parameters()).device
            dtype = next(old_attn.parameters()).dtype
            new_attn.to(device=device, dtype=dtype)
            layer.self_attn = new_attn


def _maybe_patch_chatglm_config(config: Any) -> None:
    """Transformers 4.5x DynamicCache expects num_hidden_layers on the text config."""
    if getattr(config, "num_hidden_layers", None) is not None:
        return
    num_layers = getattr(config, "num_layers", None)
    if num_layers is not None:
        config.num_hidden_layers = num_layers


def _build_model_kwargs(
    *,
    device: str,
    dtype: Optional[str],
    attn_implementation: Optional[str],
    trust_remote_code: bool,
    load_in_4bit: bool,
) -> dict[str, Any]:
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        compute_dtype = _resolve_dtype(dtype) or torch.bfloat16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        torch_dtype = _resolve_dtype(dtype)
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    if device == "cuda":
        model_kwargs["device_map"] = "auto"
    return model_kwargs


def release_vlm_adapter(adapter: Any) -> None:
    """Free GPU memory held by a loaded adapter."""
    for attr in ("model", "processor", "tokenizer", "image_processor", "vl_chat_processor"):
        obj = getattr(adapter, attr, None)
        if obj is not None:
            del obj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _ensure_internvl_language_model_generation_config(llm: Any) -> None:
    """Remote InternLM2 may leave generation_config unset; HF 4.5x generate() requires it."""
    if getattr(llm, "generation_config", None) is not None:
        return
    llm.generation_config = GenerationConfig.from_model_config(llm.config)


def _internvl_greedy_decode(
    llm: Any,
    *,
    input_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int,
) -> torch.Tensor:
    """Greedy decode without GenerationMixin.generate (missing on InternLM2 + TF 4.5x)."""
    generated: list[torch.Tensor] = []
    cur_embeds = input_embeds
    cur_attn = attention_mask

    for _ in range(max_new_tokens):
        outputs = llm(
            inputs_embeds=cur_embeds,
            attention_mask=cur_attn,
            use_cache=False,
            return_dict=True,
        )
        next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
        generated.append(next_token)
        if int(next_token.item()) == eos_token_id:
            break
        next_embed = llm.get_input_embeddings()(next_token)
        cur_embeds = torch.cat([cur_embeds, next_embed], dim=1)
        pad = torch.ones(
            (cur_attn.shape[0], 1),
            device=cur_attn.device,
            dtype=cur_attn.dtype,
        )
        cur_attn = torch.cat([cur_attn, pad], dim=1)

    if not generated:
        return prompt_input_ids
    return torch.cat([prompt_input_ids, torch.cat(generated, dim=1)], dim=1)


def _patch_internvl_chat_model_generate(model: Any) -> None:
    """Remote InternVL calls InternLM2.generate, which is absent on Transformers 4.5x."""
    if getattr(model, "_oric_chat_generate_patch", False):
        return
    from types import MethodType

    import torch

    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            bsz, seq_len, hidden = input_embeds.shape
            flat_embeds = input_embeds.reshape(bsz * seq_len, hidden)
            flat_ids = input_ids.reshape(bsz * seq_len)
            selected = flat_ids == self.img_context_token_id
            assert selected.sum() != 0
            flat_embeds[selected] = vit_embeds.reshape(-1, hidden).to(flat_embeds.device)
            input_embeds = flat_embeds.reshape(bsz, seq_len, hidden)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        max_new = int(generate_kwargs.get("max_new_tokens", 128))
        eos_id = int(generate_kwargs.get("eos_token_id", 0))
        return _internvl_greedy_decode(
            self.language_model,
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            prompt_input_ids=input_ids,
            max_new_tokens=max_new,
            eos_token_id=eos_id,
        )

    model.generate = MethodType(generate, model)
    model._oric_chat_generate_patch = True


def _patch_internvl_language_model_generate(model: Any) -> None:
    """InternVL3 uses InternLM2 remote code with Transformers 4.5x generation quirks."""
    llm = getattr(model, "language_model", None)
    if llm is not None:
        _ensure_internvl_language_model_generation_config(llm)
    _patch_internvl_chat_model_generate(model)


def _disable_internvl_flash_attention(model: Any) -> None:
    """InternVL ViT uses flash_attn directly; disable on pre-Ampere GPUs."""
    vision_cfg = getattr(model.config, "vision_config", None)
    if vision_cfg is not None:
        vision_cfg.use_flash_attn = False
    llm_cfg = getattr(model.config, "llm_config", None)
    if llm_cfg is not None:
        llm_cfg.attn_implementation = "eager"

    vision = getattr(model, "vision_model", None)
    encoder = getattr(vision, "encoder", None) if vision is not None else None
    if encoder is None:
        return
    for layer in encoder.layers:
        attn = getattr(layer, "attn", None)
        if attn is not None and hasattr(attn, "use_flash_attn"):
            attn.use_flash_attn = False


def _load_vlm_model(model_name_or_path: str, model_kwargs: dict[str, Any]):
    model_type = _get_model_type(
        model_name_or_path, trust_remote_code=model_kwargs.get("trust_remote_code", True)
    )

    if model_type == "llava_next":
        from transformers import LlavaNextForConditionalGeneration

        return LlavaNextForConditionalGeneration.from_pretrained(model_name_or_path, **model_kwargs), model_type

    if model_type == "emu3":
        from transformers import Emu3ForConditionalGeneration

        return Emu3ForConditionalGeneration.from_pretrained(model_name_or_path, **model_kwargs), model_type

    if model_type == "internvl_chat" and not _cuda_supports_flash_attn():
        model_kwargs = dict(model_kwargs)
        model_kwargs["use_flash_attn"] = False
        if not model_kwargs.get("attn_implementation"):
            model_kwargs["attn_implementation"] = "eager"

    internvl_kwargs = dict(model_kwargs)
    if model_type == "internvl_chat":
        internvl_kwargs.setdefault("trust_remote_code", True)

    try:
        model = AutoModelForImageTextToText.from_pretrained(model_name_or_path, **internvl_kwargs)
        if model_type == "internvl_chat":
            if not _cuda_supports_flash_attn():
                _disable_internvl_flash_attention(model)
            _patch_internvl_language_model_generate(model)
        return model, model_type
    except (ValueError, OSError, KeyError):
        pass

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **internvl_kwargs)
    if model_type == "internvl_chat":
        if not _cuda_supports_flash_attn():
            _disable_internvl_flash_attention(model)
        _patch_internvl_language_model_generate(model)
    return model, model_type


def _get_tokenizer(processor: Any) -> Any:
    tok = getattr(processor, "tokenizer", None) or getattr(processor, "_tokenizer", None)
    if tok is None:
        raise RuntimeError("Processor does not expose a tokenizer; cannot decode generations.")
    return tok


def _build_messages(image: Image.Image, question: str, model_type: str) -> list[dict]:
    if model_type in _STRING_CHAT_MODEL_TYPES:
        return [{"role": "user", "content": f"<image>\n{question}"}]
    if model_type == "phi3_v":
        return [{"role": "user", "content": f"<|image_1|>\n{question}"}]
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]


def _apply_chat_template(
    processor: Any,
    messages: list[dict],
    *,
    model_type: str,
    tokenize: bool = False,
    return_dict: bool = False,
) -> Any:
    if model_type == "phi3_v":
        tok = _get_tokenizer(processor)
        return tok.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=True,
            return_dict=return_dict,
        )

    return processor.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=True,
        return_dict=return_dict,
    )


def _filter_inputs_for_generate(model: Any, inputs: dict[str, Any]) -> dict[str, Any]:
    try:
        allowed = set(inspect.signature(model.forward).parameters)
    except (TypeError, ValueError):
        return inputs
    return {k: v for k, v in inputs.items() if k in allowed}


def _move_inputs_to_model(inputs: dict[str, Any], model: Any) -> dict[str, Any]:
    device = getattr(model, "device", None)
    if device is None and hasattr(model, "hf_device_map"):
        device = next(iter(model.hf_device_map.values()), None)
    if device is None:
        return inputs
    out: dict[str, Any] = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _decode_generation(
    processor: Any,
    output_ids: torch.Tensor,
    input_len: Optional[int] = None,
) -> str:
    if input_len is not None and input_len > 0:
        gen_ids = output_ids[input_len:]
    else:
        gen_ids = output_ids

    if hasattr(processor, "decode"):
        return processor.decode(gen_ids, skip_special_tokens=True).strip()

    tok = _get_tokenizer(processor)
    return tok.decode(gen_ids, skip_special_tokens=True).strip()


@dataclass
class VLMGeneration:
    raw_text: str
    predicted_answer: str


class Qwen3VLAdapter:
    """Minimal inference adapter for Qwen3-VL models using 🤗 Transformers."""

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        dtype: Optional[str] = "bfloat16",
        attn_implementation: Optional[str] = None,
        trust_remote_code: bool = True,
        load_in_4bit: bool = False,
    ):
        from transformers import Qwen3VLForConditionalGeneration

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_kwargs = _build_model_kwargs(
            device=self.device,
            dtype=dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            load_in_4bit=load_in_4bit,
        )

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        self.model_type = "qwen3_vl"

    def generate_yes_no(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ) -> VLMGeneration:
        messages = _build_messages(image, question, self.model_type)
        prompt = _apply_chat_template(
            self.processor, messages, model_type=self.model_type, tokenize=False
        )
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        return self._generate_from_inputs(inputs, max_new_tokens, temperature)

    def _generate_from_inputs(
        self,
        inputs: dict[str, Any],
        max_new_tokens: int,
        temperature: float,
    ) -> VLMGeneration:
        inputs = _move_inputs_to_model(inputs, self.model)
        input_len = inputs.get("input_ids").shape[-1] if "input_ids" in inputs else None

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        text = _decode_generation(self.processor.tokenizer, out[0], input_len)
        return VLMGeneration(raw_text=text, predicted_answer=_extract_yes_no(text))


class HFGenericVLMAdapter:
    """Generic HF VLM adapter (InternVL, LLaVA, Phi-Vision, GLM-4V, Emu3, etc.)."""

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        dtype: Optional[str] = "bfloat16",
        attn_implementation: Optional[str] = None,
        trust_remote_code: bool = True,
        load_in_4bit: bool = False,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_kwargs = _build_model_kwargs(
            device=self.device,
            dtype=dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            load_in_4bit=load_in_4bit,
        )

        self.model, self.model_type = _load_vlm_model(model_name_or_path, model_kwargs)
        self.model.eval()
        if self.model_type == "chatglm":
            _maybe_patch_chatglm_config(self.model.config)
        if self.model_type == "internvl_chat" and not _cuda_supports_flash_attn():
            _disable_internvl_flash_attention(self.model)
        if self.model_type == "internvl_chat":
            _patch_internvl_language_model_generate(self.model)
        if self.model_type == "phi3_v":
            force_eager = attn_implementation in ("eager", "sdpa") or not _cuda_supports_flash_attn()
            if force_eager and not load_in_4bit:
                inner = getattr(self.model, "model", None)
                layers = getattr(inner, "layers", None)
                if layers and "Flash" in type(layers[0].self_attn).__name__:
                    _force_phi3_eager_attention(self.model)

        if self.model_type in ("chatglm", "internvl_chat"):
            self.processor = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
            )
        else:
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
        if self.model_type == "emu3":
            return self._generate_emu3(image, question, max_new_tokens, temperature)
        if self.model_type == "chatglm":
            return self._generate_chatglm(image, question, max_new_tokens, temperature)
        if self.model_type == "internvl_chat":
            return self._generate_internvl(image, question, max_new_tokens, temperature)

        messages = _build_messages(image, question, self.model_type)
        prompt = _apply_chat_template(
            self.processor, messages, model_type=self.model_type, tokenize=False
        )
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = _filter_inputs_for_generate(self.model, inputs)
        inputs = _move_inputs_to_model(inputs, self.model)
        input_len = inputs.get("input_ids").shape[-1] if "input_ids" in inputs else None

        gen_kwargs = self._generation_kwargs(max_new_tokens, temperature)

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        if hasattr(self.processor, "decode") and input_len is not None:
            text = self.processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
        else:
            text = _decode_generation(_get_tokenizer(self.processor), out[0], input_len)

        return VLMGeneration(raw_text=text, predicted_answer=_extract_yes_no(text))

    def _generation_kwargs(self, max_new_tokens: int, temperature: float) -> dict[str, Any]:
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        # Remote/custom VLMs often break with Transformers 4.5x DynamicCache.
        if self.model_type in ("chatglm", "phi3_v", "internvl_chat"):
            gen_kwargs["use_cache"] = False
        return gen_kwargs

    def _generate_internvl(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int,
        temperature: float,
    ) -> VLMGeneration:
        pixel_values = _internvl_pixel_values(image, self.model)
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation_config["temperature"] = temperature

        with torch.no_grad():
            response = self.model.chat(
                self.processor,
                pixel_values,
                question,
                generation_config,
            )

        return VLMGeneration(raw_text=str(response).strip(), predicted_answer=_extract_yes_no(str(response)))

    def _generate_chatglm(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int,
        temperature: float,
    ) -> VLMGeneration:
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "image": image, "content": question}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = _filter_inputs_for_generate(self.model, inputs)
        inputs = _move_inputs_to_model(inputs, self.model)
        input_len = inputs.get("input_ids").shape[-1] if "input_ids" in inputs else None

        gen_kwargs = self._generation_kwargs(max_new_tokens, temperature)
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        text = _decode_generation(self.processor, out[0], input_len)
        return VLMGeneration(raw_text=text, predicted_answer=_extract_yes_no(text))

    def _generate_emu3(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int,
        temperature: float,
    ) -> VLMGeneration:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = _move_inputs_to_model(inputs, self.model)
        input_len = inputs.get("input_ids").shape[-1] if "input_ids" in inputs else None

        gen_kwargs = self._generation_kwargs(max_new_tokens, temperature)

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        if hasattr(self.processor, "decode"):
            text = self.processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
        else:
            text = _decode_generation(_get_tokenizer(self.processor), out[0], input_len)

        return VLMGeneration(raw_text=text, predicted_answer=_extract_yes_no(text))


def pick_adapter(
    model_family: str,
    model_name_or_path: str,
    *,
    load_in_4bit: bool = False,
    dtype: Optional[str] = "bfloat16",
    device: Optional[str] = None,
    attn_implementation: Optional[str] = None,
    detector_type: Optional[str] = None,
):
    mf = model_family.lower()
    if mf == "auto":
        name = model_name_or_path.lower()
        if "qwen3-vl" in name or "qwen3_vl" in name:
            mf = "qwen3_vl"
        elif "vila" in name:
            mf = "vila"
        elif "janus" in name:
            mf = "janus"
        elif "eve" in name:
            mf = "eve"
        elif "owlv2" in name or "owl" in name or "grounding-dino" in name or "grounding_dino" in name:
            mf = "detector"
        else:
            mf = "hf_generic"

    adapter_kwargs = {
        "model_name_or_path": model_name_or_path,
        "load_in_4bit": load_in_4bit,
        "dtype": dtype,
        "device": device,
        "attn_implementation": attn_implementation,
    }
    repo_kwargs = {
        "model_name_or_path": model_name_or_path,
        "load_in_4bit": load_in_4bit,
        "dtype": dtype,
        "device": device,
    }
    if mf == "qwen3_vl":
        return Qwen3VLAdapter(**adapter_kwargs)
    if mf == "hf_generic":
        return HFGenericVLMAdapter(**adapter_kwargs)
    if mf == "detector":
        from utils.detector_adapters import pick_detector_adapter

        return pick_detector_adapter(
            model_name_or_path,
            detector_type=detector_type or "auto",
            load_in_4bit=load_in_4bit,
            dtype=dtype,
            device=device,
        )
    if mf == "vila":
        from utils.repo_adapters import VilaAdapter

        return VilaAdapter(**repo_kwargs)
    if mf == "janus":
        from utils.repo_adapters import JanusAdapter

        return JanusAdapter(**repo_kwargs)
    if mf == "eve":
        from utils.repo_adapters import EveAdapter

        return EveAdapter(**repo_kwargs)

    raise ValueError(
        f"Unsupported model_family: {model_family}. "
        "Use auto/qwen3_vl/hf_generic/vila/janus/eve/detector."
    )
