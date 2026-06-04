from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Owlv2ForObjectDetection, Owlv2Processor

from utils.vlm_adapters import VLMGeneration, _extract_yes_no


def _tensor_device(model) -> torch.device:
    return next(model.parameters()).device


@dataclass
class DetectorPrediction:
    predicted_answer: str
    raw_text: str
    score: float


def _build_detector_model_kwargs(
    *,
    device: str,
    dtype: Optional[str],
    load_in_4bit: bool,
) -> dict[str, Any]:
    from utils.vlm_adapters import _resolve_dtype

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if load_in_4bit:
        compute_dtype = _resolve_dtype(dtype) or torch.float16
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        if device == "cuda":
            model_kwargs["device_map"] = {"": 0}
    else:
        torch_dtype = _resolve_dtype(dtype) or torch.float16
        model_kwargs["torch_dtype"] = torch_dtype
    return model_kwargs


def _move_floating_inputs_to_model_dtype(inputs: dict[str, Any], model: Any) -> dict[str, Any]:
    model_dtype = next(model.parameters()).dtype
    out: dict[str, Any] = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            out[k] = v.to(dtype=model_dtype)
        else:
            out[k] = v
    return out


class OWLv2DetectorAdapter:
    """Open-vocabulary detector: predict object presence via OWLv2 text-conditioned detection."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: Optional[str] = None,
        dtype: Optional[str] = "bfloat16",
        load_in_4bit: bool = False,
        score_threshold: float = 0.1,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.score_threshold = score_threshold
        model_kwargs = _build_detector_model_kwargs(
            device=self.device,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        self.processor = Owlv2Processor.from_pretrained(model_name_or_path)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name_or_path, **model_kwargs)
        if not load_in_4bit:
            self.model = self.model.to(self.device)
        self.model.eval()

    def predict_presence(self, image: Image.Image, target_object: str) -> tuple[str, str]:
        texts = [[target_object]]
        inputs = self.processor(text=texts, images=image, return_tensors="pt")
        device = _tensor_device(self.model)
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        inputs = _move_floating_inputs_to_model_dtype(inputs, self.model)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]], device=device)
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self.score_threshold,
        )[0]

        scores = results.get("scores", [])
        max_score = float(max(scores, default=0.0))
        predicted = "yes" if len(scores) > 0 else "no"
        raw = f"detections={len(scores)} max_score={max_score:.4f}"
        return predicted, raw


class GroundingDinoDetectorAdapter:
    """Zero-shot open-vocabulary detector via Grounding DINO (HF weights)."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: Optional[str] = None,
        dtype: Optional[str] = "bfloat16",
        load_in_4bit: bool = False,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        # Grounding DINO is unstable with 4-bit quantization on many GPUs.
        load_in_4bit = False
        model_kwargs = _build_detector_model_kwargs(
            device=self.device,
            dtype="float32",
            load_in_4bit=load_in_4bit,
        )
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict_presence(self, image: Image.Image, target_object: str) -> tuple[str, str]:
        text = target_object.strip().lower()
        if not text.endswith("."):
            text = f"{text}."

        inputs = self.processor(images=image, text=text, return_tensors="pt")
        device = _tensor_device(self.model)
        model_dtype = next(self.model.parameters()).dtype
        inputs = {
            k: (
                v.to(device=device, dtype=model_dtype)
                if isinstance(v, torch.Tensor) and v.is_floating_point()
                else v.to(device) if isinstance(v, torch.Tensor) else v
            )
            for k, v in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        scores = [float(score) for score in results.get("scores", [])]
        max_score = max(scores, default=0.0)
        predicted = "yes" if scores else "no"
        raw = f"detections={len(scores)} max_score={max_score:.4f}"
        return predicted, raw

    def generate_yes_no(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ) -> VLMGeneration:
        del max_new_tokens, temperature
        predicted, raw = self.predict_presence(image, question)
        return VLMGeneration(raw_text=raw, predicted_answer=predicted)


def pick_detector_adapter(
    model_name_or_path: str,
    *,
    detector_type: str = "auto",
    load_in_4bit: bool = False,
    dtype: Optional[str] = "bfloat16",
    device: Optional[str] = None,
):
    name = model_name_or_path.lower()
    kind = detector_type.lower()
    if kind == "auto":
        if "owlv2" in name or "owl" in name:
            kind = "owlv2"
        else:
            kind = "grounding_dino"

    common = {
        "model_name_or_path": model_name_or_path,
        "load_in_4bit": load_in_4bit,
        "dtype": dtype,
        "device": device,
    }
    if kind == "owlv2":
        return OWLv2DetectorAdapter(**common)
    if kind in ("grounding_dino", "grounding-dino", "dino"):
        return GroundingDinoDetectorAdapter(**common)
    raise ValueError(f"Unsupported detector_type: {detector_type}")
