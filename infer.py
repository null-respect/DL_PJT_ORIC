import os
import json
import argparse
from collections import Counter
from datetime import datetime
import traceback

from PIL import Image
from tqdm import tqdm

from utils.vlm_adapters import pick_adapter, release_vlm_adapter
from utils.prompt_strategies import (
    PROMPT_STRATEGIES,
    PromptStrategy,
    default_max_new_tokens,
    get_questions_for_example,
)
from evaluate import load_predictions_file, load_solutions, compute_metrics, save_results


def parse_args():
    p = argparse.ArgumentParser(description="Run VLM inference on ORIC-Bench and save predictions.json")
    p.add_argument("--bench_path", type=str, default="./dataset/oric_bench.json", help="Path to ORIC-Bench JSON.")
    p.add_argument("--image_dir", type=str, default="./dataset/val2014", help="Directory containing COCO val images.")
    # Single-model mode (default)
    p.add_argument(
        "--model_family",
        type=str,
        default="auto",
        choices=["auto", "qwen3_vl", "hf_generic"],
        help="Which adapter to use.",
    )
    p.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="HF model id or local checkpoint path.",
    )
    p.add_argument("--output_path", type=str, default="./predictions.json", help="Where to save predictions JSON.")
    # Multi-model mode
    p.add_argument(
        "--models_file",
        type=str,
        default="",
        help=(
            "Path to a JSON file describing multiple models to run sequentially. "
            "Format: see models_paper.json (ORIC Table 12: 18 LVLMs + 2 detectors)."
        ),
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="If set with --models_file, write per-model outputs into this directory.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="If set in multi-model mode, skip models whose output file already exists.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="If set in multi-model mode, re-run models even when output files already exist.",
    )
    p.add_argument(
        "--hf_token",
        type=str,
        default="",
        help="Optional Hugging Face token for gated models (sets HUGGINGFACE_HUB_TOKEN).",
    )
    p.add_argument(
        "--evaluate_each",
        action="store_true",
        help="If set (multi-model mode), run evaluate.py-equivalent after each model and save results per model.",
    )
    p.add_argument(
        "--eval_dir",
        type=str,
        default="",
        help="If set with --evaluate_each, write per-model eval results into this directory (default: <output_dir>/eval).",
    )
    p.add_argument("--num_prompts", type=int, default=1, choices=[1, 2, 3, 4], help="How many prompt variants to use.")
    p.add_argument(
        "--prompt_strategy",
        type=str,
        default="",
        choices=[""] + list(PROMPT_STRATEGIES),
        help=(
            "Optional prompt ablation strategy. When set, uses a single instruction-augmented "
            "prompt instead of --num_prompts ensemble. Choices: "
            + ", ".join(PROMPT_STRATEGIES)
        ),
    )
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=0, help="If >0, only run first N questions (for smoke tests).")
    p.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load models in 4-bit (bitsandbytes). Helps on 16GB GPUs. Per-model JSON can override.",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Model weight/compute dtype when not using 4-bit.",
    )
    p.add_argument(
        "--attn_implementation",
        type=str,
        default="",
        help="Attention backend (e.g. sdpa, eager). Use sdpa on GPUs without FlashAttention.",
    )
    return p.parse_args()


def majority_vote(answers: list[str]) -> str:
    c = Counter(answers)
    if c["yes"] == c["no"]:
        # deterministic tie-break: prefer "no" (conservative)
        return "no"
    return "yes" if c["yes"] > c["no"] else "no"


def _slug(s: str) -> str:
    s = s.strip().lower()
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        elif ch in ("-", "_", ".", "/"):
            out.append("_")
        else:
            out.append("_")
    res = "".join(out)
    while "__" in res:
        res = res.replace("__", "_")
    return res.strip("_") or "model"


def _map_answer_for_eval(answer: str, *, prompt_strategy: str) -> str:
    """Map model output to binary yes/no for ORIC evaluation."""
    if answer == "uncertain":
        return "no"
    return answer


def run_one_model(
    *,
    bench: list[dict],
    image_dir: str,
    adapter,
    num_prompts: int,
    max_new_tokens: int,
    temperature: float,
    prompt_strategy: str = "",
) -> list[dict]:
    strategy: PromptStrategy | None = prompt_strategy or None
    effective_max_new_tokens = (
        default_max_new_tokens(strategy, max_new_tokens) if strategy else max_new_tokens
    )

    preds = []
    for ex in tqdm(bench, desc="Infer ORIC"):
        qid = str(ex["id"])
        img_path = os.path.join(image_dir, ex["image"])
        image = Image.open(img_path).convert("RGB")

        if hasattr(adapter, "predict_presence"):
            predicted, _raw = adapter.predict_presence(
                image=image, target_object=ex["target_object"]
            )
            pred_entry = {
                "question_id": qid,
                "predicted_answer": predicted,
                "solution": ex["solution"],
            }
        else:
            if strategy:
                problems = get_questions_for_example(ex, strategy)
            else:
                problems = ex["problem"][:num_prompts]

            answers = []
            raw_texts = []
            for q in problems:
                gen = adapter.generate_yes_no(
                    image=image,
                    question=q,
                    max_new_tokens=effective_max_new_tokens,
                    temperature=temperature,
                )
                answers.append(gen.predicted_answer)
                raw_texts.append(gen.raw_text)

            raw_answer = answers[0] if len(answers) == 1 else majority_vote(answers)
            binary_answers = [
                _map_answer_for_eval(a, prompt_strategy=prompt_strategy) for a in answers
            ]
            predicted = (
                majority_vote(binary_answers) if len(binary_answers) > 1 else binary_answers[0]
            )
            pred_entry = {
                "question_id": qid,
                "predicted_answer": predicted,
                "solution": ex["solution"],
            }
            if strategy:
                pred_entry["prompt_strategy"] = strategy
                pred_entry["prompt"] = problems[0]
                pred_entry["raw_answer"] = raw_answer
                if len(raw_texts) == 1:
                    pred_entry["raw_text"] = raw_texts[0]

        preds.append(pred_entry)
    return preds


def main():
    args = parse_args()

    if args.hf_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token
        os.environ["HF_TOKEN"] = args.hf_token

    with open(args.bench_path, "r", encoding="utf-8") as f:
        bench = json.load(f)

    if args.limit and args.limit > 0:
        bench = bench[: args.limit]

    # Multi-model mode
    if args.models_file:
        with open(args.models_file, "r", encoding="utf-8") as f:
            models = json.load(f)
        if not isinstance(models, list) or not models:
            raise ValueError("--models_file must be a non-empty JSON list")

        if not args.output_dir:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output_dir = os.path.join(".", "predictions_runs", ts)
        os.makedirs(args.output_dir, exist_ok=True)

        if args.evaluate_each and not args.eval_dir:
            args.eval_dir = os.path.join(args.output_dir, "eval")
        if args.eval_dir:
            os.makedirs(args.eval_dir, exist_ok=True)

        run_log_path = os.path.join(args.output_dir, "run_log.json")
        run_log: list[dict] = []

        for m in models:
            name = str(m.get("name") or m.get("model_name_or_path") or "model")
            family = str(m.get("family") or "auto")
            model_name_or_path = str(m.get("model_name_or_path") or "")

            if m.get("run") is False:
                print(f"[skip] {name}: run=false ({m.get('note', '')})")
                run_log.append(
                    {
                        "name": name,
                        "family": family,
                        "model_name_or_path": model_name_or_path,
                        "status": "skipped",
                        "reason": "run=false",
                        "note": m.get("note", ""),
                    }
                )
                continue
            if family == "openai_api":
                print(f"[skip] {name}: family={family} not implemented in infer.py ({m.get('note', '')})")
                run_log.append(
                    {
                        "name": name,
                        "family": family,
                        "model_name_or_path": model_name_or_path,
                        "status": "skipped",
                        "reason": f"family={family}",
                        "note": m.get("note", ""),
                    }
                )
                continue
            if not model_name_or_path:
                raise ValueError(f"Missing model_name_or_path for entry: {m}")

            out_path = os.path.join(args.output_dir, f"predictions_{_slug(name)}.json")
            if args.resume and not args.force and os.path.exists(out_path):
                print(f"[resume] Skip {name} (exists): {out_path}")
                run_log.append(
                    {
                        "name": name,
                        "family": family,
                        "model_name_or_path": model_name_or_path,
                        "status": "skipped",
                        "reason": "resume_exists",
                        "output_path": out_path,
                    }
                )
                continue

            load_in_4bit = bool(m.get("load_in_4bit", args.load_in_4bit))
            dtype = str(m.get("dtype") or args.dtype)
            attn_impl = str(m.get("attn_implementation") or args.attn_implementation or "")
            print(
                f"Running model: name={name} family={family} model={model_name_or_path}"
                + (f" load_in_4bit={load_in_4bit}" if load_in_4bit else "")
            )
            adapter = None
            try:
                adapter = pick_adapter(
                    family,
                    model_name_or_path,
                    load_in_4bit=load_in_4bit,
                    dtype=dtype,
                    attn_implementation=attn_impl or None,
                    detector_type=str(m.get("detector_type") or "auto"),
                )
                preds = run_one_model(
                    bench=bench,
                    image_dir=args.image_dir,
                    adapter=adapter,
                    num_prompts=args.num_prompts,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    prompt_strategy=args.prompt_strategy,
                )

                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(preds, f, indent=2, ensure_ascii=False)
                print(f"Saved {len(preds)} predictions to {out_path}")

                eval_out_folder = ""
                if args.evaluate_each:
                    eval_out_folder = os.path.join(args.eval_dir, _slug(name))
                    data_json = load_predictions_file(out_path)
                    y_true, y_pred = load_solutions(data_json)
                    results = compute_metrics(y_true, y_pred)
                    save_results(results, eval_out_folder)

                run_log.append(
                    {
                        "name": name,
                        "family": family,
                        "model_name_or_path": model_name_or_path,
                        "status": "ok",
                        "output_path": out_path,
                        "num_predictions": len(preds),
                        "eval_output_folder": eval_out_folder or None,
                        "load_in_4bit": load_in_4bit,
                    }
                )
            except Exception as e:
                msg = str(e)
                gated_hint = (
                    "gated repo" in msg.lower()
                    or "cannot access gated repo" in msg.lower()
                    or "forbidden" in msg.lower()
                    or "403" in msg
                )
                print(f"[error] {name}: {type(e).__name__}: {msg}")
                if gated_hint:
                    print(
                        f"[hint] {name} looks gated on Hugging Face. "
                        "Request access on the model page and/or run `huggingface-cli login`, "
                        "or pass `--hf_token` / set `HUGGINGFACE_HUB_TOKEN`."
                    )
                run_log.append(
                    {
                        "name": name,
                        "family": family,
                        "model_name_or_path": model_name_or_path,
                        "status": "error",
                        "error_type": type(e).__name__,
                        "error_message": msg,
                        "traceback": traceback.format_exc(limit=30),
                        "load_in_4bit": load_in_4bit,
                    }
                )
            finally:
                if adapter is not None:
                    release_vlm_adapter(adapter)

            with open(run_log_path, "w", encoding="utf-8") as f:
                json.dump(run_log, f, indent=2, ensure_ascii=False)
        return

    # Single-model mode
    adapter = pick_adapter(
        args.model_family,
        args.model_name_or_path,
        load_in_4bit=args.load_in_4bit,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation or None,
    )
    try:
        preds = run_one_model(
            bench=bench,
            image_dir=args.image_dir,
            adapter=adapter,
            num_prompts=args.num_prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            prompt_strategy=args.prompt_strategy,
        )

        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(preds, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(preds)} predictions to {args.output_path}")
    finally:
        release_vlm_adapter(adapter)


if __name__ == "__main__":
    main()

