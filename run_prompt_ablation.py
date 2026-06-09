#!/usr/bin/env python3
"""Run ORIC prompt-strategy ablation across models and strategies."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import datetime

from evaluate import compute_metrics, load_predictions_file, load_solutions, save_results
from infer import _slug, run_one_model
from utils.prompt_strategies import PROMPT_STRATEGIES, PROMPT_STRATEGY_DESCRIPTIONS
from utils.vlm_adapters import pick_adapter, release_vlm_adapter

SKIP_FAMILIES = frozenset({"openai_api", "detector"})


def parse_args():
    p = argparse.ArgumentParser(
        description="Run prompt-strategy ablation on ORIC-Bench (all models × all strategies)."
    )
    p.add_argument("--bench_path", type=str, default="./dataset/oric_bench.json")
    p.add_argument("--image_dir", type=str, default="./dataset/val2014")
    # Single-model mode
    p.add_argument("--model_family", type=str, default="auto")
    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    # Multi-model mode (same as infer.py)
    p.add_argument(
        "--models_file",
        type=str,
        default="",
        help="JSON list of models (e.g. models_requested.json). Runs every strategy per model.",
    )
    p.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=list(PROMPT_STRATEGIES),
        choices=list(PROMPT_STRATEGIES),
        help="Prompt strategies to compare.",
    )
    p.add_argument("--output_dir", type=str, default="", help="Root dir for preds/eval outputs.")
    p.add_argument("--limit", type=int, default=0, help="Smoke test: only first N questions.")
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--attn_implementation", type=str, default="")
    p.add_argument("--hf_token", type=str, default="")
    p.add_argument("--resume", action="store_true", help="Skip existing prediction files.")
    p.add_argument("--force", action="store_true", help="Re-run even when prediction files exist.")
    return p.parse_args()


def _evaluate_strategy(pred_path: str, eval_out: str) -> dict:
    data = load_predictions_file(pred_path)
    y_true, y_pred = load_solutions(data)
    metrics = compute_metrics(y_true, y_pred)
    uncertain_count = sum(1 for d in data if d.get("raw_answer") == "uncertain")
    metrics["uncertain_count"] = uncertain_count
    metrics["uncertain_rate"] = (
        round(uncertain_count / len(data) * 100, 2) if data else 0.0
    )
    save_results(metrics, eval_out)
    return metrics


def _run_strategies_for_model(
    *,
    model_entry: dict,
    bench: list[dict],
    args,
    preds_root: str,
    eval_root: str,
) -> list[dict]:
    name = str(model_entry.get("name") or model_entry.get("model_name_or_path") or "model")
    family = str(model_entry.get("family") or "auto")
    model_name_or_path = str(model_entry.get("model_name_or_path") or "")
    model_slug = _slug(name)

    if model_entry.get("run") is False:
        return [
            {
                "model_name": name,
                "model_slug": model_slug,
                "strategy": "*",
                "status": "skipped",
                "reason": "run=false",
                "note": model_entry.get("note", ""),
            }
        ]

    if family in SKIP_FAMILIES:
        return [
            {
                "model_name": name,
                "model_slug": model_slug,
                "strategy": "*",
                "status": "skipped",
                "reason": f"family={family}",
                "note": model_entry.get("note", ""),
            }
        ]

    if not model_name_or_path:
        raise ValueError(f"Missing model_name_or_path for entry: {model_entry}")

    load_in_4bit = bool(model_entry.get("load_in_4bit", args.load_in_4bit))
    dtype = str(model_entry.get("dtype") or args.dtype)
    attn_impl = str(model_entry.get("attn_implementation") or args.attn_implementation or "")

    model_preds_dir = os.path.join(preds_root, model_slug)
    model_eval_dir = os.path.join(eval_root, model_slug)
    os.makedirs(model_preds_dir, exist_ok=True)
    os.makedirs(model_eval_dir, exist_ok=True)

    pending = []
    for strategy in args.strategies:
        pred_path = os.path.join(model_preds_dir, f"predictions_{strategy}.json")
        if args.resume and not args.force and os.path.exists(pred_path) and os.path.getsize(pred_path) > 0:
            pending.append((strategy, pred_path, True))
        else:
            pending.append((strategy, pred_path, False))

    need_infer = [item for item in pending if not item[2]]
    results: list[dict] = []

    adapter = None
    try:
        if need_infer:
            print(
                f"Loading model: name={name} family={family} path={model_name_or_path}"
                + (f" load_in_4bit={load_in_4bit}" if load_in_4bit else "")
            )
            adapter = pick_adapter(
                family,
                model_name_or_path,
                load_in_4bit=load_in_4bit,
                dtype=dtype,
                attn_implementation=attn_impl or None,
                detector_type=str(model_entry.get("detector_type") or "auto"),
            )

            for strategy, pred_path, skipped in pending:
                if skipped:
                    print(f"[resume] {name} / {strategy}: {pred_path}")
                    continue

                print(f"Running {name} / strategy={strategy}")
                preds = run_one_model(
                    bench=bench,
                    image_dir=args.image_dir,
                    adapter=adapter,
                    num_prompts=1,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    prompt_strategy=strategy,
                )
                with open(pred_path, "w", encoding="utf-8") as f:
                    json.dump(preds, f, indent=2, ensure_ascii=False)
                print(f"Saved {len(preds)} predictions to {pred_path}")

        for strategy, pred_path, skipped in pending:
            eval_out = os.path.join(model_eval_dir, strategy)
            try:
                metrics = _evaluate_strategy(pred_path, eval_out)
                results.append(
                    {
                        "model_name": name,
                        "model_slug": model_slug,
                        "family": family,
                        "model_name_or_path": model_name_or_path,
                        "strategy": strategy,
                        "description": PROMPT_STRATEGY_DESCRIPTIONS[strategy],
                        "status": "ok",
                        "inference_skipped": skipped,
                        "prediction_path": pred_path,
                        "eval_path": os.path.join(eval_out, "results.json"),
                        "metrics": metrics,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "model_name": name,
                        "model_slug": model_slug,
                        "strategy": strategy,
                        "status": "error",
                        "prediction_path": pred_path,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    }
                )
    except Exception as e:
        msg = str(e)
        print(f"[error] {name}: {type(e).__name__}: {msg}")
        results.append(
            {
                "model_name": name,
                "model_slug": model_slug,
                "strategy": "*",
                "status": "error",
                "error_type": type(e).__name__,
                "error_message": msg,
                "traceback": traceback.format_exc(limit=30),
            }
        )
    finally:
        if adapter is not None:
            release_vlm_adapter(adapter)

    return results


def _run_single_model(args, bench: list[dict], preds_root: str, eval_root: str) -> list[dict]:
    model_entry = {
        "name": _slug(args.model_name_or_path),
        "family": args.model_family,
        "model_name_or_path": args.model_name_or_path,
        "load_in_4bit": args.load_in_4bit,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
    }
    return _run_strategies_for_model(
        model_entry=model_entry,
        bench=bench,
        args=args,
        preds_root=preds_root,
        eval_root=eval_root,
    )


def main():
    args = parse_args()

    if args.hf_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token
        os.environ["HF_TOKEN"] = args.hf_token

    with open(args.bench_path, "r", encoding="utf-8") as f:
        bench = json.load(f)
    if args.limit > 0:
        bench = bench[: args.limit]

    if not args.output_dir:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(".", "prompt_ablation_runs", ts)
    os.makedirs(args.output_dir, exist_ok=True)

    preds_root = os.path.join(args.output_dir, "predictions")
    eval_root = os.path.join(args.output_dir, "eval")
    os.makedirs(preds_root, exist_ok=True)
    os.makedirs(eval_root, exist_ok=True)

    all_results: list[dict] = []
    run_log: list[dict] = []

    if args.models_file:
        with open(args.models_file, "r", encoding="utf-8") as f:
            models = json.load(f)
        if not isinstance(models, list) or not models:
            raise ValueError("--models_file must be a non-empty JSON list")

        for m in models:
            model_results = _run_strategies_for_model(
                model_entry=m,
                bench=bench,
                args=args,
                preds_root=preds_root,
                eval_root=eval_root,
            )
            all_results.extend(model_results)
            run_log.append(
                {
                    "model_name": str(m.get("name") or ""),
                    "strategies": [
                        r for r in model_results if r.get("strategy") != "*"
                    ],
                }
            )
            with open(os.path.join(args.output_dir, "run_log.json"), "w", encoding="utf-8") as f:
                json.dump(run_log, f, indent=2, ensure_ascii=False)
    else:
        all_results = _run_single_model(args, bench, preds_root, eval_root)

    meta = {
        "bench_path": args.bench_path,
        "image_dir": args.image_dir,
        "limit": args.limit,
        "strategies": args.strategies,
        "models_file": args.models_file or None,
        "model_name_or_path": None if args.models_file else args.model_name_or_path,
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": all_results}, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
