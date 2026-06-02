import os
import json
import argparse
from sklearn.metrics import classification_report, confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate binary classification results from JSON file.")
    parser.add_argument(
        "--result_path",
        type=str,
        required=False,
        default="",
        help="Path to the input JSON file containing predictions.",
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default="",
        help="If set, evaluate all predictions_*.json under this directory (recursively).",
    )

    parser.add_argument(
        "--models_file",
        type=str,
        default="",
        help=(
            "Optional. JSON list of model entries (like models_requested.json). "
            "If set with --results_dir, evaluations are mapped to each entry's `name`."
        ),
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="If set with --models_file, fail if any model's prediction file is missing.",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="predictions_*.json",
        help="Glob pattern to match prediction files inside --results_dir.",
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        default="./results",
        help="Folder to save evaluation results.",
    )
    return parser.parse_args()


def compute_metrics(y_true, y_pred):
    report = classification_report(
        y_true,
        y_pred,
        labels=["no", "yes"],
        target_names=["no", "yes"],
        zero_division=0,
        output_dict=True,
    )

    f1_yes = round(report["yes"]["f1-score"] * 100, 2)
    f1_no = round(report["no"]["f1-score"] * 100, 2)

    metrics = {
        "yes": {
            "precision": round(report["yes"]["precision"] * 100, 2),
            "recall": round(report["yes"]["recall"] * 100, 2),
            "f1": f1_yes,
        },
        "no": {
            "precision": round(report["no"]["precision"] * 100, 2),
            "recall": round(report["no"]["recall"] * 100, 2),
            "f1": f1_no,
        },
        "macro": {
            "precision": round(report["macro avg"]["precision"] * 100, 2),
            "recall": round(report["macro avg"]["recall"] * 100, 2),
            "f1": round(report["macro avg"]["f1-score"] * 100, 2),
        },
        "yes_proportion": (
            round(y_pred.count("yes") / len(y_pred) * 100, 2) if y_pred else 0.0
        ),
    }
    return metrics


def load_solutions(data_json):
    y_true = [d["solution"] for d in data_json]
    y_pred = [d["predicted_answer"] for d in data_json]
    return y_true, y_pred


def load_predictions_file(result_path: str):
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Result file not found: {result_path}")

    if os.path.getsize(result_path) == 0:
        raise ValueError(
            "Result file is empty (0 bytes). "
            "Did inference fail or write to a different path? "
            "If you used infer.py defaults, try: --result_path predictions.json"
        )

    try:
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse JSON from: {result_path} ({e}). "
            "Expected a JSON list of objects like "
            "[{'predicted_answer': 'yes'|'no', 'solution': 'yes'|'no', ...}, ...]. "
            "If your file is JSONL, convert to a JSON array first."
        ) from e

    if isinstance(data, dict) and "predictions" in data:
        data = data["predictions"]

    if not isinstance(data, list):
        raise ValueError(
            f"Unexpected JSON top-level type: {type(data).__name__}. "
            "Expected a JSON list of prediction objects."
        )

    return data


def save_results(results, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    json_path = os.path.join(output_folder, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {json_path}")

def _slug_filename(path: str) -> str:
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]
    out = []
    for ch in name.lower():
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_") or "model"


def _slug_name(name: str) -> str:
    out = []
    for ch in name.strip().lower():
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_") or "model"


def _iter_prediction_files(root: str, pattern: str):
    import glob

    # recursive glob
    glob_pat = os.path.join(root, "**", pattern)
    for p in sorted(glob.glob(glob_pat, recursive=True)):
        if os.path.isfile(p):
            yield p


def main():
    args = parse_args()

    # Multi-file mode
    if args.results_dir:
        os.makedirs(args.output_folder, exist_ok=True)
        summary: list[dict] = []

        if args.models_file:
            with open(args.models_file, "r", encoding="utf-8") as f:
                models = json.load(f)
            if not isinstance(models, list) or not models:
                raise ValueError("--models_file must be a non-empty JSON list")

            for m in models:
                name = str(m.get("name") or "").strip()
                if not name:
                    continue
                if m.get("run") is False:
                    summary.append(
                        {
                            "model_name": name,
                            "status": "skipped",
                            "reason": "run=false",
                            "note": m.get("note", ""),
                        }
                    )
                    continue
                family = str(m.get("family") or "").strip()
                if family in ("openai_api", "detector"):
                    summary.append(
                        {
                            "model_name": name,
                            "status": "skipped",
                            "reason": f"family={family}",
                            "note": m.get("note", ""),
                        }
                    )
                    continue

                slug = _slug_name(name)

                # expected file name from infer.py
                expected = os.path.join(args.results_dir, f"predictions_{slug}.json")
                if not os.path.exists(expected):
                    matches = list(_iter_prediction_files(args.results_dir, f"predictions_{slug}.json"))
                    if matches:
                        expected = matches[0]

                if not os.path.exists(expected):
                    msg = (
                        f"Missing predictions for model '{name}' "
                        f"(expected predictions_{slug}.json under {args.results_dir})"
                    )
                    if args.strict:
                        raise FileNotFoundError(msg)
                    summary.append({"model_name": name, "status": "missing", "message": msg})
                    continue

                data_json = load_predictions_file(expected)
                y_true, y_pred = load_solutions(data_json)
                results = compute_metrics(y_true, y_pred)

                out_dir = os.path.join(args.output_folder, slug)
                save_results(results, out_dir)
                summary.append(
                    {
                        "model_name": name,
                        "model_key": slug,
                        "file": expected,
                        "status": "ok",
                        "metrics": results,
                    }
                )
        else:
            pred_files = list(_iter_prediction_files(args.results_dir, args.pattern))
            if not pred_files:
                raise FileNotFoundError(
                    f"No files matched pattern '{args.pattern}' under: {args.results_dir}"
                )

            for p in pred_files:
                data_json = load_predictions_file(p)
                y_true, y_pred = load_solutions(data_json)
                results = compute_metrics(y_true, y_pred)

                model_key = _slug_filename(p)
                out_dir = os.path.join(args.output_folder, model_key)
                save_results(results, out_dir)
                summary.append(
                    {
                        "file": p,
                        "model_key": model_key,
                        "status": "ok",
                        "metrics": results,
                    }
                )

        summary_path = os.path.join(args.output_folder, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Summary saved to {summary_path}")
        return

    # Single-file mode
    if not args.result_path:
        raise ValueError("Either --result_path or --results_dir must be provided.")

    data_json = load_predictions_file(args.result_path)
    y_true, y_pred = load_solutions(data_json)
    results = compute_metrics(y_true, y_pred)
    save_results(results, args.output_folder)


if __name__ == "__main__":
    main()
