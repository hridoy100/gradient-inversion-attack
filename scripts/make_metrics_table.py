#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional


METRIC_KEYS = ("mse", "psnr", "feature_similarity", "class_accuracy")
CSV_ALIASES = {
    "class_accuracy": ("class_accuracy", "correct"),
    "feature_similarity": ("feature_similarity", "cosine_similarity"),
    "mse": ("mse",),
    "psnr": ("psnr",),
}


def _mean(values):
    return sum(values) / max(len(values), 1)


def load_means_from_metrics_csv(metrics_csv: Path) -> Dict[str, Dict[str, float]]:
    with metrics_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        by_mode = {}
        for row in reader:
            mode = (row.get("mode") or "").strip()
            if not mode:
                continue
            by_mode.setdefault(mode, {k: [] for k in METRIC_KEYS})
            for k in METRIC_KEYS:
                for col in CSV_ALIASES.get(k, (k,)):
                    if col in row and row[col] != "":
                        by_mode[mode][k].append(float(row[col]))
                        break

    out = {}
    for mode, metrics in by_mode.items():
        out[mode] = {k: _mean(metrics[k]) for k in METRIC_KEYS if metrics[k]}
    return out


def load_means_from_metrics_summary(metrics_summary_json: Path) -> Dict[str, Dict[str, float]]:
    obj = json.loads(metrics_summary_json.read_text())
    out = {}

    per_client = obj.get("per_client") or {}
    if per_client:
        # Weight each client equally by sample count if present; otherwise simple mean.
        client_means = []
        weights = []
        for _, summary in per_client.items():
            count = float(summary.get("count", 1))
            client_means.append({k: float(summary[k]["mean"]) for k in METRIC_KEYS if k in summary})
            weights.append(count)
        if client_means:
            wsum = sum(weights) or 1.0
            out["per-client"] = {
                k: sum(cm.get(k, 0.0) * w for cm, w in zip(client_means, weights)) / wsum
                for k in METRIC_KEYS
            }

    aggregated = obj.get("aggregated")
    if aggregated:
        out["aggregated"] = {k: float(aggregated[k]["mean"]) for k in METRIC_KEYS if k in aggregated}
    return out


def load_run_means(run_dir: Path) -> Dict[str, Dict[str, float]]:
    metrics_csv = run_dir / "metrics.csv"
    if metrics_csv.exists():
        return load_means_from_metrics_csv(metrics_csv)

    metrics_summary_json = run_dir / "metrics_summary.json"
    if metrics_summary_json.exists():
        return load_means_from_metrics_summary(metrics_summary_json)

    raise FileNotFoundError(f"Expected {metrics_csv} or {metrics_summary_json} to exist.")


def fmt(val: Optional[float], kind: str) -> str:
    if val is None:
        return "--"
    if kind == "psnr":
        return f"{val:.2f}"
    if kind == "class_accuracy":
        return f"{val:.3f}"
    return f"{val:.4f}"


def pick_mode(means: Dict[str, Dict[str, float]], preferred: str) -> Optional[Dict[str, float]]:
    if preferred in means:
        return means[preferred]
    return None


def main():
    p = argparse.ArgumentParser(description="Create a LaTeX metrics table from reconstruction run folders.")
    p.add_argument(
        "--dlg-run",
        type=str,
        required=True,
        help="Run directory produced by `main.py` (e.g. outputs/reconstructions/run_YYYYMMDD_HHMMSS).",
    )
    p.add_argument(
        "--diffusion-run",
        type=str,
        default=None,
        help="Run directory produced by `evaluations/diffusion_reconstruction/run.py` (optional).",
    )
    p.add_argument(
        "--diffusion-mode",
        type=str,
        default="per-client",
        choices=["per-client", "aggregated"],
        help="Which diffusion mode to use for the table row (if present).",
    )
    args = p.parse_args()

    dlg_means = load_run_means(Path(args.dlg_run))
    per_client = pick_mode(dlg_means, "per-client")
    aggregated = pick_mode(dlg_means, "aggregated")

    diffusion = None
    if args.diffusion_run:
        diffusion_means = load_run_means(Path(args.diffusion_run))
        diffusion = pick_mode(diffusion_means, args.diffusion_mode) or diffusion_means.get("aggregated") or diffusion_means.get("per-client")

    def row(name: str, m: Optional[Dict[str, float]]) -> str:
        return (
            f"{name} & {fmt(m.get('mse') if m else None, 'mse')} & {fmt(m.get('psnr') if m else None, 'psnr')} & "
            f"{fmt(m.get('feature_similarity') if m else None, 'feature_similarity')} & {fmt(m.get('class_accuracy') if m else None, 'class_accuracy')} \\\\"
        )

    print("\\caption{Reconstruction performance (mean across samples).}")
    print("\\label{tab:metrics}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("Method & MSE & PSNR & Cosine Sim. & Label Acc. \\\\")
    print("\\midrule")
    print(row("Per-client inversion", per_client))
    print(row("Aggregated gradients", aggregated))
    print(row("Diffusion-guided reconstruction", diffusion))
    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    main()
