import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path


def parse_float_list(text):
    text = str(text).strip()
    if not text:
        raise ValueError("empty float list")

    if ":" in text:
        parts = text.split(":")
        if len(parts) != 3:
            raise ValueError(f"range form must be start:step:end, got: {text}")
        start, step, end = [float(item) for item in parts]
        if step <= 0:
            raise ValueError(f"step must be positive, got: {step}")
        values = []
        cur = start
        eps = abs(step) * 1e-6 + 1e-9
        while cur <= end + eps:
            values.append(round(cur, 10))
            cur += step
        if len(values) == 0:
            raise ValueError(f"invalid range specification: {text}")
        return values

    parts = text.replace(",", " ").split()
    values = [float(item) for item in parts]
    if len(values) == 0:
        raise ValueError("empty float list")
    return values


def metric_display_name(metric_key):
    mapping = {
        "mota": "MOTA",
        "motp": "MOTP",
        "idf1": "IDF1",
        "precision": "Precision",
        "recall": "Recall",
        "fp": "FP",
        "fn": "FN",
        "id_switches": "IDS",
        "hota": "HOTA",
    }
    return mapping.get(metric_key, metric_key)


def maybe_add(cmd, flag, value):
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
        return
    if isinstance(value, (list, tuple)):
        cmd.append(flag)
        cmd.extend([str(v) for v in value])
        return
    cmd.extend([flag, str(value)])


def run_single_eval(args, u3d_value, u2d_value, run_dir):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "eval_udca_policy.py"),
        "--cache_dir",
        args.cache_dir,
        "--gt_pkl",
        args.gt_pkl,
        "--save_dir",
        str(run_dir),
        "--u3d_high",
        str(u3d_value),
        "--u2d_high",
        str(u2d_value),
    ]

    maybe_add(cmd, "--data_cfg", args.data_cfg)
    maybe_add(cmd, "--dataset_preset", args.dataset_preset)
    maybe_add(cmd, "--class_names", args.class_names)
    maybe_add(cmd, "--score_thresh", args.score_thresh)
    maybe_add(cmd, "--match_iou", args.match_iou)
    maybe_add(cmd, "--center_gate", args.center_gate)
    maybe_add(cmd, "--rescue_center_gate", args.rescue_center_gate)
    maybe_add(cmd, "--fallback_center_gate", args.fallback_center_gate)
    maybe_add(cmd, "--max_age", args.max_age)
    maybe_add(cmd, "--min_hits", args.min_hits)
    maybe_add(cmd, "--motion_model", args.motion_model)
    maybe_add(cmd, "--motion_horizon", args.motion_horizon)
    maybe_add(cmd, "--velocity_momentum", args.velocity_momentum)
    maybe_add(cmd, "--accel_gain", args.accel_gain)
    maybe_add(cmd, "--max_speed", args.max_speed)
    maybe_add(cmd, "--max_distance", args.max_distance)
    maybe_add(cmd, "--bev_range", args.bev_range)
    maybe_add(cmd, "--rescue_min_visual_conf", args.rescue_min_visual_conf)
    maybe_add(cmd, "--disable_stage2", args.disable_stage2)
    maybe_add(cmd, "--disable_stage3", args.disable_stage3)

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    metric_path = run_dir / "udca_policy_metrics.json"
    with open(metric_path, "r") as f:
        return json.load(f)


def save_csv(summary_rows, csv_path):
    fieldnames = [
        "u3d_high",
        "u2d_high",
        "mota",
        "motp",
        "idf1",
        "precision",
        "recall",
        "fp",
        "fn",
        "id_switches",
        "hota",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot_results(args, u3d_values, u2d_values, grid, out_dir):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm

    x_vals = np.asarray(u2d_values, dtype=np.float32)
    y_vals = np.asarray(u3d_values, dtype=np.float32)
    z_vals = np.asarray(grid, dtype=np.float32)

    metric_name = metric_display_name(args.metric)

    heatmap_path = out_dir / f"{args.metric}_heatmap.png"
    plt.figure(figsize=(7.5, 6.0))
    im = plt.imshow(z_vals, origin="lower", aspect="auto", cmap="viridis")
    plt.colorbar(im, label=metric_name)
    plt.xticks(range(len(x_vals)), [f"{v:.2f}" for v in x_vals])
    plt.yticks(range(len(y_vals)), [f"{v:.2f}" for v in y_vals])
    plt.xlabel(r"$u_{2D}$ threshold")
    plt.ylabel(r"$u_{3D}$ threshold")
    plt.title(f"{metric_name} Heatmap")
    for i in range(z_vals.shape[0]):
        for j in range(z_vals.shape[1]):
            plt.text(j, i, f"{z_vals[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=220)
    plt.close()

    surface_path = out_dir / f"{args.metric}_surface.png"
    fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    xx, yy = np.meshgrid(x_vals, y_vals)
    surf = ax.plot_surface(xx, yy, z_vals, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.95)
    ax.set_xlabel(r"$u_{2D}$ threshold")
    ax.set_ylabel(r"$u_{3D}$ threshold")
    ax.set_zlabel(metric_name)
    ax.set_title(f"{metric_name} Surface")
    fig.colorbar(surf, shrink=0.7, aspect=18, pad=0.08, label=metric_name)
    plt.tight_layout()
    plt.savefig(surface_path, dpi=220)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Grid search u3d/u2d thresholds for eval_udca_policy.py")
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--gt_pkl", type=str, required=True)
    parser.add_argument("--save_root", type=str, required=True)
    parser.add_argument("--data_cfg", type=str, default=None)
    parser.add_argument("--dataset_preset", type=str, default="default")
    parser.add_argument("--class_names", nargs="+", default=["Car", "Pedestrian", "Cyclist"])
    parser.add_argument("--score_thresh", type=float, default=0.1)
    parser.add_argument("--match_iou", type=float, default=0.1)
    parser.add_argument("--center_gate", type=float, default=8.0)
    parser.add_argument("--rescue_center_gate", type=float, default=12.0)
    parser.add_argument("--fallback_center_gate", type=float, default=8.0)
    parser.add_argument("--max_age", type=int, default=2)
    parser.add_argument("--min_hits", type=int, default=2)
    parser.add_argument("--motion_model", type=str, default="constant_velocity")
    parser.add_argument("--motion_horizon", type=float, default=1.0)
    parser.add_argument("--velocity_momentum", type=float, default=0.0)
    parser.add_argument("--accel_gain", type=float, default=0.0)
    parser.add_argument("--max_speed", type=float, default=100.0)
    parser.add_argument("--max_distance", type=float, default=100.0)
    parser.add_argument("--bev_range", type=str, default=None)
    parser.add_argument("--rescue_min_visual_conf", type=float, default=0.35)
    parser.add_argument("--disable_stage2", action="store_true")
    parser.add_argument("--disable_stage3", action="store_true")
    parser.add_argument("--u3d_values", type=str, default="0.15:0.05:0.75")
    parser.add_argument("--u2d_values", type=str, default="0.15:0.05:0.75")
    parser.add_argument("--metric", type=str, default="mota")
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    u3d_values = parse_float_list(args.u3d_values)
    u2d_values = parse_float_list(args.u2d_values)
    print(
        f"u3d grid: {u3d_values}\n"
        f"u2d grid: {u2d_values}\n"
        f"total runs: {len(u3d_values) * len(u2d_values)}"
    )
    save_root = Path(args.save_root)
    runs_dir = save_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    grid = [[math.nan for _ in u2d_values] for _ in u3d_values]

    for i, u3d_value in enumerate(u3d_values):
        for j, u2d_value in enumerate(u2d_values):
            run_name = f"u3d_{u3d_value:.2f}_u2d_{u2d_value:.2f}"
            run_dir = runs_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            metric_path = run_dir / "udca_policy_metrics.json"

            if args.skip_existing and metric_path.exists():
                with open(metric_path, "r") as f:
                    metric_dict = json.load(f)
            else:
                metric_dict = run_single_eval(args, u3d_value, u2d_value, run_dir)

            row = {
                "u3d_high": u3d_value,
                "u2d_high": u2d_value,
                "mota": metric_dict.get("mota", math.nan),
                "motp": metric_dict.get("motp", math.nan),
                "idf1": metric_dict.get("idf1", math.nan),
                "precision": metric_dict.get("precision", math.nan),
                "recall": metric_dict.get("recall", math.nan),
                "fp": metric_dict.get("fp", math.nan),
                "fn": metric_dict.get("fn", math.nan),
                "id_switches": metric_dict.get("id_switches", math.nan),
                "hota": metric_dict.get("hota", math.nan),
            }
            summary_rows.append(row)
            grid[i][j] = float(metric_dict.get(args.metric, math.nan))

    csv_path = save_root / "grid_summary.csv"
    save_csv(summary_rows, csv_path)

    try:
        plot_results(args, u3d_values, u2d_values, grid, save_root)
    except Exception as exc:
        print(f"[Warning] Plot generation failed: {exc}")

    best_row = None
    for row in summary_rows:
        metric_value = row.get(args.metric, math.nan)
        if math.isnan(metric_value):
            continue
        if best_row is None or metric_value > best_row[args.metric]:
            best_row = row

    if best_row is not None:
        with open(save_root / "best_result.json", "w") as f:
            json.dump(best_row, f, indent=2)
        print(
            f"Best {args.metric}: {best_row[args.metric]:.6f} "
            f"at u3d={best_row['u3d_high']:.2f}, u2d={best_row['u2d_high']:.2f}"
        )
    print(f"Saved summary to: {csv_path}")


if __name__ == "__main__":
    main()
