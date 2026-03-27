#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "eval_result"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "visual" / "output"

PREFERRED_CJK_FONTS = [
    "PingFang SC",
    "Hiragino Sans GB",
    "Source Han Sans SC",
    "Noto Sans CJK SC",
    "WenQuanYi Zen Hei",
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
]

MODEL_COLORS = [
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#76B7B2",
    "#EDC948",
]


@dataclass(frozen=True)
class EvalRecord:
    scene: str
    model: str
    file_path: Path
    total_cases: int
    metadata_score_max: float
    score_max: float
    overall_average: float
    normalized_overall: float
    average_scores: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize model evaluation scores under eval_result with matplotlib."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory that stores eval json files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory used to save generated figures (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output image DPI.",
    )
    return parser.parse_args()


def configure_matplotlib() -> str | None:
    plt.style.use("seaborn-v0_8-whitegrid")

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected_font = None
    for font_name in PREFERRED_CJK_FONTS:
        if font_name in available_fonts:
            selected_font = font_name
            break

    if selected_font:
        plt.rcParams["font.sans-serif"] = [selected_font, "DejaVu Sans"]

    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.facecolor": "#fffaf5",
            "axes.facecolor": "#fffdf8",
            "savefig.facecolor": "#fffaf5",
            "axes.edgecolor": "#7c6a59",
            "axes.labelcolor": "#2f241b",
            "text.color": "#2f241b",
            "xtick.color": "#4c3d30",
            "ytick.color": "#4c3d30",
            "grid.color": "#dbcbbd",
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "axes.titleweight": "bold",
            "axes.titlepad": 12,
        }
    )
    return selected_font


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def format_score(value: float, digits: int = 2) -> str:
    if not math.isfinite(value):
        return "-"
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def wrap_label(text: str, width: int = 8) -> str:
    if len(text) <= width:
        return text
    return "\n".join(
        text[index : index + width] for index in range(0, len(text), width)
    )


def collect_numeric_scores(payload: dict) -> list[float]:
    values: list[float] = []
    overall = payload.get("overall", {})
    values.append(safe_float(overall.get("overall_average")))

    for score in overall.get("average_scores", {}).values():
        values.append(safe_float(score))

    for item in payload.get("items", []):
        for metric_result in item.get("metrics_results", []):
            values.append(safe_float(metric_result.get("score")))

    return [value for value in values if math.isfinite(value)]


def infer_scene_score_max(entries: list[tuple[str, Path, dict]]) -> float:
    metadata_max = 0.0
    observed_max = 0.0

    for _model, _path, payload in entries:
        overall = payload.get("overall", {})
        metadata_max = max(
            metadata_max,
            safe_float(overall.get("score_range", {}).get("max")),
        )

        numeric_scores = collect_numeric_scores(payload)
        if numeric_scores:
            observed_max = max(observed_max, max(numeric_scores))

    if metadata_max > 0 and observed_max <= metadata_max + 1e-9:
        return metadata_max

    for candidate in (5.0, 10.0, 100.0):
        if observed_max <= candidate + 1e-9:
            return candidate

    inferred = max(metadata_max, math.ceil(observed_max))
    return inferred if inferred > 0 else 1.0


def load_records(input_dir: Path) -> list[EvalRecord]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    payloads_by_scene: dict[str, list[tuple[str, Path, dict]]] = defaultdict(list)
    for json_path in sorted(input_dir.glob("*/*.json")):
        with json_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        payloads_by_scene[json_path.parent.name].append(
            (json_path.stem, json_path, payload)
        )

    if not payloads_by_scene:
        raise FileNotFoundError(f"No json files found under: {input_dir}")

    scene_score_max = {
        scene: infer_scene_score_max(entries)
        for scene, entries in payloads_by_scene.items()
    }

    records: list[EvalRecord] = []
    for scene, entries in payloads_by_scene.items():
        inferred_max = scene_score_max[scene]
        for model, json_path, payload in entries:
            overall = payload.get("overall", {})
            metadata_score_max = safe_float(overall.get("score_range", {}).get("max"))
            overall_average = safe_float(overall.get("overall_average"))
            average_scores = {
                str(metric): safe_float(score)
                for metric, score in overall.get("average_scores", {}).items()
            }

            normalized_overall = (
                overall_average / inferred_max * 100
                if inferred_max > 0 and math.isfinite(overall_average)
                else math.nan
            )

            total_cases = overall.get("total_cases", 0)
            records.append(
                EvalRecord(
                    scene=scene,
                    model=model,
                    file_path=json_path,
                    total_cases=int(total_cases) if total_cases else 0,
                    metadata_score_max=metadata_score_max,
                    score_max=inferred_max,
                    overall_average=overall_average,
                    normalized_overall=normalized_overall,
                    average_scores=average_scores,
                )
            )

    return records


def ordered_union(keys_iterable: Iterable[Iterable[str]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for keys in keys_iterable:
        for key in keys:
            if key not in seen:
                seen.add(key)
                ordered.append(key)
    return ordered


def get_scene_order(records: list[EvalRecord]) -> list[str]:
    return sorted({record.scene for record in records})


def get_model_order(records: list[EvalRecord]) -> list[str]:
    scores_by_model: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if math.isfinite(record.normalized_overall):
            scores_by_model[record.model].append(record.normalized_overall)

    ranked_models = sorted(
        scores_by_model.items(),
        key=lambda item: (-sum(item[1]) / len(item[1]), item[0].lower()),
    )
    return [model for model, _scores in ranked_models]


def get_color_map(models: list[str]) -> dict[str, str]:
    return {
        model: MODEL_COLORS[index % len(MODEL_COLORS)]
        for index, model in enumerate(models)
    }


def plot_overall_heatmap(
    records: list[EvalRecord],
    scene_order: list[str],
    model_order: list[str],
    output_dir: Path,
    dpi: int,
) -> Path:
    row_index = {scene: index for index, scene in enumerate(scene_order)}
    col_index = {model: index for index, model in enumerate(model_order)}

    matrix = np.full((len(scene_order), len(model_order)), np.nan)
    annotations: dict[tuple[int, int], str] = {}

    for record in records:
        row = row_index[record.scene]
        col = col_index[record.model]
        matrix[row, col] = record.normalized_overall
        annotations[(row, col)] = (
            f"{format_score(record.normalized_overall, 1)}%\n"
            f"({format_score(record.overall_average)}/{format_score(record.score_max, 0)})"
        )

    masked = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#f2ebe2")

    fig, ax = plt.subplots(
        figsize=(
            max(7.5, len(model_order) * 2.1 + 2.5),
            max(4.8, len(scene_order) * 1.2 + 2.0),
        )
    )
    image = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(
        range(len(model_order)), [wrap_label(model, 12) for model in model_order]
    )
    ax.set_yticks(
        range(len(scene_order)), [wrap_label(scene, 8) for scene in scene_order]
    )
    ax.set_title("模型跨场景综合得分热力图（按场景分值上限归一化）")

    ax.set_xticks(np.arange(-0.5, len(model_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(scene_order), 1), minor=True)
    ax.grid(which="minor", color="#fffaf5", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if not math.isfinite(value):
                label = "-"
                text_color = "#2f241b"
            else:
                label = annotations[(row, col)]
                text_color = "white" if value >= 62 else "#2f241b"
            ax.text(
                col, row, label, ha="center", va="center", fontsize=9, color=text_color
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("综合得分率 (%)")

    output_path = output_dir / "overall_normalized_heatmap.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_overall_by_scene(
    records: list[EvalRecord],
    scene_order: list[str],
    model_order: list[str],
    model_colors: dict[str, str],
    output_dir: Path,
    dpi: int,
) -> Path:
    records_by_scene: dict[str, list[EvalRecord]] = defaultdict(list)
    for record in records:
        records_by_scene[record.scene].append(record)

    columns = 2 if len(scene_order) > 1 else 1
    rows = math.ceil(len(scene_order) / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(columns * 7.5, rows * 4.8), squeeze=False
    )
    axes_list = axes.ravel()

    for axis, scene in zip(axes_list, scene_order):
        scene_records = sorted(
            records_by_scene[scene],
            key=lambda record: model_order.index(record.model),
        )

        labels = [record.model for record in scene_records]
        values = [record.overall_average for record in scene_records]
        colors = [model_colors[record.model] for record in scene_records]
        scene_max = max((record.score_max for record in scene_records), default=1.0)
        x_positions = np.arange(len(scene_records))

        bars = axis.bar(x_positions, values, color=colors, width=0.62)
        axis.set_xticks(x_positions, [wrap_label(label, 12) for label in labels])
        axis.set_ylim(0, scene_max * 1.18)
        axis.set_ylabel(f"综合均分（满分 {format_score(scene_max, 0)}）")
        axis.set_title(f"{scene} | {scene_records[0].total_cases} 条样本")
        axis.grid(axis="x", visible=False)

        for bar, record in zip(bars, scene_records):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + scene_max * 0.03,
                f"{format_score(record.overall_average)}\n{format_score(record.normalized_overall, 1)}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    for axis in axes_list[len(scene_order) :]:
        axis.remove()

    fig.suptitle("各场景下的模型综合均分", fontsize=16, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output_path = output_dir / "overall_by_scene.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_model_leaderboard(
    records: list[EvalRecord],
    model_order: list[str],
    model_colors: dict[str, str],
    output_dir: Path,
    dpi: int,
) -> Path:
    model_scores: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if math.isfinite(record.normalized_overall):
            model_scores[record.model].append(record.normalized_overall)

    leaderboard = sorted(
        (
            (model, sum(scores) / len(scores), len(scores))
            for model, scores in model_scores.items()
            if scores
        ),
        key=lambda item: (-item[1], item[0].lower()),
    )

    labels = [item[0] for item in leaderboard]
    values = [item[1] for item in leaderboard]
    colors = [model_colors[item[0]] for item in leaderboard]

    fig, ax = plt.subplots(figsize=(max(7.0, len(labels) * 2.2 + 1.5), 5.2))
    x_positions = np.arange(len(labels))
    bars = ax.bar(x_positions, values, color=colors, width=0.62)

    upper_bound = max(100.0, (max(values) if values else 0.0) * 1.15)
    ax.set_ylim(0, upper_bound)
    ax.set_xticks(x_positions, [wrap_label(label, 12) for label in labels])
    ax.set_ylabel("平均综合得分率 (%)")
    ax.set_title("模型跨场景平均表现排行榜")
    ax.grid(axis="x", visible=False)

    for bar, score, count in zip(bars, values, [item[2] for item in leaderboard]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + upper_bound * 0.015,
            f"{format_score(score, 1)}%\n{count} 个场景",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    output_path = output_dir / "model_leaderboard.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_scene_metric_heatmap(
    scene: str,
    scene_records: list[EvalRecord],
    model_order: list[str],
    scene_index: int,
    output_dir: Path,
    dpi: int,
) -> Path:
    metrics = ordered_union(record.average_scores.keys() for record in scene_records)
    scene_model_order = [
        model
        for model in model_order
        if any(record.model == model for record in scene_records)
    ]
    row_index = {metric: index for index, metric in enumerate(metrics)}
    col_index = {model: index for index, model in enumerate(scene_model_order)}

    matrix = np.full((len(metrics), len(scene_model_order)), np.nan)
    for record in scene_records:
        for metric, score in record.average_scores.items():
            matrix[row_index[metric], col_index[record.model]] = score

    masked = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap("YlOrBr").copy()
    cmap.set_bad("#f2ebe2")

    scene_max = max((record.score_max for record in scene_records), default=1.0)
    figure_height = max(5.2, len(metrics) * 0.52 + 2.2)
    figure_width = max(7.5, len(scene_model_order) * 1.9 + 3.5)

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    image = ax.imshow(masked, cmap=cmap, vmin=0, vmax=scene_max, aspect="auto")

    ax.set_xticks(
        range(len(scene_model_order)),
        [wrap_label(model, 12) for model in scene_model_order],
    )
    ax.set_yticks(range(len(metrics)), [wrap_label(metric, 10) for metric in metrics])
    ax.set_title(f"{scene}：各维度平均得分（满分 {format_score(scene_max, 0)}）")

    ax.set_xticks(np.arange(-0.5, len(scene_model_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(metrics), 1), minor=True)
    ax.grid(which="minor", color="#fffaf5", linestyle="-", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    threshold = scene_max * 0.62
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            label = format_score(value)
            text_color = (
                "white" if math.isfinite(value) and value >= threshold else "#2f241b"
            )
            ax.text(
                col, row, label, ha="center", va="center", fontsize=9, color=text_color
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("平均得分")

    output_path = output_dir / f"scene_{scene_index:02d}_metric_heatmap.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_manifest(generated_files: list[tuple[str, Path]], output_dir: Path) -> Path:
    manifest_path = output_dir / "generated_files.txt"
    with manifest_path.open("w", encoding="utf-8") as file:
        for title, path in generated_files:
            file.write(f"{title}: {path.name}\n")
    return manifest_path


def main() -> None:
    args = parse_args()
    selected_font = configure_matplotlib()
    records = load_records(args.input_dir)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_order = get_scene_order(records)
    model_order = get_model_order(records)
    model_colors = get_color_map(model_order)

    generated_files: list[tuple[str, Path]] = []
    generated_files.append(
        (
            "跨场景综合热力图",
            plot_overall_heatmap(
                records, scene_order, model_order, output_dir, args.dpi
            ),
        )
    )
    generated_files.append(
        (
            "分场景综合均分柱状图",
            plot_overall_by_scene(
                records,
                scene_order,
                model_order,
                model_colors,
                output_dir,
                args.dpi,
            ),
        )
    )
    generated_files.append(
        (
            "模型排行榜",
            plot_model_leaderboard(
                records, model_order, model_colors, output_dir, args.dpi
            ),
        )
    )

    records_by_scene: dict[str, list[EvalRecord]] = defaultdict(list)
    for record in records:
        records_by_scene[record.scene].append(record)

    for scene_index, scene in enumerate(scene_order, start=1):
        generated_files.append(
            (
                f"{scene} 维度热力图",
                plot_scene_metric_heatmap(
                    scene,
                    records_by_scene[scene],
                    model_order,
                    scene_index,
                    output_dir,
                    args.dpi,
                ),
            )
        )

    manifest_path = write_manifest(generated_files, output_dir)

    print(f"Loaded {len(records)} evaluation files from: {args.input_dir}")
    if selected_font:
        print(f"Using CJK font: {selected_font}")
    else:
        print("No preferred CJK font detected. Chinese labels may render incorrectly.")

    print(f"Saved figures to: {output_dir}")
    for title, path in generated_files:
        print(f"- {title}: {path.name}")
    print(f"- Manifest: {manifest_path.name}")


if __name__ == "__main__":
    main()
