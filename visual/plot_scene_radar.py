#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "eval_result"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "visual" / "output" / "radar"

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

MODEL_COLORS = {
    "glm-5": "#4E79A7",
    "Kimi-K25": "#F28E2B",
    "qwen3.5-397b": "#59A14F",
}

FALLBACK_COLORS = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#76B7B2"]


@dataclass(frozen=True)
class ModelMetrics:
    model: str
    overall_average: float
    metrics: dict[str, float]


@dataclass(frozen=True)
class SceneData:
    scene: str
    score_max: float
    metric_order: list[str]
    models: list[ModelMetrics]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one radar chart per scene from eval_result JSON files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory that stores eval json files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory used to save radar charts (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
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
            "figure.facecolor": "#f8f5ef",
            "savefig.facecolor": "#f8f5ef",
            "axes.facecolor": "#fffdf9",
            "grid.color": "#d8cfbf",
            "grid.alpha": 0.45,
            "axes.edgecolor": "#8a7b67",
            "text.color": "#34281f",
            "axes.labelcolor": "#34281f",
            "xtick.color": "#493b2e",
            "ytick.color": "#493b2e",
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


def ordered_union(items: list[list[str]]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for item_list in items:
        for item in item_list:
            if item not in seen:
                seen.add(item)
                merged.append(item)
    return merged


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


def infer_scene_score_max(payloads: list[dict]) -> float:
    metadata_max = 0.0
    observed_max = 0.0

    for payload in payloads:
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


def load_scene_data(input_dir: Path) -> list[SceneData]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    payloads_by_scene: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for json_path in sorted(input_dir.glob("*/*.json")):
        with json_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        payloads_by_scene[json_path.parent.name].append((json_path.stem, payload))

    if not payloads_by_scene:
        raise FileNotFoundError(f"No json files found under: {input_dir}")

    scene_data_list: list[SceneData] = []
    for scene in sorted(payloads_by_scene):
        model_payloads = sorted(
            payloads_by_scene[scene], key=lambda item: item[0].lower()
        )
        score_max = infer_scene_score_max(
            [payload for _model, payload in model_payloads]
        )
        metric_order = ordered_union(
            [
                list(payload.get("overall", {}).get("average_scores", {}).keys())
                for _model, payload in model_payloads
            ]
        )

        model_metrics: list[ModelMetrics] = []
        for model, payload in model_payloads:
            overall = payload.get("overall", {})
            metrics = {
                str(metric): safe_float(score)
                for metric, score in overall.get("average_scores", {}).items()
            }
            model_metrics.append(
                ModelMetrics(
                    model=model,
                    overall_average=safe_float(overall.get("overall_average")),
                    metrics=metrics,
                )
            )

        scene_data_list.append(
            SceneData(
                scene=scene,
                score_max=score_max,
                metric_order=metric_order,
                models=model_metrics,
            )
        )

    return scene_data_list


def get_model_color(model: str, index: int) -> str:
    return MODEL_COLORS.get(model, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])


def build_tick_values(score_max: float) -> list[float]:
    if score_max <= 5:
        return [1, 2, 3, 4, 5][: max(1, int(round(score_max)))]
    return np.linspace(score_max / 5, score_max, 5).tolist()


def plot_scene_radar(scene_data: SceneData, output_path: Path, dpi: int) -> None:
    if not scene_data.metric_order:
        raise ValueError(
            f"Scene {scene_data.scene} does not contain any average_scores metrics."
        )

    metric_labels = [wrap_label(metric, 10) for metric in scene_data.metric_order]
    angles = np.linspace(0, 2 * np.pi, len(metric_labels), endpoint=False).tolist()
    closed_angles = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(10.5, 8.5), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.tick_params(axis="x", pad=14)

    tick_values = build_tick_values(scene_data.score_max)
    ax.set_ylim(0, scene_data.score_max)
    ax.set_yticks(tick_values)
    ax.set_yticklabels([format_score(value, 1) for value in tick_values], fontsize=9)
    ax.set_rlabel_position(12)
    ax.spines["polar"].set_color("#a4937e")
    ax.spines["polar"].set_linewidth(1.1)

    for index, model_data in enumerate(scene_data.models):
        values = []
        for metric in scene_data.metric_order:
            value = model_data.metrics.get(metric, math.nan)
            values.append(0.0 if not math.isfinite(value) else value)

        closed_values = values + values[:1]
        color = get_model_color(model_data.model, index)
        label = (
            f"{model_data.model} (综合均分 {format_score(model_data.overall_average)})"
        )

        ax.plot(closed_angles, closed_values, color=color, linewidth=2.3, label=label)
        ax.fill(closed_angles, closed_values, color=color, alpha=0.14)
        ax.scatter(angles, values, color=color, s=28, zorder=3)

    ax.set_title(
        f"{scene_data.scene}\n3 个模型的指标均分雷达图（满分 {format_score(scene_data.score_max, 0)}）",
        fontsize=15,
        pad=28,
    )

    legend = ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.32, 1.16),
        frameon=True,
        fontsize=10,
    )
    legend.get_frame().set_facecolor("#fffdf9")
    legend.get_frame().set_edgecolor("#d1c4b3")

    fig.text(
        0.08,
        0.05,
        "说明：每条轴代表该场景中的一个评分指标，图例中的数值为 overall_average。",
        fontsize=9,
        color="#5d4d3d",
    )

    fig.tight_layout(rect=(0.02, 0.08, 0.93, 0.96))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_manifest(scene_files: list[tuple[str, Path]], output_dir: Path) -> Path:
    manifest_path = output_dir / "radar_generated_files.txt"
    with manifest_path.open("w", encoding="utf-8") as file:
        for scene, path in scene_files:
            file.write(f"{scene}: {path.name}\n")
    return manifest_path


def main() -> None:
    args = parse_args()
    selected_font = configure_matplotlib()
    scene_data_list = load_scene_data(args.input_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    scene_files: list[tuple[str, Path]] = []
    for index, scene_data in enumerate(scene_data_list, start=1):
        output_path = args.output_dir / f"scene_{index:02d}_radar.png"
        plot_scene_radar(scene_data, output_path, args.dpi)
        scene_files.append((scene_data.scene, output_path))

    manifest_path = write_manifest(scene_files, args.output_dir)

    print(f"Loaded {len(scene_data_list)} scenes from: {args.input_dir}")
    if selected_font:
        print(f"Using CJK font: {selected_font}")
    else:
        print("No preferred CJK font detected. Chinese labels may render incorrectly.")

    print(f"Saved radar charts to: {args.output_dir}")
    for scene, path in scene_files:
        print(f"- {scene}: {path.name}")
    print(f"- Manifest: {manifest_path.name}")


if __name__ == "__main__":
    main()
