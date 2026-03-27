import argparse
import asyncio
import json
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, create_model
from tqdm.asyncio import tqdm

EVALUATOR_MODEL = "gpt-5.2-2025-12-11"
TARGET_MODEL = "Kimi-K25"

# 默认不筛选场景；未传 --test-scenes 时会评测四个场景
TEST_ALL = False
# 仅当 TEST_ALL=False 且传入 --test-scenes 时生效
TEST_SCENES: List[str] = []

HERE = Path(__file__).parent
SCENES_DIR = HERE.parent / "四个场景"
OUTPUT_DIR = HERE.parent / "eval_result"
OUTPUT_DIR.mkdir(exist_ok=True)

with open(HERE / "secret.json", "r", encoding="utf-8") as f:
    secret = json.load(f)
    if "*" in secret["base_url"]:
        raise ValueError("请将secret.json中的base_url替换为实际的URL")

openai_client = AsyncOpenAI(
    base_url=secret["base_url"], api_key=secret["api_key"], timeout=30
)

TASK_FIELD_PATTERN = re.compile(r"\{task\.([^}]+)\}")


class Metric(BaseModel):
    name: str
    rubric: str
    score_type: str = "float"


@dataclass
class Case:
    scene_name: str
    target_model: str
    inputs: str
    file_path: Path
    task_index: Optional[int]
    task_data: Dict[str, Any]
    run_data: Dict[str, Any]


@dataclass
class EvaluationRecord:
    scene_name: str
    target_model: str
    file_path: str
    conversation: str
    task_index: Optional[int]
    task_data: Dict[str, Any]
    run_info: Dict[str, Any]
    metrics_results: List[Dict[str, Any]]
    raw_response: Dict[str, Any]


def normalize_target_model(target_model: str) -> str:
    return target_model.removeprefix("qz__")


def target_model_storage_key(target_model: str) -> str:
    return f"qz__{normalize_target_model(target_model)}"


def to_relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(HERE.parent))
    except ValueError:
        return str(path)


def discover_scenes() -> List[str]:
    if not SCENES_DIR.exists():
        return []
    return sorted(
        [
            scene_dir.name
            for scene_dir in SCENES_DIR.iterdir()
            if scene_dir.is_dir() and (scene_dir / "config.yaml").exists()
        ]
    )


def discover_target_models(scene_name: str) -> List[str]:
    scene_dir = SCENES_DIR / scene_name
    target_models = {
        normalize_target_model(conversation_file.parent.name)
        for conversation_file in scene_dir.glob("*/qz__*/conversation-messages.json")
    }
    return sorted(target_models)


def load_scene_config(scene_name: str) -> Dict[str, Any]:
    config_path = SCENES_DIR / scene_name / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_task_catalog(scene_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    tasks_config = scene_config.get("tasks", {})
    mode = tasks_config.get("mode", "iter")
    content = tasks_config.get("content", [])

    if mode == "iter":
        if not isinstance(content, list):
            return []
        return [item if isinstance(item, dict) else {"value": item} for item in content]

    if mode == "union":
        if not isinstance(content, dict):
            return []
        keys = list(content.keys())
        value_lists = []
        for key in keys:
            value = content.get(key, [])
            if isinstance(value, list):
                value_lists.append(value)
            else:
                value_lists.append([value])
        return [dict(zip(keys, combo)) for combo in product(*value_lists)]

    return []


def get_task_index(run_data: Dict[str, Any]) -> Optional[int]:
    value = run_data.get("metadata", {}).get("sceneQuestionIndex")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def format_messages_as_dialog(messages: List[Dict[str, Any]]) -> str:
    dialog_lines = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or content is None:
            continue
        speaker = "Student" if role == "user" else "Teacher"
        dialog_lines.append(f"{speaker}: {content}")
    return "\n".join(dialog_lines).strip()


def build_run_info(run_data: Dict[str, Any]) -> Dict[str, Any]:
    metadata = run_data.get("metadata", {})
    return {
        "run_id": run_data.get("runId"),
        "title": run_data.get("title"),
        "scene_question_index": metadata.get("sceneQuestionIndex"),
        "scene_question_total": metadata.get("sceneQuestionTotal"),
        "student_initial_question": run_data.get("studentInitialQuestion"),
        "follow_up_count": run_data.get("followUpCount"),
        "stop_reason": run_data.get("stopReason"),
        "status": run_data.get("status"),
    }


def get_metrics(scene_config: Dict[str, Any]) -> List[Metric]:
    formats = scene_config.get("evaluation", {}).get("format", [])
    metrics = []
    for format_item in formats:
        field_name = format_item.get("field")
        if not field_name:
            continue
        metrics.append(
            Metric(
                name=str(field_name),
                rubric=str(format_item.get("description", "")),
                score_type=str(format_item.get("type", "float")).lower(),
            )
        )
    return metrics


def render_prompt_template(template: str, case: Case) -> str:
    rendered = template.replace("{messages.as_dialog()}", case.inputs)

    def replace_task_field(match: re.Match[str]) -> str:
        key = match.group(1)
        value = case.task_data.get(key, "")
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    return TASK_FIELD_PATTERN.sub(replace_task_field, rendered)


def build_default_prompt(case: Case, metrics: List[Metric]) -> List[Dict[str, str]]:
    metrics_description = "\n\n".join(
        [
            f"指标 {index + 1}: {metric.name}\n评分标准: {metric.rubric}"
            for index, metric in enumerate(metrics)
        ]
    )
    return [
        {
            "role": "system",
            "content": "你是一个专业的教学评估专家，负责根据评分标准评估教学对话质量。",
        },
        {
            "role": "user",
            "content": (
                "请根据以下多个评分标准对给定的教学对话进行综合评估。\n\n"
                f"{metrics_description}\n\n"
                f"请为每个指标给出 0~5 分范围内的评分，并提供简要的评分理由。\n\n"
                f"待评估的对话：\n{case.inputs}"
            ),
        },
    ]


def get_scene_cases(
    scene_name: str, target_model: str, scene_config: Dict[str, Any]
) -> List[Case]:
    scene_dir = SCENES_DIR / scene_name
    storage_key = target_model_storage_key(target_model)
    task_catalog = build_task_catalog(scene_config)
    conversation_files = sorted(
        scene_dir.glob(f"*/{storage_key}/conversation-messages.json")
    )

    cases = []
    for conversation_file in conversation_files:
        with open(conversation_file, "r", encoding="utf-8") as f:
            conversation_data = json.load(f)

        run_file = conversation_file.with_name("run.json")
        run_data: Dict[str, Any] = {}
        if run_file.exists():
            with open(run_file, "r", encoding="utf-8") as f:
                run_data = json.load(f)

        task_index = get_task_index(run_data)
        task_data: Dict[str, Any] = {}
        if task_index is not None and 1 <= task_index <= len(task_catalog):
            task_data = task_catalog[task_index - 1]

        cases.append(
            Case(
                scene_name=scene_name,
                target_model=normalize_target_model(target_model),
                inputs=format_messages_as_dialog(conversation_data.get("messages", [])),
                file_path=conversation_file,
                task_index=task_index,
                task_data=task_data,
                run_data=run_data,
            )
        )

    return cases


@dataclass
class Evaluator:
    metrics: List[Metric]
    prompt_templates: List[Dict[str, Any]]

    def __post_init__(self):
        fields = {}
        for metric in self.metrics:
            field_name = self._sanitize_field_name(metric.name)
            score_python_type = int if metric.score_type == "int" else float
            metric_score_model = create_model(
                f"{field_name}_score_model",
                score=(
                    score_python_type,
                    Field(description=(f"{metric.name} 的评分，使用 0-5 范围内的数值")),
                ),
                reason=(str, Field(description=f"{metric.name} 的简要评分理由")),
                __doc__=f"{metric.name} 的评分结果",
            )
            fields[field_name] = (
                metric_score_model,
                Field(description=f"{metric.name}: {metric.rubric}"),
            )

        self._response_model = create_model(
            "EvaluationResult",
            **fields,
            __doc__="多个指标的评估结果",
        )

    def _sanitize_field_name(self, name: str) -> str:
        sanitized = re.sub(r"[^\w]", "_", name)
        if sanitized and sanitized[0].isdigit():
            sanitized = f"metric_{sanitized}"
        return sanitized or "metric"

    def _build_messages(self, case: Case) -> List[Dict[str, str]]:
        if not self.prompt_templates:
            return build_default_prompt(case, self.metrics)

        metric_names = "、".join(metric.name for metric in self.metrics)
        messages = [
            {
                "role": "system",
                "content": (
                    f"请严格评估以下指标：{metric_names}。"
                    f"每个指标都必须输出 score 和 reason，score 使用 0-5 范围内的数值。"
                ),
            }
        ]

        for template in self.prompt_templates:
            messages.append(
                {
                    "role": str(template.get("role", "user")),
                    "content": render_prompt_template(
                        str(template.get("content", "")), case
                    ),
                }
            )

        return messages

    async def evaluate(self, case: Case) -> EvaluationRecord:
        response = await openai_client.beta.chat.completions.parse(
            model=EVALUATOR_MODEL,
            messages=self._build_messages(case),  # pyright: ignore[reportArgumentType]
            temperature=0,
            response_format=self._response_model,
            timeout=9999999999999,
        )

        result = response.choices[0].message.parsed
        if result is None:
            raise ValueError(f"结构化评测失败: {case.file_path}")

        raw_response = {
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens
                if response.usage
                else None,
                "completion_tokens": response.usage.completion_tokens
                if response.usage
                else None,
                "total_tokens": response.usage.total_tokens if response.usage else None,
            },
            "parsed_result": result.model_dump(),
        }

        metrics_results = []
        for metric in self.metrics:
            field_name = self._sanitize_field_name(metric.name)
            metric_result = getattr(result, field_name)
            metrics_results.append(
                {
                    "metric_name": metric.name,
                    "score": metric_result.score,
                    "reason": metric_result.reason,
                }
            )

        return EvaluationRecord(
            scene_name=case.scene_name,
            target_model=case.target_model,
            file_path=to_relative_path(case.file_path),
            conversation=case.inputs,
            task_index=case.task_index,
            task_data=case.task_data,
            run_info=build_run_info(case.run_data),
            metrics_results=metrics_results,
            raw_response=raw_response,
        )


class EvaluatorSet:
    def __init__(self, evaluator: Evaluator, cases: List[Case], concurrency: int):
        self.evaluator = evaluator
        self.cases = cases
        self.concurrency = max(concurrency, 1)

    async def evaluate_all(self) -> List[EvaluationRecord]:
        if not self.cases:
            return []

        semaphore = asyncio.Semaphore(self.concurrency)

        async def evaluate_case(case: Case) -> EvaluationRecord:
            async with semaphore:
                return await self.evaluator.evaluate(case)

        tasks = [evaluate_case(case) for case in self.cases]
        return await tqdm.gather(*tasks)


def save_results(
    records: List[EvaluationRecord],
    scene_name: str,
    target_model: str,
):
    scene_output_dir = OUTPUT_DIR / scene_name
    scene_output_dir.mkdir(parents=True, exist_ok=True)
    output_file = scene_output_dir / f"{normalize_target_model(target_model)}.json"

    metric_scores: Dict[str, List[float]] = {}
    for record in records:
        for metric_result in record.metrics_results:
            metric_name = metric_result["metric_name"]
            metric_scores.setdefault(metric_name, []).append(metric_result["score"])

    average_scores_raw = {
        name: sum(scores) / len(scores)
        for name, scores in metric_scores.items()
        if scores
    }
    overall_average = (
        sum(average_scores_raw.values()) / len(average_scores_raw)
        if average_scores_raw
        else 0
    )

    output_data = {
        "overall": {
            "scene_name": scene_name,
            "evaluator_model": EVALUATOR_MODEL,
            "target_model": normalize_target_model(target_model),
            "score_range": {"min": 0, "max": 5},
            "total_cases": len(records),
            "average_scores": {
                name: round(score, 2) for name, score in average_scores_raw.items()
            },
            "overall_average": round(overall_average, 2),
        },
        "items": [
            {
                "file_path": record.file_path,
                "task_index": record.task_index,
                "task_data": record.task_data,
                "run_info": record.run_info,
                "conversation": record.conversation,
                "metrics_results": record.metrics_results,
                "raw_response": record.raw_response,
            }
            for record in records
        ],
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {to_relative_path(output_file)}")
    print(f"Total cases evaluated: {len(records)}")
    print(f"Overall average score: {overall_average:.2f}")


async def evaluate_scene(scene_name: str, target_model: str):
    scene_config = load_scene_config(scene_name)
    metrics = get_metrics(scene_config)
    prompt_templates = scene_config.get("evaluation", {}).get("prompt", [])
    concurrency = int(scene_config.get("globals", {}).get("concurrency", 8))

    evaluator = Evaluator(
        metrics=metrics,
        prompt_templates=prompt_templates,
    )
    cases = get_scene_cases(scene_name, target_model, scene_config)

    if not cases:
        print(
            f"\nSkipping scene '{scene_name}': 未找到目标模型 {normalize_target_model(target_model)} 的数据。"
        )
        return

    print(f"\nEvaluating scene: {scene_name}")
    print(f"Target model: {normalize_target_model(target_model)}")
    print(f"Metrics: {[metric.name for metric in metrics]}")
    print(f"Total cases: {len(cases)}")

    evaluator_set = EvaluatorSet(
        evaluator=evaluator,
        cases=cases,
        concurrency=concurrency,
    )
    records = await evaluator_set.evaluate_all()
    save_results(records, scene_name, target_model)


async def main(test_all: bool, test_scenes: List[str], target_models: List[str]):
    available_scenes = discover_scenes()
    if test_all or not test_scenes:
        scenes_to_test = available_scenes
    else:
        invalid_scenes = [
            scene for scene in test_scenes if scene not in available_scenes
        ]
        for invalid_scene in invalid_scenes:
            print(f"Skipping unknown scene: {invalid_scene}")
        scenes_to_test = [scene for scene in test_scenes if scene in available_scenes]

    print(f"Testing scenes: {scenes_to_test}")

    for scene_name in scenes_to_test:
        scene_target_models = target_models or discover_target_models(scene_name)
        for target_model in scene_target_models:
            await evaluate_scene(scene_name, target_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the four-scene datasets with GPT-5.4."
    )
    parser.add_argument(
        "-ta", "--test-all", action="store_true", help="Evaluate all available scenes"
    )
    parser.add_argument(
        "-ts",
        "--test-scenes",
        nargs="+",
        default=TEST_SCENES,
        help="List of scenes to evaluate (ignored if --test-all is set)",
    )
    parser.add_argument(
        "-em",
        "--evaluator-model",
        type=str,
        default=EVALUATOR_MODEL,
        help="Model to use for evaluation",
    )
    parser.add_argument(
        "-tm",
        "--target-model",
        action="append",
        dest="target_models",
        help="Target model to evaluate; can be passed multiple times",
    )
    parser.add_argument(
        "--all-target-models",
        action="store_true",
        help="Evaluate all target models found under each scene",
    )
    args = parser.parse_args()

    EVALUATOR_MODEL = args.evaluator_model
    TEST_ALL = args.test_all or TEST_ALL
    TEST_SCENES = args.test_scenes

    if args.all_target_models:
        selected_target_models: List[str] = []
    elif args.target_models:
        selected_target_models = [
            normalize_target_model(model) for model in args.target_models
        ]
    else:
        selected_target_models = [TARGET_MODEL]

    asyncio.run(main(TEST_ALL, TEST_SCENES, selected_target_models))
