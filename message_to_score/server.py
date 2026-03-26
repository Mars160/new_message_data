import json
import re
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, create_model

EVALUATOR_MODEL = "gpt-4o"
TARGET_MODEL = "Kimi-k2.5"

HERE = Path(__file__).parent
YAML_DIR = HERE.parent / "generate_message" / "scene_output"


class ToolCall(BaseModel):
    id: str
    type: str
    function: dict[str, Any]


class Message(BaseModel):
    content: str | None = None
    role: str
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


class RequestData(BaseModel):
    scene: str | int
    messages: list[Message]


class Metric(BaseModel):
    name: str
    rubric: str


class MetricScore(BaseModel):
    score: int = Field(description="1-5分的评分")
    reason: str = Field(description="评分的简要理由")


class EvaluationResponse(BaseModel):
    scene_id: str
    scene_name: str
    evaluator_model: str
    target_model: str
    conversation: str
    metrics_results: list[dict[str, Any]]
    average_scores: dict[str, float]
    overall_average: float
    raw_response: dict[str, Any]


app = FastAPI(title="Message Evaluation API")

with open(HERE / "secret.json", "r", encoding="utf-8") as f:
    secret = json.load(f)
    if "*" in secret["base_url"]:
        raise ValueError("请将secret.json中的base_url替换为实际的URL")

openai_client = AsyncOpenAI(
    base_url=secret["base_url"], api_key=secret["api_key"], timeout=30
)

with open(HERE / "agent_list.json", "r", encoding="utf-8") as f:
    scene_id_to_name = json.load(f)

scene_name_to_id = {name: scene_id for scene_id, name in scene_id_to_name.items()}


def sanitize_field_name(name: str) -> str:
    sanitized = re.sub(r"[^\w]", "_", name)
    if sanitized and sanitized[0].isdigit():
        sanitized = f"metric_{sanitized}"
    return sanitized


def resolve_scene(scene: str | int) -> tuple[str, str]:
    scene_value = str(scene)
    if scene_value in scene_id_to_name:
        return scene_value, scene_id_to_name[scene_value]
    if scene_value in scene_name_to_id:
        return scene_name_to_id[scene_value], scene_value
    raise HTTPException(status_code=404, detail=f"未找到场景: {scene}")


def get_metrics(scene_id: str) -> list[Metric]:
    yaml_file = YAML_DIR / f"scene{int(scene_id):03d}.yaml"
    if not yaml_file.exists():
        raise HTTPException(status_code=404, detail=f"未找到场景配置: {yaml_file.name}")

    with open(yaml_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    formats = data["evaluation"]["format"]
    metrics = []
    for format_ in formats:
        name = format_["field"]
        if name.endswith("_理由"):
            continue
        metrics.append(Metric(name=name, rubric=format_["description"]))
    return metrics


def build_conversation(messages: list[Message]) -> tuple[str, str]:
    conversation_lines = []
    instruction = ""
    for message in messages:
        if message.role == "user":
            speaker = "Student"
        elif message.role == "assistant":
            speaker = "Teacher"
        elif message.role == "tool":
            speaker = "Tool"
        elif message.role == "system":
            instruction = message.content
            continue
        else:
            raise HTTPException(
                status_code=400,
                detail=f"消息角色不合法: {message.role}，必须是 'user', 'assistant' 或 'tool'",
            )

        if message.content:
            content = message.content
        else:
            # Tool Call
            content = ""
            for tool_call in message.tool_calls or []:
                content += f"Tool Call - Name: {tool_call.function['name']}, Arguments: {json.dumps(tool_call.function['arguments'], ensure_ascii=False)}\n"

        conversation_lines.append(f"{speaker}: {content}")

    if not conversation_lines:
        raise HTTPException(
            status_code=400,
            detail="messages 中至少需要包含一条 role 为 user 或 assistant 的有效消息",
        )
    return "\n\n".join(conversation_lines), instruction


def build_response_model(metrics: list[Metric]) -> type[BaseModel]:
    fields = {}
    for metric in metrics:
        field_name = sanitize_field_name(metric.name)
        fields[field_name] = (
            MetricScore,
            Field(description=f"{metric.name}: {metric.rubric}"),
        )
    return create_model(
        "EvaluationResult",
        **fields,
        __doc__="多个指标的评估结果",
    )


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: RequestData) -> EvaluationResponse:
    scene_id, scene_name = resolve_scene(request.scene)
    metrics = get_metrics(scene_id)
    conversation, instruction = build_conversation(request.messages)
    response_model = build_response_model(metrics)

    metrics_description = "\n\n".join(
        [
            f"指标 {index + 1}: {metric.name}\n评分标准: {metric.rubric}"
            for index, metric in enumerate(metrics)
        ]
    )

    prompt = f"""请根据以下多个评分标准对给定的教学对话进行综合评估。

{metrics_description}

请为每个指标给出1-5分的评分，并提供简要的评分理由。

其中教师人物画像为：
{instruction}

待评估的对话为：
{conversation}
"""

    try:
        response = await openai_client.beta.chat.completions.parse(
            model=EVALUATOR_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的教学评估专家，负责根据评分标准评估对话质量。",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format=response_model,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"调用评测模型失败: {exc}") from exc

    parsed_result = response.choices[0].message.parsed
    if parsed_result is None:
        raise HTTPException(status_code=500, detail="评测模型未返回可解析结果")

    metrics_results = []
    metric_scores = {}
    for metric in metrics:
        field_name = sanitize_field_name(metric.name)
        metric_result = getattr(parsed_result, field_name)
        metrics_results.append(
            {
                "metric_name": metric.name,
                "score": metric_result.score,
                "reason": metric_result.reason,
            }
        )
        metric_scores[metric.name] = float(metric_result.score)

    overall_average = (
        round(sum(metric_scores.values()) / len(metric_scores), 2)
        if metric_scores
        else 0.0
    )

    raw_response = {
        "model": response.model,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
            "completion_tokens": response.usage.completion_tokens
            if response.usage
            else None,
            "total_tokens": response.usage.total_tokens if response.usage else None,
        },
        "parsed_result": parsed_result.model_dump(),
    }

    return EvaluationResponse(
        scene_id=scene_id,
        scene_name=scene_name,
        evaluator_model=EVALUATOR_MODEL,
        target_model=TARGET_MODEL,
        conversation=conversation,
        metrics_results=metrics_results,
        average_scores=metric_scores,
        overall_average=overall_average,
        raw_response=raw_response,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
