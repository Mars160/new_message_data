# 评测 API 文档

`server.py` 提供一个基于 `eval.py` 同款评测逻辑的 HTTP API，用于对单段教学对话进行在线评分。

## 服务信息

- 默认地址：`http://127.0.0.1:8000`
- 接口路径：`POST /evaluate`
- Content-Type：`application/json`

## 评测逻辑

- 根据传入的 `scene` 定位场景配置
- 从 `../generate_message/scene_output/scene{id}.yaml` 读取 `evaluation.format`
- 自动忽略以 `_理由` 结尾的字段，只保留真正的评分指标
- 将 `messages` 拼接成 `Student:` / `Teacher:` 格式的对话文本
- 调用 OpenAI 结构化输出接口，返回每个指标的 `score` 和 `reason`

## 请求体

```json
{
  "scene": 138,
  "messages": [
    {
      "role": "user",
      "content": "我总是分不清渗透和扩散。"
    },
    {
      "role": "assistant",
      "content": "可以从定义、发生条件和生活例子三个角度来区分渗透和扩散。"
    }
  ]
}
```

### 字段说明

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `scene` | `string \| int` | 是 | 场景 id 或场景名称，例如 `138` 或 `初中生物术语辨析助手` |
| `messages` | `array` | 是 | 待评测消息列表 |
| `messages[].role` | `string` | 是 | 角色，通常使用 `user` 或 `assistant` |
| `messages[].content` | `string \| null` | 是 | 消息内容；`null` 或非 `user/assistant` 的消息会被忽略 |

## 返回体

```json
{
  "scene_id": "138",
  "scene_name": "初中生物术语辨析助手",
  "evaluator_model": "gpt-4o",
  "target_model": "Kimi-k2.5",
  "conversation": "Student: 我总是分不清渗透和扩散。\nTeacher: 可以从定义、发生条件和生活例子三个角度来区分渗透和扩散。",
  "metrics_results": [
    {
      "metric_name": "个性化画像适配度",
      "score": 2,
      "reason": "对话中没有明显体现个性化适配，教师的回答较为通用，没有根据学生的具体画像进行调整。"
    }
  ],
  "average_scores": {
    "个性化画像适配度": 2.0
  },
  "overall_average": 2.88,
  "raw_response": {
    "model": "gpt-4o-2024-08-06",
    "usage": {
      "prompt_tokens": 4355,
      "completion_tokens": 394,
      "total_tokens": 4749
    },
    "parsed_result": {
      "个性化画像适配度": {
        "score": 2,
        "reason": "对话中没有明显体现个性化适配，教师的回答较为通用，没有根据学生的具体画像进行调整。"
      }
    }
  }
}
```

### 返回字段说明

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `scene_id` | `string` | 解析后的场景 id |
| `scene_name` | `string` | 解析后的场景名称 |
| `evaluator_model` | `string` | 当前评测模型 |
| `target_model` | `string` | 被评测目标模型名称 |
| `conversation` | `string` | 按评测格式拼接后的对话内容 |
| `metrics_results` | `array` | 每个评测指标的详细评分结果 |
| `average_scores` | `object` | 每个指标的分数汇总 |
| `overall_average` | `number` | 所有指标平均分，保留 2 位小数 |
| `raw_response` | `object` | 评测模型的原始解析结果与 token 使用信息 |

## 错误码

| 状态码 | 场景 | 说明 |
| --- | --- | --- |
| `400` | 消息无效 | `messages` 中没有可用于评测的 `user/assistant` 内容 |
| `404` | 场景不存在 | `scene` 未在 `agent_list.json` 中注册，或 YAML 配置不存在 |
| `502` | 模型调用失败 | 评测模型请求异常 |
| `500` | 模型结果异常 | 模型没有返回可解析的结构化结果 |

## 调用示例

### curl

```bash
curl -X POST "http://127.0.0.1:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "scene": 138,
    "messages": [
      {"role": "user", "content": "我总是分不清渗透和扩散。"},
      {"role": "assistant", "content": "可以从定义、发生条件和生活例子三个角度来区分渗透和扩散。"}
    ]
  }'
```

### Python

```python
import requests

payload = {
    "scene": 138,
    "messages": [
        {"role": "user", "content": "我总是分不清渗透和扩散。"},
        {
            "role": "assistant",
            "content": "可以从定义、发生条件和生活例子三个角度来区分渗透和扩散。",
        },
    ],
}

response = requests.post("http://127.0.0.1:8000/evaluate", json=payload, timeout=120)
response.raise_for_status()
print(response.json())
```

## 自测脚本

仓库内提供了 `server_test.py`，会直接向本地服务发请求：

```bash
python server.py
python server_test.py
```

最近一次实测返回了：

- `scene_id`: `138`
- `scene_name`: `初中生物术语辨析助手`
- `overall_average`: `2.88`
- `raw_response.usage.total_tokens`: `4749`
