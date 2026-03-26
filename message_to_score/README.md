# 教学对话评估工具

基于 GPT-4o 的自动化教学对话质量评估系统，使用 OpenAI 结构化输出对生成的教学对话进行多维度评分。

## 功能特性

- **多维度评估**：根据 YAML 配置中的评分标准，对教学对话进行多指标综合评估
- **批量处理**：支持异步并行评估多个对话案例，带进度条显示
- **结构化输出**：使用 OpenAI 结构化输出，确保评分格式统一
- **统计分析**：自动计算各项指标平均分及总体平均分
- **灵活配置**：支持命令行参数指定评估模型、目标模型和评测场景
- **在线评测 API**：支持通过 `server.py` 提供单条教学对话评分接口

## API 文档

在线评测接口说明见 `API.md`。

## 快速开始

### 1. 配置环境

```bash
# 安装依赖
pip install openai pydantic pyyaml tqdm

# 或如果使用 uv
uv sync
```

### 2. 配置 API

复制 `secret.json.example` 为 `secret.json`，填写你的 OpenAI API 配置：

```json
{
  "base_url": "https://api.openai.com/v1",
  "api_key": "your-api-key"
}
```

### 3. 准备数据

将教学对话数据放在 `message_data/default_generate/{agent_name}/` 目录下，格式为：

```
message_data/
└── default_generate/
    └── {agent_name}/
        └── {date}/
            └── {uuid}/
                └── conversation-messages.json
```

`conversation-messages.json` 格式示例：

```json
{
  "messages": [
    { "role": "user", "content": "学生提问内容" },
    { "role": "assistant", "content": "AI助手回答内容" }
  ]
}
```

### 4. 运行评估

```bash
# 评估指定场景
python eval.py

# 评估所有场景
python eval.py --test-all

# 评估指定场景列表
python eval.py --test-scenes "场景A" "场景B"

# 指定评估模型和目标模型
python eval.py --evaluator-model gpt-4o-mini --target-model "其他模型"
```

### 5. 启动在线评测 API

```bash
python server.py
```

启动后可访问：

- 接口地址：`http://127.0.0.1:8000/evaluate`
- 详细文档：`API.md`

## 输出结果

评估结果保存在 `evaluation_results/{agent_name}_evaluation_results.json`，格式如下：

```json
{
  "overall": {
    "evaluator_model": "gpt-4o",
    "target_model": "Kimi-k2.5",
    "total_cases": 10,
    "average_scores": {
      "个性化画像适配度": 4.2,
      "指令遵循精准性": 4.5,
      "教学方法科学性": 4.3
    },
    "overall_average": 4.33
  },
  "items": [
    {
      "file_path": "/path/to/conversation-messages.json",
      "conversation": "Student: ...\nTeacher: ...",
      "metrics_results": [
        {
          "metric_name": "个性化画像适配度",
          "score": 4,
          "reason": "回答充分考虑了学生的基础水平..."
        }
      ],
      "raw_response": {
        "model": "gpt-4o",
        "usage": {...},
        "parsed_result": {...}
      }
    }
  ]
}
```

## 项目结构

```
message_to_score/
├── eval.py                    # 主评估脚本
├── agent_list.json            # Agent ID 映射表
├── secret.json                # API 配置（需自行创建）
├── secret.json.example        # API 配置示例
├── message_data/              # 对话数据目录
│   └── default_generate/
│       └── {agent_name}/
│           └── .../conversation-messages.json
├── evaluation_results/        # 评估结果输出目录
└── generate_message/          # 场景配置（上级目录）
    └── scene_output/
        └── scene{id}.yaml     # 评分标准配置
```

## 评分标准配置

评分标准从 `../generate_message/scene_output/scene{id}.yaml` 读取，格式示例：

```yaml
evaluation:
  format:
    - field: "个性化画像适配度"
      type: "float"
      description: "评价回答是否充分考虑学生的具体画像..."
    - field: "个性化画像适配度_理由"
      type: "str"
      description: "解释个性化画像适配度评分的详细理由"
```

## 命令行参数

| 参数                | 简写  | 说明           | 默认值                   |
| ------------------- | ----- | -------------- | ------------------------ |
| `--test-all`        | `-ta` | 评估所有场景   | False                    |
| `--test-scenes`     | `-ts` | 指定场景列表   | ["初中生物术语辨析助手"] |
| `--evaluator-model` | `-em` | 评估模型       | gpt-4o                   |
| `--target-model`    | `-tm` | 被评估模型名称 | Kimi-k2.5                |

## 依赖项

- Python 3.9+
- openai
- pydantic
- pyyaml
- tqdm

## 注意事项

1. 确保 `secret.json` 中的 `base_url` 已替换为实际的 API 地址
2. `message_data` 目录默认在 `.gitignore` 中，不会被提交到版本控制
3. 每个场景需要先在 `agent_list.json` 中注册
4. 对应的 YAML 评分标准文件需要存在于 `generate_message/scene_output/` 目录

## License

MIT
