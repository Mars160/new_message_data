# 统计yaml中每个指标出现的频次

import yaml
from pathlib import Path

metrics_count = {}

yamls = list(
    (Path(__file__).parent.parent / "generate_message" / "scene_output").glob("*.yaml")
)
for yaml_file in yamls:
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)
        formats = data["evaluation"]["format"]
        for format_ in formats:
            name = format_["field"]
            if name.endswith("_理由"):
                continue

            if name not in metrics_count:
                metrics_count[name] = 0
            metrics_count[name] += 1

# 打印Top30
sorted_metrics = sorted(metrics_count.items(), key=lambda x: x[1], reverse=True)
for metric, count in sorted_metrics[:30]:
    print(f"{metric}: {count}")
