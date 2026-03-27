# 可视化脚本

这个目录提供了一个基于 `matplotlib` 的评测结果可视化脚本，会自动读取 `eval_result/[场景]/[模型].json` 并生成多张图表。

## 运行方式

在仓库根目录执行：

```bash
python visual/plot_eval_results.py
```

如需指定输入、输出目录或图片清晰度：

```bash
python visual/plot_eval_results.py \
  --input-dir eval_result \
  --output-dir visual/output \
  --dpi 200
```

## 生成内容

- `overall_normalized_heatmap.png`：跨场景综合得分热力图，已按每个场景的分值上限归一化，便于横向比较。
- `overall_by_scene.png`：每个场景下不同模型的综合均分柱状图。
- `model_leaderboard.png`：模型在全部场景上的平均综合得分率排行榜。
- `scene_01_metric_heatmap.png` 等：每个场景的各维度平均得分热力图。
- `generated_files.txt`：输出文件清单。

## 说明

- 脚本会自动扫描全部 `json` 文件，不需要手工维护模型或场景列表。
- 如果 `score_range.max` 与实际得分不一致，脚本会结合场景内观测到的得分自动推断一个更合理的上限，再进行归一化。
- 脚本会尝试自动选择系统里的中文字体，避免图表中文乱码。
