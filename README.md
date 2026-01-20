# VLM-parking-labeling

用于对比多模态模型（seed-1.8、Qwen3）在停车场目标检测/框选任务上的效果，并输出：

- 预测框可视化对比图
- bbox 文本结果
- CSV 汇总（包含 IoU、推理耗时、tokens 等）
- 实验结论报告（docs）

## 目录结构

- `compare.py`：主脚本（并发跑多次，对比 seed-1.8 vs Qwen3）
- `prompts/`：提示词
- `data/`
  - `annotations/`：标注 XML（与图片同名）
  - `images/`：本地图片缓存（可选）
- `outputs/`
  - `csv/`：CSV 汇总结果
  - `visualizations/`：可视化对比图
  - `bboxes/`：bbox 文本（含 IoU）
  - `tmp/`：下载的临时图片
- `docs/`：实验对比报告/结论

## 环境准备

建议使用 Python 3.9+。

安装依赖（按你本地环境自行安装）：

- `requests`
- `pillow`
- `openai`
- `volcengine-sdk-ark-runtime`（或项目中对应的 `volcenginesdkarkruntime`）

## 配置 API Key

在项目根目录创建 `.env`（可参考 `.env_example`）：

```bash
DASHSCOPE_API_KEY=YOUR_DASHSCOPE_API_KEY
ARK_API_KEY=YOUR_ARK_API_KEY
```

脚本会优先从环境变量读取，若未配置再读取根目录的 `.env`。

## 运行

直接运行：

```bash
python3 compare.py
```

默认行为：

- 对 `image_urls` 中的每张图运行 `runs_per_image=5` 次
- 并发数 `max_concurrent=5`
- 读取 `prompts/` 下的提示词文件（按 `compare.py` 中的候选列表自动选择）
- 读取 `data/annotations/*.xml` 作为 GT

输出位置：

- `outputs/csv/model_comparison_results_*.csv`
- `outputs/visualizations/result_with_bboxes_*.jpg`
- `outputs/bboxes/bbox_info_*.txt`

## 报告

实验对比结论保存在 `docs/` 下，可直接打开查看。

