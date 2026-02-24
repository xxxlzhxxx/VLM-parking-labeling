# VLM-parking-labeling

用于对比多模态模型（Doubao Seed 1.8、Seed 2.0 Pro）在停车场目标检测/框选任务上的效果，并输出：

- 预测框可视化对比图
- bbox 文本结果
- CSV 汇总（包含 Precision、Recall、IoU、推理耗时、Tokens 等）
- 实验结论报告（docs）

## 目录结构

```
VLM-parking-labeling/
├── compare.py                  # 旧版主脚本（seed-1.8 vs Qwen3 对比）
├── test_prompt_comparison.py   # 新版多模型×多提示词对比测试脚本
├── runners/                    # 模型推理 Runner
│   ├── seed18_runner.py        #   Doubao Seed 1.8 Runner
│   └── seed20_runner.py        #   Doubao Seed 2.0 Pro Runner
├── prompts/                    # 提示词文件
│   ├── prompt优化.md           #   优化后的提示词
│   └── prompt原始.md           #   原始提示词
├── data/
│   ├── annotations/            # 标注 XML（与图片同名，Pascal VOC 格式）
│   └── images/                 # 本地图片缓存（自动下载）
├── outputs/
│   ├── csv/                    # CSV 汇总结果
│   └── visualizations/         # 可视化对比图
├── docs/                       # 实验对比报告/结论
├── .env_example                # 环境变量示例
├── .env                        # 环境变量（git ignored）
└── env                         # 环境变量（git ignored）
```

## 环境准备

### Python 版本

建议使用 **Python 3.9+**。

### 安装依赖

```bash
pip install requests pillow openai volcengine-sdk-ark-runtime
```

## 配置 API Key

在项目根目录创建 `env` 或 `.env` 文件（可参考 `.env_example`）：

```bash
ARK_API_KEY=YOUR_ARK_API_KEY

# 可选：为不同模型指定独立的 API Key 和 Endpoint
SEED18_API_KEY=YOUR_SEED18_API_KEY
SEED18_EP=ep-xxxx
SEED18_MODEL_ID=doubao-seed-1-8-251228

SEED20_API_KEY=YOUR_SEED20_API_KEY
SEED20_EP=ep-xxxx
SEED20_MODEL_ID=doubao-seed-2-0-pro-preview-260115
```

> **优先级**：命令行参数 > 环境变量 > `env` 文件 > `.env` 文件 > 脚本内默认值

## 运行

### 1. 多模型 × 多提示词对比测试（推荐）

使用 `test_prompt_comparison.py`，支持 Seed 1.8 与 Seed 2.0 Pro 的多维度对比。

#### 基本用法

```bash
python3 test_prompt_comparison.py
```

默认行为：测试全部模型 × 全部提示词 × 全部图片，每组跑 5 次，并发 5 线程。

#### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--models` | 要测试的模型 | 全部：`seed1.8 seed2.0-pro` |
| `--prompts` | 要测试的提示词 | 全部：`prompt优化 prompt优化_v2 prompt_原始` |
| `--runs` | 每个组合的测试次数 | `5` |
| `--max-concurrent` | 最大并发线程数 | `5` |
| `--images` | 指定测试图片文件名 | 全部图片 |
| `--ark-base-url` | Ark API 地址 | `https://ark.cn-beijing.volces.com/api/v3` |
| `--seed18-api-key` | Seed 1.8 API Key | 从环境变量读取 |
| `--seed18-ep` | Seed 1.8 Endpoint | 从环境变量读取 |
| `--seed20-api-key` | Seed 2.0 API Key | 从环境变量读取 |
| `--seed20-ep` | Seed 2.0 Endpoint | 从环境变量读取 |

#### 使用示例

**只用 `prompt优化` 对比两个模型，每组 5 次，5 线程并发：**

```bash
python3 test_prompt_comparison.py \
  --models seed1.8 seed2.0-pro \
  --prompts prompt优化 \
  --runs 5 \
  --max-concurrent 5
```

**只测试 Seed 2.0 Pro，跑 3 次：**

```bash
python3 test_prompt_comparison.py \
  --models seed2.0-pro \
  --runs 3
```

**测试指定图片（img1.jpg）：**

```bash
python3 test_prompt_comparison.py \
  --images img1.jpg
```

**使用命令行指定 API Key 和 Endpoint：**

```bash
python3 test_prompt_comparison.py \
  --seed18-api-key YOUR_KEY \
  --seed18-ep ep-xxxx \
  --seed20-api-key YOUR_KEY \
  --seed20-ep ep-xxxx
```

#### 输出

运行结束后会生成：

- **单组结果 CSV**：`outputs/csv/{模型}_{提示词}_{时间戳}.csv`
- **汇总对比 CSV**：`outputs/csv/model_prompt_comparison_{时间戳}.csv`
- **终端对比报告**：包含精度、召回率、IoU、错标/漏标、耗时、Tokens 等指标的对比表

示例输出：

```
📊 prompt优化 × img2.jpg
------------------------------------------------------------
  指标            seed1.8             seed2.0-pro         差异
  ----------------------------------------------------------------
  精度            66.9%               70.0%               +3.1% ✅
  召回率          55.0%               52.5%               -2.5% ⚠️
  IoU             0.666               0.687               +0.021 ✅
  错标(FP)        4.4                 3.6                 -0.8 ✅
  耗时(s)         56.4                54.2                -2.2 ✅
  Tokens          8370                5725                -2645 ✅
```

### 2. 旧版对比脚本

```bash
python3 compare.py
```

默认对 `image_urls` 中的每张图运行 5 次，并发数 5，对比 Seed 1.8 与 Qwen3。

## 评估指标

| 指标 | 说明 |
|------|------|
| Precision | 预测框中正确的比例（TP / 预测总数） |
| Recall | 真实框中被检出的比例（TP / GT 总数） |
| IoU | 匹配框对的平均交并比 |
| FP (错标) | 预测了但没有对应 GT 的框 |
| FN (漏标) | GT 中未被检出的框 |
| IoU 阈值 | 默认 0.5，匹配时 IoU ≥ 阈值才算 TP |

## 报告

实验对比结论保存在 `docs/` 下，可直接打开查看。
