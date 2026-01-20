# Seed-1.8 vs Qwen3 测试比较结果（IoU 重点）

## 概述

- 对比对象：豆包 seed-1.8 vs Qwen3（compare.py 中的 `qwen3-vl-plus`）
- 测试输入：2 张图片（img1.jpg、img2.jpg），每张图片计划运行 5 次
- 结果记录：`../outputs/csv/model_comparison_results_20260120_163816.csv`
- 本次执行：共 10 次计划运行，成功写入 CSV 的数据行 9 条（另 1 次因 `Connection error` 失败未入表）

## 结论摘要

- 总体上 seed-1.8 平均 IoU 更高（+0.0310），但不同图片优势方向不一致
- img1.jpg 上 seed-1.8 明显更优；img2.jpg 上 Qwen3 略优
- Qwen3 波动更大（标准差更高），端到端推理耗时也更长

## 数据来源

- CSV：`../outputs/csv/model_comparison_results_20260120_163816.csv`
- 结果图像（9 张）：
  - `../outputs/visualizations/result_with_bboxes_20.jpg`
  - `../outputs/visualizations/result_with_bboxes_21.jpg`
  - `../outputs/visualizations/result_with_bboxes_22.jpg`
  - `../outputs/visualizations/result_with_bboxes_23.jpg`
  - `../outputs/visualizations/result_with_bboxes_24.jpg`
  - `../outputs/visualizations/result_with_bboxes_25.jpg`
  - `../outputs/visualizations/result_with_bboxes_26.jpg`
  - `../outputs/visualizations/result_with_bboxes_27.jpg`
  - `../outputs/visualizations/result_with_bboxes_28.jpg`

## IoU 指标口径（与脚本一致）

- 标注来源：`../data/annotations/*.xml`
- bbox 后处理：脚本会把模型输出 bbox 转到像素坐标并裁剪到图像范围
- IoU 计算：对每个预测框，找与其 IoU 最大的 GT 框；得到一组 IoU
- 平均 IoU：只统计 IoU > 0.1 的预测框后再求平均（若没有满足条件的框，则平均 IoU 记为 0）

## 总体 IoU 对比（9 条有效样本）

| 指标 | seed-1.8 | Qwen3 | seed-1.8 - Qwen3 |
|---|---:|---:|---:|
| 样本数 | 9 | 9 | 9 |
| 平均值 | 0.5686 | 0.5375 | +0.0310 |
| 中位数 | 0.5721 | 0.5346 | +0.0332 |
| 最小值 | 0.5345 | 0.4527 | -0.1032 |
| 最大值 | 0.5949 | 0.6735 | +0.1421 |
| 标准差（总体） | 0.0211 | 0.0658 | 0.0766 |

胜负统计（按单条样本的平均 IoU）：

- seed-1.8 更高：7 / 9
- Qwen3 更高：2 / 9
- 持平：0 / 9

## 分图片 IoU 对比（按图片聚合）

| 图片 | 样本数 | seed-1.8 平均 IoU | Qwen3 平均 IoU | seed-1.8 - Qwen3 |
|---|---:|---:|---:|---:|
| img1.jpg | 4 | 0.5821 | 0.4883 | +0.0938 |
| img2.jpg | 5 | 0.5578 | 0.5769 | -0.0191 |

解读：

- img1.jpg 上 seed-1.8 明显更优（平均 IoU 高约 0.094）
- img2.jpg 上 Qwen3 略优（平均 IoU 高约 0.019）
- Qwen3 的波动更大（总体标准差更高），表现更“尖锐”：有更高的上限（最大 0.6735），也有更低的下限（最小 0.4527）

## 推理耗时与 Token（9 条样本）

说明：这里的“推理时间(s)”为端到端耗时（包含网络与服务端处理时间），并非纯模型计算时间。

| 指标 | seed-1.8 | Qwen3 |
|---|---:|---:|
| 推理时间均值（s） | 76.28 | 154.60 |
| 推理时间中位数（s） | 73.10 | 94.33 |
| 推理时间最小/最大（s） | 53.38 / 113.05 | 78.60 / 362.49 |
| 输入 tokens（均值） | 6093 | 9024 |
| 输出 tokens（均值） | 3816.56 | 3782.33 |

按图片聚合的推理时间均值（s）：

| 图片 | seed-1.8 | Qwen3 |
|---|---:|---:|
| img1.jpg | 95.01 | 229.70 |
| img2.jpg | 61.30 | 94.51 |

## 备注

- 本次计划运行 10 次，缺失的 1 次是调用 Qwen3 时出现 `Connection error`，未生成对应行数据与对比图。
- 若需要更稳的批量对比，可在 compare.py 的模型调用处加入重试机制（例如对连接错误进行指数退避重试），以减少偶发失败导致的样本缺失。
