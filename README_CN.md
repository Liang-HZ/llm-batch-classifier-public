# LLM Batch Classifier

面向批量分类任务的 LLM 工具：给你一份 CSV/Excel、一个标签列表和一个提示词，它会帮你稳定地跑完整批数据，带限流、断点续跑和失败重试。

[![PyPI version](https://badge.fury.io/py/llm-batch-classifier.svg)](https://pypi.org/project/llm-batch-classifier/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

[English](README.md) | 中文

## 先说结论

如果你只关心一件事：

> “我有很多条文本，想让 LLM 按固定标签批量分类，而且中途不要炸、超时后能续跑、失败的还能重试。”

这个项目就是干这个的。

它最适合：

- 固定标签体系的批量分类，例如专业分类、工单分类、意图分类、内容标签、行业归类
- 大批量任务，运行时间长，API 有速率限制
- 你已经有一列旧标签，想批量复跑并对比新旧结果

它不适合：

- 任意 JSON 字段抽取
- 开放式问答、总结、生成型任务
- 多机分布式共享同一个 API key 的在线服务

## 3 分钟跑通：直接用仓库自带示例

如果你是第一次看这个项目，先不要写自己的配置。先跑通内置示例。

```bash
git clone <your-repo-url>
cd llm-batch-classifier
python -m pip install -e .
export LLM_API_KEY=your-api-key
llm-classify run --config examples/university-programs/classify.yaml
```

Windows PowerShell:

```powershell
$env:LLM_API_KEY="your-api-key"
```

这个示例会把 20 条大学专业名称分类到 12 个公开演示用的大类标签中。相关文件在：

- [examples/university-programs/classify.yaml](examples/university-programs/classify.yaml)
- [examples/university-programs/prompt.txt](examples/university-programs/prompt.txt)
- [examples/university-programs/sample_input.csv](examples/university-programs/sample_input.csv)
- [examples/university-programs/README.md](examples/university-programs/README.md)

跑完以后，你最关心这 3 个结果：

- `output/run_时间戳_.../classification_result.csv`：最终分类结果
- `output/run_时间戳_.../classification_report.md`：统计报告
- `output/run_时间戳_.../run_summary.json`：机器可读的运行摘要

## 5 分钟用你自己的数据

### 第 1 步：准备一个 CSV 或 Excel 文件

最简单的输入长这样：

```csv
text,context
MSc Finance,金融学硕士
MSc Computer Science,计算机科学硕士
MBA,工商管理硕士
```

- `text`：要分类的主文本
- `context`：可选的补充上下文，没有就留空

### 第 2 步：生成配置模板

```bash
llm-classify init
```

这会生成一个 `classify.yaml`。

### 第 3 步：最少只改这几项

第一次使用时，不用把所有配置都看懂。优先改下面这几项就够了：

```yaml
categories:
  - "金融"
  - "计算机"
  - "管理"

prompt:
  system: |
    你是一位分类专家。请将输入内容分到以下类别中：
    {categories}

    只返回 JSON。
    输出格式：
    {{"labels": [{{"name": "类别名称", "confidence": 95, "reason": "分类理由"}}]}}
  user: "{text} / {context}"

model:
  name: deepseek-chat
  api_base: https://api.deepseek.com/v1

input:
  file: data.csv
  text_column: text
  context_column: context
```

你要理解的只有 4 件事：

1. `categories`：你自己的目标标签
2. `prompt.system`：告诉模型“只能从这些标签里选”
3. `model`：你要调用哪个 OpenAI-compatible API
4. `input`：你的文件路径和列名

如果你没有 `context` 列：

- 把 `context_column` 设成空字符串
- 把 `prompt.user` 改成 `"{text}"`

### 第 4 步：正式运行

```bash
llm-classify run --config classify.yaml
```

先不确定配置对不对时，建议先跑：

```bash
llm-classify run --config classify.yaml --dry-run
llm-classify run --config classify.yaml --test 20
```

- `--dry-run`：只检查配置和工作量，不调用 API
- `--test 20`：只跑前 20 条，适合先验证提示词

### 第 5 步：看结果

默认结果在 `output/` 目录。

最重要的字段有：

- `label`：最终分类标签，多个标签会用 `|` 拼接
- `confidence`：最高置信度
- `is_low_confidence`：是否低于阈值
- `parse_status`：解析/校验状态

如果任务中断了，继续跑：

```bash
llm-classify run --config classify.yaml --resume
```

如果想重试失败项：

```bash
llm-classify retry output/run_xxx/classification_result.csv
```

## 小白最容易踩的坑

### 1. `401` / `403`

通常是下面两个原因之一：

- `LLM_API_KEY` 没设置
- `model.api_base` 填错了，不是正确的 OpenAI-compatible endpoint

### 2. 提示 `missing columns`

说明你的文件列名和配置不一致。重点检查：

- `input.text_column`
- `input.context_column`

### 3. 提示词里的 JSON 花括号报错

在 YAML 里写 JSON 示例时，字面量花括号要转义：

- 写成 `{{`
- 和 `}}`

不要直接写裸 `{` 和 `}`。

### 4. 一直遇到 `429`

先不要一味提高重试次数，先检查：

- `rate_limit.rps`
- `rate_limit.tps`
- `concurrency`

保守一点通常更稳。

## 它是怎么工作的（白话版）

1. 读取你的 CSV/Excel
2. 按 `text + 可选 context` 去重
3. 把每条数据和标签列表一起发给 LLM
4. 校验模型返回的标签是否在你的标签列表里
5. 每处理完一条就立刻写盘，所以中断后可以续跑
6. 遇到超时或 `429` 自动重试，遇到坏结果单独标记出来

如果你想知道更技术一点的实现：

- 限流：滑动窗口控制 RPS/TPS，再加一个可选的周期预算上限
- 断点续跑：每条结果单独落盘，不等整批结束
- 自动重试：区分瞬时错误和语义错误

## 常用命令

```text
llm-classify run --config FILE         执行批量分类
  --resume                             从上次中断处继续（追加模式）
  --fresh                              运行前清空历史结果
  --dry-run                            仅展示配置和预估工作量，不调用 API
  --test N                             仅处理前 N 条数据
  --random N                           随机抽取 N 条数据处理
  --concurrency N                      覆盖配置文件中的并发数
  --input-csv FILE                     使用已有 CSV 进行重新分类对比

llm-classify retry SOURCE              对结果 CSV 中的失败条目自动重试
  --config FILE                        YAML 配置文件（可自动推断，可省略）
  --max-rounds N                       最大重试轮次（默认：3）
  --dry-run                            仅展示重试计划，不调用 API
  --concurrency N                      覆盖并发数

llm-classify init                      生成初始 classify.yaml 模板
  --output FILE                        输出路径（默认：classify.yaml）
```

## 完整配置参考

当你已经跑通第一次任务后，再来看这一节。

```yaml
# LLM Batch Classifier 配置文件

categories:
  - "类别 A"
  - "类别 B"
  - "类别 C"

prompt:
  # {categories} 会自动从上方列表注入
  system: |
    你是一位分类专家。请将输入内容分到以下类别中：
    {categories}

    要求：
    1. 选取所有匹配的类别，并给出置信度分数（0-100）
    2. 只包含置信度 >= 85 的类别
    3. 使用上方列表中的精确类别名称
    4. 仅输出 JSON

    输出格式：
    {{"labels": [{{"name": "类别名称", "confidence": 95, "reason": "分类理由"}}]}}
  # {text} 和 {context} 来自你的输入文件列
  user: "{text} / {context}"

  # 也可以从独立文件加载系统提示词
  # system_file: prompt.txt

model:
  name: deepseek-chat                    # 模型标识符
  api_base: https://api.deepseek.com/v1  # 任意 OpenAI 兼容的 Base URL
  temperature: 0.1
  max_tokens: 500
  timeout: 30
  max_retries: 3

rate_limit:
  rps: 3
  tps: 0
  window: 1
  tokens_per_call: 850

cycle:
  duration: 60
  max_calls: 180

throttle:
  max_attempts: 10
  base_wait: 30.0
  max_wait: 300.0
  jitter: 0.5

input:
  file: data.csv
  text_column: text
  context_column: context

output:
  dir: output
  format: auto  # auto | csv | xlsx

threshold: 95
concurrency: 15
```

## 什么时候用 `--input-csv`

如果你的输入文件里已经有一列旧标签，想看“同一批数据换了模型/提示词后结果变了多少”，就用：

```bash
llm-classify run --config classify.yaml --input-csv old_results.csv
```

运行后会额外生成：

- `compare_old_label`
- `compare_is_match`
- `classification_diff.csv`

这很适合做提示词回归测试。

## 为什么这个项目值得用

- 不只是 `for row in csv: call_llm(row)` 的脚本
- 能在长任务中稳定运行
- 限流、续跑、重试都是开箱即用
- 用 YAML 改任务，不用改代码

## 参与贡献

欢迎贡献代码。提交较大的 Pull Request 前，请先开一个 Issue 讨论方案。

1. Fork 仓库并创建功能分支
2. 安装开发依赖：`python -m pip install -e ".[dev]"`
3. 运行测试：`pytest`
4. 提交 Pull Request，并说明你的修改内容

## License

[MIT](https://opensource.org/licenses/MIT) — Copyright (c) 2024 LLM Batch Classifier contributors.
