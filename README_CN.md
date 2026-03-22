# LLM Batch Classifier

如果你需要把一大批 CSV/Excel 文本按固定标签交给 LLM 分类，而且希望流程稳定、可续跑、好排查，这个项目就是给你用的。

[![PyPI version](https://badge.fury.io/py/llm-batch-classifier.svg)](https://pypi.org/project/llm-batch-classifier/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

[English](README.md) | 中文

## 你可以用它做什么

这个仓库适合这样的读者：

- 固定标签体系的批量分类，例如专业分类、工单分类、意图分类、内容标签、行业归类
- 大批量任务，运行时间长，API 有速率限制
- 你已经有一列旧标签，想批量复跑并对比新旧结果

它不适合：

- 任意 JSON 字段抽取
- 开放式问答、总结、生成型任务
- 多机分布式共享同一个 API key 的在线服务

## 仓库导航

- 从这里开始：[3 分钟跑通](#3-分钟跑通直接用仓库自带示例)
- 跑你自己的数据：[5 分钟用你自己的数据](#5-分钟用你自己的数据)
- 示例文件：[examples/university-programs](examples/university-programs)
- 提问或报问题：[GitHub Issues](https://github.com/Liang-HZ/llm-batch-classifier-public/issues/new/choose)
- 贡献说明：[CONTRIBUTING.md](CONTRIBUTING.md)
- 安全策略：[SECURITY.md](SECURITY.md)
- 社区规范：[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

## 开始前你需要准备什么

只要 3 样东西：

- Python 3.11 或更高版本
- 一个 OpenAI-compatible API 的 key
- 一份至少包含一列主文本的 CSV 或 Excel

## 3 分钟跑通：直接用仓库自带示例

如果你是第一次看这个项目，先不要写自己的配置。先跑通内置示例。

```bash
git clone https://github.com/Liang-HZ/llm-batch-classifier-public.git
cd llm-batch-classifier-public
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

## 遇到问题去哪里

按问题类型走这几个入口：

- 使用问题和 Bug：去 [GitHub Issues](https://github.com/Liang-HZ/llm-batch-classifier-public/issues/new/choose)
- 安全相关问题：看 [SECURITY.md](SECURITY.md)
- 想参与贡献：看 [CONTRIBUTING.md](CONTRIBUTING.md)

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

## `window` 和 `cycle` 到底有什么区别

这两个配置不是重复的，它们管的是两层不同的事情：

- `rate_limit` 管短时间内的发送节奏
- `cycle` 管更长时间段里的总预算

你可以这样理解 `rate_limit`：

> “我现在这一小段时间里，最多能打多快？”

- `rate_limit.rps`：滑动窗口内最多允许多少次请求
- `rate_limit.tps`：滑动窗口内最多允许多少估算 token
- `rate_limit.window`：这个“滑动窗口”往前看多少秒
- `rate_limit.tokens_per_call`：当启用 `tps` 时，单次请求大约按多少 token 估算

你可以这样理解 `cycle`：

> “拉长到一分钟、一小时这种尺度，我总共最多能打多少次？”

- `cycle.duration`：一个周期有多长，单位秒
- `cycle.max_calls`：这个周期里最多允许多少次 API 调用

一句话区分：

- `rate_limit` 负责把流量打平，避免短时间冲得太猛
- `cycle` 负责卡住长一点时间范围内的总调用量

例子：

- `rate_limit.rps: 3` 且 `rate_limit.window: 1`，表示任意 1 秒窗口内最多 3 个请求
- `cycle.duration: 60` 且 `cycle.max_calls: 180`，表示每 60 秒周期内最多 180 次调用

对大多数用户来说：

- 先只配 `rate_limit`
- 只有当你的提供商限额或你自己的预算表达方式是“每分钟/每小时最多 N 次”时，再加 `cycle`

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
#
# API key 不放在 YAML 里，而是从环境变量读取：
#   export LLM_API_KEY=your-key
#   export OPENAI_API_KEY=your-key

# 列出模型允许返回的全部标签。
# 模型输出的标签名称必须和这里完全一致。
categories:
  - "类别 A"
  - "类别 B"
  - "类别 C"

# 每条输入都会使用到的提示词配置。
prompt:
  # 主系统提示词。
  # {categories} 会自动替换成上方的标签列表。
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

  # 也可以不用上面的 prompt.system，而是从独立文本文件加载长提示词。
  # system_file: prompt.txt

  # 用户提示词模板，从输入文件列中拼出来。
  # 这里只支持 {text} 和 {context} 两个占位符。
  user: "{text} / {context}"

# 模型与 API 端点配置。
model:
  # 提供商识别的模型名称。
  name: deepseek-chat

  # 任意 OpenAI 兼容接口的 Base URL。
  api_base: https://api.deepseek.com/v1

  # 分类任务通常建议用较低温度，结果更稳定。
  temperature: 0.1

  # 单条请求允许模型返回的最大 token 数。
  max_tokens: 500

  # 单次请求超时时间，单位秒。
  timeout: 30

  # 单次 API 调用内部对瞬时错误的重试次数。
  max_retries: 3

# 滑动窗口限流配置，用来控制短时间内的发送节奏。
# 它回答的是：“我现在这一小段时间里最多能打多快？”
rate_limit:
  # 每秒最多请求数。设为 0 表示关闭按请求数限流。
  rps: 3

  # 每秒最多 token 数。设为 0 表示关闭按 token 限流。
  tps: 0

  # 滑动窗口大小，单位秒。
  window: 1

  # 单次请求大约会消耗多少 token。只有在 tps > 0 时才会用到。
  tokens_per_call: 850

# 更长周期的调用预算上限。
# 它回答的是：“拉长一点看，我总共最多能打多少次？”
# 两个字段都设为 0 就表示关闭。
cycle:
  # 一个预算周期有多长，单位秒。
  duration: 60

  # 一个周期内最多允许多少次 API 调用。
  max_calls: 180

# 遇到 429 等限流错误时的退避重试配置。
throttle:
  # 最多做多少次退避重试。
  max_attempts: 10

  # 第一次重试前等待多久，单位秒。
  base_wait: 30.0

  # 指数退避等待时间的上限，单位秒。
  max_wait: 300.0

  # 每次等待时额外加入的随机抖动，单位秒。
  jitter: 0.5

# 输入文件与列映射。
input:
  # 输入文件路径，支持 CSV 和 Excel。
  file: data.csv

  # 要分类的主文本列名。
  text_column: text

  # 可选的补充上下文列名。
  context_column: context

# 输出目录与格式。
output:
  # 结果文件和报告写入到哪个目录。
  dir: output

  # auto 表示跟随输入类型，也可以强制写成 csv 或 xlsx。
  format: auto

# 低于这个置信度阈值的标签会被过滤掉。
threshold: 95

# 同时允许多少个请求在飞行中。
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

如果你想提代码、改文档或补测试，先看 [CONTRIBUTING.md](CONTRIBUTING.md)。

简要流程是：

1. Fork 仓库并创建功能分支
2. 安装开发依赖：`python -m pip install -e ".[dev]"`
3. 运行测试：`pytest`
4. 提交 Pull Request，并说明你的修改内容

提交前也建议阅读：

- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- [SECURITY.md](SECURITY.md)

## License

[MIT](https://opensource.org/licenses/MIT) — Copyright (c) 2024 LLM Batch Classifier contributors.
