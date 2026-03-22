# LLM Batch Classifier

A practical LLM tool for batch classification: give it a CSV/Excel file, a fixed label set, and a prompt, and it will run the whole job with rate limiting, checkpointing, and auto-retry.

[![PyPI version](https://badge.fury.io/py/llm-batch-classifier.svg)](https://pypi.org/project/llm-batch-classifier/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

English | [中文](README_CN.md)

## What This Is

If your problem sounds like this:

> “I have a lot of text rows, I want an LLM to classify them into a fixed set of labels, and I do not want the job to crash halfway through or burn budget on bad retries.”

This project is built for that.

It is a good fit for:

- Batch classification with a fixed taxonomy: intents, tickets, programs, content tags, product categories
- Long-running jobs against rate-limited APIs
- Re-running an already labeled CSV to compare old vs new labels

It is not a good fit for:

- Arbitrary structured extraction with custom JSON fields
- Open-ended generation or Q&A
- A distributed online service with multiple machines sharing one API key

## 3-Minute Start: Run the Built-In Example

If this is your first time here, do not start by writing your own config. Run the built-in example first.

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

This example classifies 20 university program names into 12 public demo categories. The relevant files are:

- [examples/university-programs/classify.yaml](examples/university-programs/classify.yaml)
- [examples/university-programs/prompt.txt](examples/university-programs/prompt.txt)
- [examples/university-programs/sample_input.csv](examples/university-programs/sample_input.csv)
- [examples/university-programs/README.md](examples/university-programs/README.md)

After the run, the 3 files you care about most are:

- `output/run_TIMESTAMP_.../classification_result.csv`: the final labels
- `output/run_TIMESTAMP_.../classification_report.md`: a human-readable report
- `output/run_TIMESTAMP_.../run_summary.json`: machine-readable run stats

## 5-Minute Start With Your Own Data

### 1. Prepare a CSV or Excel file

The simplest input looks like this:

```csv
text,context
MSc Finance,Finance master's program
MSc Computer Science,Computer science master's program
MBA,Business administration
```

- `text`: the main field to classify
- `context`: optional extra context

### 2. Generate a starter config

```bash
llm-classify init
```

This creates `classify.yaml`.

### 3. Edit only the minimum required fields

Do not try to understand every config option on day one. For your first run, focus on these:

```yaml
categories:
  - "Finance"
  - "Computer Science"
  - "Management"

prompt:
  system: |
    You are a classification expert. Classify the input into these categories:
    {categories}

    Output JSON only.
    Output format:
    {{"labels": [{{"name": "Category Name", "confidence": 95, "reason": "why"}}]}}
  user: "{text} / {context}"

model:
  name: deepseek-chat
  api_base: https://api.deepseek.com/v1

input:
  file: data.csv
  text_column: text
  context_column: context
```

The 4 things to understand are:

1. `categories`: your target labels
2. `prompt.system`: tell the model it must choose from those labels
3. `model`: which OpenAI-compatible API to call
4. `input`: your file path and column names

If you do not have a `context` column:

- set `context_column` to an empty string
- change `prompt.user` to `"{text}"`

### 4. Run it

```bash
llm-classify run --config classify.yaml
```

If you want a safer first run:

```bash
llm-classify run --config classify.yaml --dry-run
llm-classify run --config classify.yaml --test 20
```

- `--dry-run`: validate config and workload without making API calls
- `--test 20`: process only the first 20 rows

### 5. Read the output

By default, results go into `output/`.

The fields most people care about are:

- `label`: final label or labels, joined by `|`
- `confidence`: highest confidence score
- `is_low_confidence`: whether the result fell below your threshold
- `parse_status`: parsing and validation status

If the job stops halfway through, resume it:

```bash
llm-classify run --config classify.yaml --resume
```

If you want to retry failures:

```bash
llm-classify retry output/run_xxx/classification_result.csv
```

## Common Beginner Mistakes

### 1. `401` or `403`

Usually one of these is wrong:

- `LLM_API_KEY` is not set
- `model.api_base` is not the correct OpenAI-compatible endpoint

### 2. `missing columns`

Your file columns do not match the config. Check:

- `input.text_column`
- `input.context_column`

### 3. Prompt template errors around JSON braces

If you include a JSON example in YAML, literal braces must be escaped:

- `{{`
- `}}`

Do not use bare `{` and `}` in prompt examples.

### 4. Too many `429` errors

Before increasing retry counts, first check:

- `rate_limit.rps`
- `rate_limit.tps`
- `concurrency`

Being conservative is usually more stable.

## How It Works, In Plain English

1. Read your CSV/Excel file
2. Deduplicate rows by `text + optional context`
3. Send each item plus your label list to the LLM
4. Validate whether returned labels really belong to your label set
5. Write every item to disk immediately so the run can resume later
6. Retry timeouts and `429`s automatically, and mark bad outputs for follow-up

If you want the technical version:

- rate limiting uses a sliding window for RPS/TPS, plus an optional coarser cycle cap
- checkpointing writes each item immediately after the API response
- retries distinguish transient failures from semantic failures

## CLI Reference

```text
llm-classify run --config FILE         Run batch classification
  --resume                             Resume from a previous run (append mode)
  --fresh                              Clear previous results before running
  --dry-run                            Show config and estimated work, no API calls
  --test N                             Process only the first N items
  --random N                           Sample N random items
  --concurrency N                      Override concurrency from config
  --input-csv FILE                     Use existing CSV for re-classification

llm-classify retry SOURCE              Auto-retry failed items from a result CSV
  --config FILE                        YAML config (auto-detected if omitted)
  --max-rounds N                       Maximum retry rounds (default: 3)
  --dry-run                            Show retry plan, no API calls
  --concurrency N                      Override concurrency

llm-classify init                      Generate a starter classify.yaml
  --output FILE                        Output path (default: classify.yaml)
```

## Full Configuration Reference

Read this after your first successful run.

```yaml
# LLM Batch Classifier Configuration

categories:
  - "Category A"
  - "Category B"
  - "Category C"

prompt:
  # {categories} is injected automatically from the list above
  system: |
    You are a classification expert. Classify the input into these categories:
    {categories}

    Requirements:
    1. Select all matching categories with confidence scores (0-100)
    2. Only include categories with confidence >= 85
    3. Use exact category names from the list above
    4. Output JSON only

    Output format:
    {{"labels": [{{"name": "Category Name", "confidence": 95, "reason": "reason"}}]}}
  # {text} and {context} come from your input file columns
  user: "{text} / {context}"

  # Alternative: load system prompt from a separate file
  # system_file: prompt.txt

model:
  name: deepseek-chat                    # Model identifier
  api_base: https://api.deepseek.com/v1  # Any OpenAI-compatible base URL
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

## When To Use `--input-csv`

If your input file already contains an old `label` column and you want to compare old vs new results after changing the model or prompt:

```bash
llm-classify run --config classify.yaml --input-csv old_results.csv
```

This adds:

- `compare_old_label`
- `compare_is_match`
- `classification_diff.csv`

That makes it useful for prompt regression testing.

## Why Use This Instead of a Simple Script

- It is not just `for row in csv: call_llm(row)`
- It is designed for long-running batch jobs
- Rate limiting, resume, and retries are built in
- You switch tasks by editing YAML, not code

## Contributing

Contributions are welcome. Please open an issue before a large pull request so we can discuss the approach.

1. Fork the repository and create a feature branch
2. Install dev dependencies: `python -m pip install -e ".[dev]"`
3. Run tests: `pytest`
4. Open a pull request with a clear description of the change

## License

[MIT](https://opensource.org/licenses/MIT) — Copyright (c) 2024 LLM Batch Classifier contributors.
