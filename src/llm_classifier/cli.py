"""Typer CLI entrypoint for llm-batch-classifier."""

from __future__ import annotations

import typer
from pathlib import Path

app = typer.Typer(
    name="llm-classify",
    help="LLM Batch Classifier — classify CSV data using LLM APIs with rate limiting, checkpointing, and auto-retry.",
    add_completion=False,
)


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
    resume: bool = typer.Option(False, "--resume", help="Resume from previous run (append mode)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show config and estimated work without calling API"),
    test: int = typer.Option(None, "--test", help="Test mode: process only first N items"),
    random: int = typer.Option(None, "--random", help="Random mode: sample N random items"),
    concurrency: int = typer.Option(None, "--concurrency", help="Override concurrency from config"),
    input_csv: str = typer.Option(None, "--input-csv", help="Use existing CSV for re-classification comparison"),
    fresh: bool = typer.Option(False, "--fresh", help="Clear previous results before running"),
) -> None:
    """Run batch classification from a YAML config."""
    from .config import ClassifyConfig
    from .runner import run as runner_run

    cfg = ClassifyConfig.from_yaml(config)
    if concurrency is not None:
        cfg.concurrency = concurrency
    if not dry_run:
        cfg.validate()
    runner_run(cfg, test=test, random=random, input_csv=input_csv,
               append=resume, fresh=fresh, dry_run=dry_run)


@app.command()
def retry(
    source: Path = typer.Argument(..., help="Path to result CSV to retry failed items from"),
    config: Path = typer.Option(None, "--config", "-c", help="Path to YAML config (uses source dir if omitted)"),
    max_rounds: int = typer.Option(3, "--max-rounds", help="Maximum retry rounds"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show retry plan without calling API"),
    concurrency: int = typer.Option(None, "--concurrency", help="Override concurrency"),
) -> None:
    """Auto-retry failed items from a previous classification result."""
    from .config import ClassifyConfig
    from .runner import run_retry as runner_retry

    if config is None:
        # Try to find classify.yaml in the same directory as the source CSV
        config = source.parent / "classify.yaml"
        if not config.exists():
            typer.echo(
                f"Error: No config found. Provide --config or place classify.yaml next to {source}",
                err=True,
            )
            raise typer.Exit(1)

    cfg = ClassifyConfig.from_yaml(config)
    if concurrency is not None:
        cfg.concurrency = concurrency
    cfg.validate()
    runner_retry(cfg, retry_from=source, max_rounds=max_rounds, dry_run=dry_run)


@app.command()
def init(
    output: Path = typer.Option(Path("classify.yaml"), "--output", "-o", help="Output path for config file"),
) -> None:
    """Generate a starter YAML config template."""
    if output.exists():
        typer.echo(f"Error: {output} already exists. Use a different path.", err=True)
        raise typer.Exit(1)

    template = '''# LLM Batch Classifier Configuration
# Documentation: https://github.com/your-org/llm-batch-classifier
#
# API keys are NOT stored in this file.
# Set one of these environment variables before running:
#   export LLM_API_KEY=your-key
#   export OPENAI_API_KEY=your-key

# List every label the model is allowed to return.
# Use one label per line, and keep the names exactly as you want them in the output.
categories:
  - "Category A"
  - "Category B"
  - "Category C"

# Prompt settings used for every row.
prompt:
  # Main system prompt for the classification task.
  # This text must contain the {categories} placeholder so the label list is injected automatically.
  system: |
    You are a classification expert. Classify the input into these categories:
    {categories}

    Requirements:
    1. Select all matching categories with confidence scores (0-100)
    2. Only include categories with confidence >= 85
    3. Use exact category names from the list above
    4. Output JSON only

    Output format:
    {{"labels": [{{"name": "Category Name", "confidence": 95, "reason": "classification reason"}}]}}

  # Optional alternative to prompt.system.
  # Uncomment this and remove prompt.system if you want to keep a long prompt in a separate text file.
  # system_file: prompt.txt

  # Row-level prompt template built from your input columns.
  # Only {text} and {context} are supported placeholders.
  # If you do not have a context column, change this to "{text}".
  user: "{text} / {context}"

# Model and API endpoint settings.
model:
  # Model identifier understood by your provider.
  name: deepseek-chat

  # Base URL of an OpenAI-compatible API.
  # Usually this ends at /v1 and should not include /chat/completions.
  api_base: https://api.deepseek.com/v1

  # Sampling temperature. Lower values are usually more stable for classification.
  temperature: 0.1

  # Maximum number of output tokens the model may generate for one row.
  max_tokens: 500

  # Per-request timeout in seconds.
  timeout: 30

  # Number of retries for transient request failures within a single API call.
  max_retries: 3

# Sliding-window rate limits that protect you from sending requests too fast.
rate_limit:
  # Maximum requests per second. Set to 0 to disable request-based limiting.
  rps: 3

  # Maximum tokens per second. Set to 0 to disable token-based limiting.
  # This uses the tokens_per_call estimate below, not the exact provider-reported token count.
  tps: 0

  # Size of the sliding window in seconds.
  # Must be greater than 0 when rps or tps is enabled.
  window: 1

  # Estimated total tokens consumed by one request.
  # Only used when tps > 0. Increase this if your prompts or responses are large.
  tokens_per_call: 850

# Optional coarse-grained call budget across a longer period.
# Leave both fields at 0 to disable this feature.
cycle:
  # Length of one budget cycle in seconds.
  duration: 0

  # Maximum number of API calls allowed within one cycle.
  max_calls: 0

# Backoff behavior when the provider returns 429 or similar throttling errors.
throttle:
  # Maximum number of backoff attempts before giving up on a throttled item.
  max_attempts: 10

  # Initial wait time in seconds before the first retry after throttling.
  base_wait: 30.0

  # Upper bound in seconds for exponential backoff waits.
  max_wait: 300.0

  # Random jitter in seconds added to each wait so many retries do not synchronize.
  jitter: 0.5

# Input file and column mapping.
input:
  # Path to your source file. Supports CSV and Excel (.xlsx).
  file: data.csv

  # Column that contains the main text to classify.
  text_column: text

  # Optional column that provides extra context for the same row.
  # Leave this empty if you do not have a context column.
  context_column: context

# Output location and result file format.
output:
  # Directory where run folders, result files, and reports will be written.
  dir: output

  # Output file format.
  # Use auto to follow the input type, or force csv / xlsx explicitly.
  format: auto

# Minimum confidence required for a label to remain in the final output.
# Labels below this threshold are filtered out.
threshold: 95

# Number of requests allowed to be in flight at the same time.
# Lower this first if you see many 429 errors.
concurrency: 15
'''
    output.write_text(template, encoding="utf-8")
    typer.echo(f"Created starter config: {output}")
    typer.echo("Next steps:")
    typer.echo("  1. Edit the config with your categories and prompt")
    typer.echo("  2. Set your API key: export LLM_API_KEY=your-key")
    typer.echo(f"  3. Run: llm-classify run --config {output}")
