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

categories:
  - "Category A"
  - "Category B"
  - "Category C"

prompt:
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
  user: "{text} / {context}"

model:
  name: deepseek-chat
  api_base: https://api.deepseek.com/v1
  temperature: 0.1
  max_tokens: 500
  timeout: 30
  max_retries: 3

rate_limit:
  rps: 3          # Requests per second (0 = unlimited)
  tps: 0          # Tokens per second (0 = unlimited)
  window: 1       # Sliding window size in seconds

input:
  file: data.csv            # Supports both CSV and Excel (.xlsx)
  text_column: text
  context_column: context   # Optional, leave empty if not needed

output:
  dir: output
  format: auto              # auto (match input) | csv | xlsx

threshold: 95
concurrency: 15
'''
    output.write_text(template, encoding="utf-8")
    typer.echo(f"Created starter config: {output}")
    typer.echo("Next steps:")
    typer.echo("  1. Edit the config with your categories and prompt")
    typer.echo("  2. Set your API key: export LLM_API_KEY=your-key")
    typer.echo(f"  3. Run: llm-classify run --config {output}")
