"""Classification report and run summary generation."""

from __future__ import annotations

import collections
import json
from datetime import datetime
from pathlib import Path

from .api import RunStats
from .config import ClassifyConfig
from .logging_utils import log
from .validation import Validator


def generate_report(results: list[dict], report_path: Path, config: ClassifyConfig) -> None:
    """Generate a markdown classification report.

    Args:
        results: List of result row dicts (English field names).
        report_path: Path to write the Markdown report file.
        config: ClassifyConfig providing threshold, text_column, context_column.
    """
    log.info("Generating classification report...")
    total = len(results)
    if total == 0:
        report_path.write_text("# Classification Report\n\nNo data.\n", encoding="utf-8")
        return

    threshold = config.threshold
    validator = Validator.from_config(config)

    # Single pass to collect all metrics
    success = 0
    low_conf = 0
    failed = 0
    fuzzy_corrected = 0
    post_validated = 0
    label_counts: dict[str, int] = collections.Counter()
    label_count_dist: dict[int, int] = collections.Counter()
    conf_buckets = {">=95": 0, "90-94": 0, "80-89": 0, "50-79": 0, "<50": 0}

    for r in results:
        valid_labels = validator.extract_valid(r.get("label", ""))
        if valid_labels:
            success += 1
        else:
            failed += 1

        if r.get("is_low_confidence") == "yes":
            low_conf += 1

        status = str(r.get("parse_status", ""))
        if "fuzzy_corrected" in status:
            fuzzy_corrected += 1
        if "post_validate" in status:
            post_validated += 1

        for lbl in valid_labels:
            label_counts[lbl] += 1
        label_count_dist[len(valid_labels)] += 1

        try:
            c = int(float(r.get("confidence", 0)))
        except (ValueError, TypeError):
            c = 0
        if c >= 95:
            conf_buckets[">=95"] += 1
        elif c >= 90:
            conf_buckets["90-94"] += 1
        elif c >= 80:
            conf_buckets["80-89"] += 1
        elif c >= 50:
            conf_buckets["50-79"] += 1
        else:
            conf_buckets["<50"] += 1

    text_col = config.text_column
    ctx_col = config.context_column or "context"

    lines = [
        "# Classification Report", "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
        "## Summary", "",
        "| Metric | Value |", "|--------|-------|",
        f"| Unique items (deduplicated) | {total} |",
        f"| Classified (valid labels) | {success} ({success / total * 100:.1f}%) |",
        f"| Low confidence (max < {threshold}) | {low_conf} ({low_conf / total * 100:.1f}%) |",
        f"| Failed / unclassified | {failed} ({failed / total * 100:.1f}%) |",
        f"| Fuzzy corrections | {fuzzy_corrected} |",
        f"| Post-validation fixes | {post_validated} |", "",
        "## Confidence Distribution", "", "| Range | Count | % |", "|-------|-------|---|",
    ]
    for bucket, count in conf_buckets.items():
        lines.append(f"| {bucket} | {count} | {count / total * 100:.1f}% |")

    lines += ["", "## Multi-label Statistics", "", "| Label count | Items | % |", "|-------------|-------|---|"]
    for n in sorted(label_count_dist.keys()):
        count = label_count_dist[n]
        lines.append(f"| {n} | {count} | {count / total * 100:.1f}% |")

    lines += ["", "## Label Distribution (by deduplicated item)", "", "| Label | Items | % |", "|-------|-------|---|"]
    for lbl, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {lbl} | {count} | {count / total * 100:.1f}% |")

    # Low confidence list
    abnormal = [r for r in results if r.get("is_low_confidence") == "yes"]
    if abnormal:
        lines += [
            "", "## Low Confidence / Anomalous Items", "",
            f"| {text_col} | {ctx_col} | label | confidence | parse_status |",
            "|" + "---|" * 5,
        ]
        for r in sorted(abnormal, key=lambda x: int(float(x.get("confidence", 0) or 0))):
            lbl = str(r.get("label", ""))[:40].replace("|", "/")
            status = str(r.get("parse_status", ""))[:40].replace("|", "/")
            text_val = str(r.get(text_col, ""))[:60]
            ctx_val = str(r.get(ctx_col, ""))[:30]
            lines.append(
                f"| {text_val} | {ctx_val} | {lbl} | {r.get('confidence', '')} | {status} |"
            )

    # Fuzzy correction log
    fuzzy = [r for r in results if "fuzzy_corrected" in str(r.get("parse_status", ""))]
    if fuzzy:
        lines += ["", "## Fuzzy Correction Log", "", f"| {text_col} | parse_status |", "|---|---|"]
        for r in fuzzy:
            lines.append(f"| {str(r.get(text_col, ''))[:60]} | {r.get('parse_status', '')} |")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info(f"Report saved: {report_path}")


def generate_run_summary(
    results: list[dict],
    run_stats: RunStats,
    run_dir: Path,
    config: ClassifyConfig,
    duration_seconds: float,
) -> None:
    """Generate a machine-readable run_summary.json.

    Args:
        results: List of result row dicts.
        run_stats: Accumulated API call statistics.
        run_dir: Directory to write run_summary.json into.
        config: ClassifyConfig providing model_name, api_base, threshold.
        duration_seconds: Total wall-clock duration of the classification run.
    """
    validator = Validator.from_config(config)
    total = len(results)
    success = sum(1 for r in results if validator.extract_valid(r.get("label", "")))
    failed = total - success

    summary = {
        "generated_at": datetime.now().isoformat(),
        "model": config.model_name,
        "api_base_url": config.api_base,
        "threshold": config.threshold,
        "total": total,
        "success": success,
        "failed": failed,
        "duration_seconds": round(duration_seconds, 2),
        "tokens": {
            "prompt_tokens": run_stats.total_prompt_tokens,
            "completion_tokens": run_stats.total_completion_tokens,
            "total_tokens": run_stats.total_tokens,
        },
        "retries": {
            "total_retries": run_stats.total_retries,
            "total_429s": run_stats.total_429s,
            "total_timeouts": run_stats.total_timeouts,
            "total_other_errors": run_stats.total_other_errors,
            "total_backoff_seconds": round(run_stats.total_backoff_seconds, 2),
            "max_retries_program": run_stats.max_retries_program,
            "max_retries_count": run_stats.max_retries_count,
        },
    }

    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"Run summary saved: {summary_path}")

    if run_stats.total_tokens > 0:
        log.info(
            f"Token usage: prompt={run_stats.total_prompt_tokens}, "
            f"completion={run_stats.total_completion_tokens}, "
            f"total={run_stats.total_tokens}"
        )
    if run_stats.total_retries > 0:
        log.info(
            f"Retry stats: total_retries={run_stats.total_retries}, "
            f"429s={run_stats.total_429s}, timeouts={run_stats.total_timeouts}, "
            f"other_errors={run_stats.total_other_errors}, "
            f"backoff_total={run_stats.total_backoff_seconds:.1f}s"
        )
    if run_stats.max_retries_program:
        log.info(
            f"Most retried: {run_stats.max_retries_program} ({run_stats.max_retries_count} retries)"
        )
