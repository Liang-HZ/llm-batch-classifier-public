"""Main orchestration: run() and run_retry() entry points."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from .api import RunStats
from .config import ClassifyConfig
from .data_io import (
    deduplicate_input_rows,
    load_existing_results,
    load_and_split_results,
    sample_programs,
    read_input_file,
    resolve_output_format,
    write_dataframe,
)
from .identity import RowKey, build_row_key
from .logging_utils import log, add_file_log
from .pipeline import run_classification
from .rate_limiter import SlidingWindowRateLimiter, CycleRateLimiter, _format_duration
from .report import generate_report, generate_run_summary
from .validation import (
    Validator,
    is_retryable,
    categorize_failure,
    RESULT_FIELDNAMES,
)


# ============================================================
# Shared helpers
# ============================================================


def _build_limiters(
    config: ClassifyConfig, log_dir: Path, output_dir: Path, *, verbose: bool = True,
) -> tuple[SlidingWindowRateLimiter | None, CycleRateLimiter | None]:
    """Build rate limiters from config. Returns (window_limiter, cycle_limiter)."""
    window_limiter: SlidingWindowRateLimiter | None = None
    if config.rate_limit_rps > 0 or config.rate_limit_tps > 0:
        max_requests = int(config.rate_limit_rps * config.rate_limit_window) if config.rate_limit_rps > 0 else 0
        max_tokens = int(config.rate_limit_tps * config.rate_limit_window) if config.rate_limit_tps > 0 else 0
        window_limiter = SlidingWindowRateLimiter(
            max_requests=max_requests,
            max_tokens=max_tokens,
            tokens_per_request=config.tokens_per_call,
            window_seconds=config.rate_limit_window,
        )
        if verbose:
            log.info(
                f"Sliding-window rate limiter enabled: "
                f"RPS={config.rate_limit_rps}, TPS={config.rate_limit_tps}, "
                f"window={config.rate_limit_window}s, tokens_per_call={config.tokens_per_call}"
            )

    cycle_limiter: CycleRateLimiter | None = None
    if config.cycle_duration > 0 and config.cycle_max_calls > 0:
        cycle_db_path = output_dir / ".cycle_rate_limit.sqlite3"
        cycle_limiter = CycleRateLimiter(
            config.cycle_duration,
            config.cycle_max_calls,
            log_dir=log_dir,
            db_path=cycle_db_path,
        )
        if verbose:
            log.info(
                f"Cycle rate limiter enabled: every {_format_duration(config.cycle_duration)} "
                f"max {config.cycle_max_calls} reservations (db: {cycle_db_path})"
            )

    return window_limiter, cycle_limiter


def _log_dedup_summary(original_count: int, deduped_count: int, context_column: str) -> None:
    """Log how many duplicate input rows were removed."""
    duplicates = original_count - deduped_count
    if duplicates <= 0:
        log.info("Input deduplication: no duplicate rows found")
        return
    key_desc = "text + context" if context_column else "text"
    log.info(f"Input deduplication: removed {duplicates} duplicate row(s) using {key_desc} identity")


# ============================================================
# Main entry point
# ============================================================


def run(
    config: ClassifyConfig,
    test: int | None = None,
    random: int | None = None,
    input_csv: str | Path | None = None,
    append: bool = False,
    fresh: bool = False,
    dry_run: bool = False,
) -> None:
    """Main entry point for a classification run.

    Args:
        config: ClassifyConfig with all runtime settings.
        test: Test mode — take the first *test* rows (deterministic).
        random: Random mode — randomly sample *random* rows.
        input_csv: Path to an existing CSV to use as input (comparison mode).
        append: Accumulate mode — checkpoint-resume against a fixed output file.
        fresh: Clear accumulated append-mode results before running.
        dry_run: Print config/workload summary without calling the API.
    """
    # Log startup args
    args_parts: list[str] = []
    if test is not None:
        args_parts.append(f"--test {test}")
    if random is not None:
        args_parts.append(f"--random {random}")
    if input_csv:
        args_parts.append(f"--input-csv {input_csv}")
    if config.concurrency is not None:
        args_parts.append(f"--concurrency {config.concurrency}")
    if config.threshold != 95:
        args_parts.append(f"--threshold {config.threshold}")
    if config.input_file:
        args_parts.append(f"--input {config.input_file}")
    if config.output_dir != "output":
        args_parts.append(f"--output-dir {config.output_dir}")
    if append:
        args_parts.append("--append")
    if fresh:
        args_parts.append("--fresh")
    if dry_run:
        args_parts.append("--dry-run")
    log.info(f"Run args: {' '.join(args_parts) if args_parts else '(default full run)'}")

    # Mutual exclusion checks
    if input_csv and (test is not None or random is not None):
        log.error("--input-csv cannot be combined with --test or --random")
        sys.exit(1)
    if input_csv and append:
        log.error("--input-csv cannot be combined with --append (comparison runs need independent results)")
        sys.exit(1)

    output_dir = Path(config.output_dir)

    # Config validation (skip API key check in dry-run)
    if not dry_run:
        config.validate()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 0: load input data (auto-detects CSV vs Excel)
    old_classifications: dict[RowKey, str] = {}
    effective_input_path: Path | None = None  # track for output format auto-detection
    text_col = config.text_column
    ctx_col = config.context_column
    dedup_cols = [text_col] + ([ctx_col] if ctx_col else [])

    if input_csv:
        input_csv_path = Path(input_csv)
        effective_input_path = input_csv_path
        input_df = read_input_file(input_csv_path)
        required = [text_col] + ([ctx_col] if ctx_col else [])
        missing = [c for c in required if c not in input_df.columns]
        if missing:
            log.error(
                f"Input file must contain columns {required}; missing: {missing}; "
                f"found: {list(input_df.columns)}"
            )
            sys.exit(1)
        deduped_input = deduplicate_input_rows(input_df, text_col, ctx_col)
        _log_dedup_summary(len(input_df), len(deduped_input), ctx_col)
        if "label" in deduped_input.columns:
            for _, row in deduped_input.iterrows():
                old_val = row.get("label", "")
                old_classifications[build_row_key(row, text_col, ctx_col)] = (
                    "" if pd.isna(old_val) else str(old_val)
                )
            log.info("Found existing 'label' column — comparison report will be generated")
        unique_df = deduped_input[dedup_cols].reset_index(drop=True)
        log.info(f"Loaded {len(unique_df)} unique items")
    else:
        if not config.input_file:
            log.error("config.input_file must be set when not using --input-csv")
            sys.exit(1)
        input_path = Path(config.input_file)
        effective_input_path = input_path

        if not input_path.exists():
            log.error(f"Input file not found: {input_path}")
            sys.exit(1)
        input_df = read_input_file(input_path)
        required = [text_col] + ([ctx_col] if ctx_col else [])
        missing = [c for c in required if c not in input_df.columns]
        if missing:
            log.error(
                f"Input file must contain columns {required}; missing: {missing}; "
                f"found: {list(input_df.columns)}"
            )
            sys.exit(1)
        deduped_input = deduplicate_input_rows(input_df, text_col, ctx_col)
        _log_dedup_summary(len(input_df), len(deduped_input), ctx_col)
        unique_df = sample_programs(
            deduped_input[dedup_cols].reset_index(drop=True),
            test_n=test,
            random_n=random,
        )

    # Resolve output format: auto follows input, or user override
    out_fmt = resolve_output_format(effective_input_path, config.output_format)
    log.info(f"Output format: {out_fmt}")

    # Determine effective concurrency
    concurrency = config.concurrency if config.concurrency is not None else 15
    if config.concurrency is None:
        log.info(f"Using default concurrency: {concurrency} (rate controlled by sliding-window limiter)")

    # Dry-run: summary and exit
    if dry_run:
        log.info("=== DRY RUN ===")
        log.info(f"Model: {config.model_name} | API: {config.api_base}")
        log.info(f"Items to process: {len(unique_df)}")
        log.info(f"Concurrency: {concurrency} | Confidence threshold: {config.threshold}")
        log.info(
            f"Sliding-window rate limit: "
            f"RPS={config.rate_limit_rps}, TPS={config.rate_limit_tps}, "
            f"window={config.rate_limit_window}s"
        )
        if config.cycle_duration > 0:
            log.info(
                f"Cycle rate limit: every {_format_duration(config.cycle_duration)} "
                f"max {config.cycle_max_calls} calls"
            )
        estimated_tokens = len(unique_df) * config.tokens_per_call
        log.info(f"Estimated API calls: {len(unique_df)}, estimated tokens: {estimated_tokens:,}")
        if append:
            result_csv = output_dir / "classification_result_accumulated.csv"
            existing = load_existing_results(result_csv, config.text_column, config.context_column)
            remaining = len(unique_df) - len(existing)
            log.info(f"Append mode: {len(existing)} existing successes, {remaining} remaining")
        log.info("=== END DRY RUN (no API calls made) ===")
        return

    # Create run directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if input_csv:
        mode_tag = f"_rerun_{Path(input_csv).stem}"
    elif test:
        mode_tag = f"_test_first{test}"
    elif random:
        mode_tag = f"_test_random{random}"
    else:
        mode_tag = "_full"
    run_dir = output_dir / f"run_{ts}{mode_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_dir = run_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    main_log_handler = add_file_log(run_dir / "run.log")

    log.info(f"Run directory: {run_dir}")
    log.info(f"Model: {config.model_name} | API: {config.api_base}")

    try:
        _run_core(
            config=config,
            run_dir=run_dir,
            log_dir=log_dir,
            output_dir=output_dir,
            unique_df=unique_df,
            concurrency=concurrency,
            append=append,
            fresh=fresh,
            input_csv=input_csv,
            old_classifications=old_classifications,
            out_fmt=out_fmt,
        )
    finally:
        log.removeHandler(main_log_handler)
        main_log_handler.close()


def _run_core(
    config: ClassifyConfig,
    run_dir: Path,
    log_dir: Path,
    output_dir: Path,
    unique_df: pd.DataFrame,
    concurrency: int,
    append: bool,
    fresh: bool,
    input_csv: str | Path | None,
    old_classifications: dict[RowKey, str],
    out_fmt: str = "csv",
) -> None:
    """Core classification logic, wrapped by run() for log-handler cleanup."""
    text_col = config.text_column
    ctx_col = config.context_column

    # Checkpoint / accumulate mode
    if append:
        result_csv = output_dir / "classification_result_accumulated.csv"
        meta_path = output_dir / ".classify_meta.json"
        log.info("Append mode: writing to accumulated file (checkpoint-resume enabled)")

        if fresh:
            for f in [result_csv, meta_path]:
                if f.exists():
                    f.unlink()
                    log.info(f"Deleted: {f}")

        prompt_hash = hashlib.md5(config.system_prompt.encode()).hexdigest()[:8]
        current_meta = {
            "threshold": config.threshold,
            "concurrency": concurrency,
            "model": config.model_name,
            "base_url": config.api_base,
            "prompt_hash": prompt_hash,
        }

        existing = load_existing_results(result_csv, text_col, ctx_col)
        if existing and meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
            for key, label in [
                ("threshold", "threshold"),
                ("model", "model"),
                ("base_url", "api_base"),
                ("prompt_hash", "prompt"),
            ]:
                old_val = meta.get(key)
                new_val = current_meta.get(key)
                if old_val is not None and old_val != new_val:
                    log.warning(
                        f"Previous run used {label}={old_val}, current {label}={new_val}. "
                        f"Use --fresh to restart from scratch."
                    )
        with open(meta_path, "w") as f:
            json.dump(current_meta, f, ensure_ascii=False)
    else:
        result_csv = run_dir / "classification_result.csv"
        existing = {}

    # Save unique items snapshot
    unique_csv = run_dir / "unique_items.csv"
    unique_df.to_csv(unique_csv, index=False, encoding="utf-8-sig")
    log.info(f"Items snapshot saved: {unique_csv}")

    # Build rate limiters
    window_limiter, cycle_limiter = _build_limiters(config, log_dir, output_dir)

    # Graceful shutdown
    shutdown_event = asyncio.Event()

    def _signal_handler(signum, frame):
        if not shutdown_event.is_set():
            log.warning("Interrupt received — graceful shutdown in progress (waiting for in-flight requests)...")
            shutdown_event.set()
        else:
            log.warning("Second interrupt received — forcing exit")
            sys.exit(1)

    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _signal_handler)

    run_stats = RunStats()
    run_start_time = time.time()

    jitter_info = f", jitter: 0–{config.jitter_seconds:.1f}s" if config.jitter_seconds > 0 else ""
    log.info(f"Starting classification (concurrency: {concurrency}, threshold: {config.threshold}{jitter_info})")

    try:
        results = asyncio.run(
            run_classification(
                config=config,
                unique_df=unique_df,
                existing=existing,
                result_path=result_csv,
                window_limiter=window_limiter,
                cycle_limiter=cycle_limiter,
                run_stats=run_stats,
                shutdown_event=shutdown_event,
            )
        )
    finally:
        signal.signal(signal.SIGINT, original_sigint)
        if cycle_limiter:
            cycle_limiter.close()

    duration = time.time() - run_start_time
    log.info(f"Classification done: {len(results)} results in {duration:.1f}s")

    # Post-validation
    validator = Validator.from_config(config)
    results = validator.post_validate(results, run_dir)

    # Build fieldnames for final CSV
    if ctx_col:
        base_fields = [text_col, ctx_col] + RESULT_FIELDNAMES
    else:
        base_fields = [text_col] + RESULT_FIELDNAMES

    # Write final results
    final_csv = run_dir / "classification_result.csv"

    if old_classifications:
        compare_fields = base_fields + ["compare_old_label", "compare_is_match"]
        match_count = 0
        for r in results:
            old = old_classifications.get(build_row_key(r, text_col, ctx_col), "")
            new = r.get("label", "")
            r["compare_old_label"] = old
            is_match = set(old.split("|")) == set(new.split("|")) if old and new else old == new
            r["compare_is_match"] = "yes" if is_match else "no"
            if is_match:
                match_count += 1

        with open(final_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=compare_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

        total = len(results)
        diff_count = total - match_count
        pct = match_count / total * 100 if total > 0 else 0
        log.info(f"Comparison: {match_count}/{total} match ({pct:.1f}%), {diff_count} differ")

        if diff_count > 0:
            diff_csv = run_dir / "classification_diff.csv"
            diff_rows = [r for r in results if r["compare_is_match"] == "no"]
            diff_fields = [text_col] + ([ctx_col] if ctx_col else []) + [
                "compare_old_label", "label", "confidence", "parse_status"
            ]
            with open(diff_csv, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=diff_fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(diff_rows)
            log.info(f"Diff file saved: {diff_csv} ({diff_count} rows)")
    else:
        with open(final_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=base_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

    log.info(f"Final results: {final_csv}")

    # Append mode: rewrite accumulated file with post-validated data
    if append:
        current_keys = {build_row_key(r, text_col, ctx_col) for r in results}
        preserved = [row for k, row in existing.items() if k not in current_keys]
        if preserved:
            log.info(f"Preserving {len(preserved)} historical records not in this run")
        all_records = results + preserved
        with open(result_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=base_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_records)
        log.info(f"Accumulated file updated: {result_csv} ({len(all_records)} total records)")

    # Additional xlsx output (after append merge so it includes all accumulated data)
    if out_fmt == "xlsx":
        final_xlsx = final_csv.with_suffix(".xlsx")
        xlsx_data = all_records if append else results
        result_df = pd.DataFrame(xlsx_data)
        cols = (compare_fields if old_classifications else base_fields)
        available_cols = [c for c in cols if c in result_df.columns]
        write_dataframe(result_df[available_cols], final_xlsx, fmt="xlsx")

    # Report and summary
    report_md = run_dir / "classification_report.md"
    generate_report(results, report_md, config)
    generate_run_summary(results, run_stats, run_dir, config, duration)

    log.info(f"All done! Report directory: {run_dir}")


# ============================================================
# Auto-retry mode
# ============================================================


def run_retry(
    config: ClassifyConfig,
    retry_from: str | Path,
    max_rounds: int = 3,
    dry_run: bool = False,
) -> None:
    """Auto-retry mode: load a results CSV, identify failures, re-run until convergence.

    Convergence stops when any of these conditions is met:
    - No retryable failures remain.
    - Zero new successes this round (plateau).
    - max_rounds reached.

    Failure categories:
    - transient: API timeouts, 429s, processing errors — retried every round.
    - semantic: parse errors, unclassified, unmatched labels — skipped after
      >= 2 consecutive semantic failures for the same item.

    Args:
        config: ClassifyConfig with model and runtime settings.
        retry_from: Path to the results CSV to retry from.
        max_rounds: Maximum number of retry rounds (default 3).
        dry_run: Preview retry scope without calling the API.
    """
    retry_path = Path(retry_from)

    if not retry_path.exists():
        log.error(f"Results file not found: {retry_path}")
        sys.exit(1)

    log.info(f"Auto-retry mode: {retry_path}")
    log.info(f"Max rounds: {max_rounds} | Confidence threshold: {config.threshold}")

    # Load and split into successes / failures
    successes, failures = load_and_split_results(retry_path, config.text_column)

    if not failures:
        log.info("No failed records — nothing to retry.")
        return

    type_counts = _count_failure_types(failures)

    if dry_run:
        log.info("=== RETRY DRY RUN ===")
        log.info(f"Results file: {retry_path}")
        log.info(
            f"Total: {len(successes) + len(failures)} | "
            f"Success: {len(successes)} | Failed: {len(failures)}"
        )
        _log_failure_breakdown(type_counts, prefix="  ")
        log.info(f"Estimated re-runs: {len(failures)} rows | max {max_rounds} round(s)")
        log.info("=== END DRY RUN (no API calls made) ===")
        return

    config.validate()

    output_dir = retry_path.parent
    concurrency = config.concurrency if config.concurrency is not None else 15
    text_col = config.text_column
    ctx_col = config.context_column

    # Track consecutive semantic failures per item identity.
    semantic_fail_streak: dict[RowKey, int] = {}

    for round_num in range(1, max_rounds + 1):
        retryable = _filter_retryable(failures, semantic_fail_streak, text_col, ctx_col)

        if not retryable:
            log.info(f"Round {round_num}: no retryable failures — stopping early.")
            break

        log.info("=" * 60)
        log.info(f"Round {round_num}/{max_rounds}: retrying {len(retryable)} failed records")
        _log_failure_breakdown(_count_failure_types(retryable), prefix="  This round: ")

        # Create per-round run directory
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / f"run_{ts}_retry_round{round_num}"
        run_dir.mkdir(parents=True, exist_ok=True)
        log_dir = run_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        main_log_handler = add_file_log(run_dir / "run.log")

        try:
            new_results = _run_retry_round(
                config=config,
                retryable=retryable,
                run_dir=run_dir,
                log_dir=log_dir,
                output_dir=output_dir,
                concurrency=concurrency,
            )
        finally:
            log.removeHandler(main_log_handler)
            main_log_handler.close()

        # Count outcomes
        new_successes = [r for r in new_results if not is_retryable(r)]
        still_failed = [r for r in new_results if is_retryable(r)]

        # Update semantic streak counters
        _update_semantic_fail_streak(
            semantic_fail_streak,
            new_successes,
            still_failed,
            text_col,
            ctx_col,
        )

        # Merge results
        new_success_keys = {build_row_key(r, text_col, ctx_col) for r in new_successes}
        failures = [r for r in failures if build_row_key(r, text_col, ctx_col) not in new_success_keys]
        still_failed_keys = {build_row_key(r, text_col, ctx_col) for r in still_failed}
        failures = [r for r in failures if build_row_key(r, text_col, ctx_col) not in still_failed_keys] + still_failed
        successes = successes + new_successes

        # Write back to original CSV
        _write_back_results(retry_path, successes, failures, config)

        # Convergence summary
        new_type_counts = _count_failure_types(failures)
        log.info(f"Round {round_num}/{max_rounds} complete:")
        log.info(
            f"  {len(retryable)} retried → {len(new_successes)} recovered, {len(still_failed)} still failed"
        )
        if type_counts and new_type_counts:
            _log_convergence(type_counts, new_type_counts)
        type_counts = new_type_counts

        if not failures:
            log.info("All failures recovered!")
            break
        if len(new_successes) == 0:
            log.info("Zero new successes this round — plateau reached, stopping.")
            break

    # Final summary
    log.info("=" * 60)
    log.info(f"Retry complete: {len(successes)} successful, {len(failures)} failed")
    if failures:
        log.info("Remaining failure breakdown:")
        _log_failure_breakdown(_count_failure_types(failures), prefix="  ")
        exhausted = [
            r for r in failures
            if categorize_failure(r) == "semantic"
            and semantic_fail_streak.get(build_row_key(r, text_col, ctx_col), 0) >= 2
        ]
        if exhausted:
            log.info(
                f"{len(exhausted)} semantic failure(s) have reached the retry limit "
                f"(model cannot process these inputs)"
            )


def _run_retry_round(
    config: ClassifyConfig,
    retryable: list[dict],
    run_dir: Path,
    log_dir: Path,
    output_dir: Path,
    concurrency: int,
) -> list[dict]:
    """Execute one retry round. Returns result rows for all retried items."""
    text_col = config.text_column
    ctx_col = config.context_column

    # Build DataFrame from retryable rows
    rows = []
    for r in retryable:
        item: dict = {text_col: r[text_col]}
        if ctx_col:
            item[ctx_col] = r.get(ctx_col, "")
        rows.append(item)
    unique_df = deduplicate_input_rows(pd.DataFrame(rows), text_col, ctx_col)

    # Build rate limiters
    window_limiter, cycle_limiter = _build_limiters(config, log_dir, output_dir, verbose=False)

    shutdown_event = asyncio.Event()

    def _signal_handler(signum, frame):
        if not shutdown_event.is_set():
            log.warning("Interrupt received — graceful shutdown in progress...")
            shutdown_event.set()
        else:
            log.warning("Second interrupt — forcing exit")
            sys.exit(1)

    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _signal_handler)

    run_stats = RunStats()
    run_start_time = time.time()

    jitter_info = f", jitter: 0–{config.jitter_seconds:.1f}s" if config.jitter_seconds > 0 else ""
    log.info(f"Model: {config.model_name} | API: {config.api_base}")
    log.info(f"Starting retry round (concurrency: {concurrency}{jitter_info})")

    try:
        results = asyncio.run(
            run_classification(
                config=config,
                unique_df=unique_df,
                existing={},
                result_path=run_dir / "classification_result.csv",
                window_limiter=window_limiter,
                cycle_limiter=cycle_limiter,
                run_stats=run_stats,
                shutdown_event=shutdown_event,
            )
        )
    finally:
        signal.signal(signal.SIGINT, original_sigint)
        if cycle_limiter:
            cycle_limiter.close()

    duration = time.time() - run_start_time

    # Post-validation
    validator = Validator.from_config(config)
    results = validator.post_validate(results, run_dir)

    # Write round's final result file
    if ctx_col:
        fieldnames = [text_col, ctx_col] + RESULT_FIELDNAMES
    else:
        fieldnames = [text_col] + RESULT_FIELDNAMES

    final_csv = run_dir / "classification_result.csv"
    with open(final_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    # Report and summary
    report_md = run_dir / "classification_report.md"
    generate_report(results, report_md, config)
    generate_run_summary(results, run_stats, run_dir, config, duration)

    return results


def _write_back_results(
    result_path: Path,
    successes: list[dict],
    failures: list[dict],
    config: ClassifyConfig,
) -> None:
    """Write merged results back to *result_path* atomically via .tmp + rename."""
    text_col = config.text_column
    ctx_col = config.context_column
    if ctx_col:
        fieldnames = [text_col, ctx_col] + RESULT_FIELDNAMES
    else:
        fieldnames = [text_col] + RESULT_FIELDNAMES

    all_records = successes + failures
    tmp_path = result_path.with_suffix(".csv.tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_records)
    tmp_path.replace(result_path)
    log.info(f"Results written back: {result_path} ({len(all_records)} records)")


def _filter_retryable(
    failures: list[dict],
    semantic_fail_streak: dict[RowKey, int],
    text_column: str,
    context_column: str = "",
) -> list[dict]:
    """Return failures eligible for retry this round.

    Skips items with >= 2 consecutive semantic failures (model cannot handle them).
    """
    retryable = []
    skipped_exhausted = 0
    for r in failures:
        key = build_row_key(r, text_column, context_column)
        if categorize_failure(r) == "semantic" and semantic_fail_streak.get(key, 0) >= 2:
            skipped_exhausted += 1
            continue
        retryable.append(r)
    if skipped_exhausted:
        log.info(f"Skipping {skipped_exhausted} item(s) that reached the semantic retry limit")
    return retryable


def _update_semantic_fail_streak(
    semantic_fail_streak: dict[RowKey, int],
    new_successes: list[dict],
    still_failed: list[dict],
    text_column: str,
    context_column: str = "",
) -> None:
    """Update consecutive semantic-failure streaks after a retry round."""
    for row in new_successes:
        semantic_fail_streak.pop(build_row_key(row, text_column, context_column), None)

    for row in still_failed:
        key = build_row_key(row, text_column, context_column)
        if categorize_failure(row) == "semantic":
            semantic_fail_streak[key] = semantic_fail_streak.get(key, 0) + 1
        else:
            semantic_fail_streak.pop(key, None)


def _count_failure_types(failures: list[dict]) -> dict[str, int]:
    """Return a {failure_type: count} breakdown for a list of failure rows."""
    counts: dict[str, int] = {}
    for r in failures:
        label = str(r.get("label", ""))
        if label in {"api_error", "processing_error"}:
            key = label
        elif label == "parse_error":
            key = "parse_error"
        elif label == "unclassified":
            key = "unclassified"
        elif any(s.strip().startswith("unmatched_") for s in label.split("|")):
            key = "unmatched"
        else:
            key = "other"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _log_failure_breakdown(counts: dict[str, int], prefix: str = "") -> None:
    """Log a human-readable failure type breakdown."""
    parts = [f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])]
    log.info(f"{prefix}{' | '.join(parts)}")


def _log_convergence(before: dict[str, int], after: dict[str, int]) -> None:
    """Log a convergence comparison between two failure breakdowns."""
    all_keys = sorted(set(before) | set(after))
    parts = []
    for k in all_keys:
        b = before.get(k, 0)
        a = after.get(k, 0)
        if b != a:
            parts.append(f"{k}: {b} → {a}")
    if parts:
        log.info(f"  Convergence: {' | '.join(parts)}")
