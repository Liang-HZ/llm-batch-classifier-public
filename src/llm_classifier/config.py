"""YAML-driven configuration for llm-batch-classifier.

This module is intentionally self-contained — it must NOT import anything
else from llm_classifier to avoid circular dependencies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


class ConfigError(Exception):
    """Raised when the configuration is invalid or incomplete."""


@dataclass
class ClassifyConfig:
    # ------------------------------------------------------------------
    # Categories
    # ------------------------------------------------------------------
    categories: list[str]
    category_set: set[str] = field(default_factory=set, init=False)

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------
    prompt_template: str = ""
    user_prompt_template: str = "{text} / {context}"

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_name: str = ""
    api_base: str = ""
    api_key: str = ""
    temperature: float = 0.1
    max_tokens: int = 500
    timeout: int = 30
    max_retries: int = 3

    # ------------------------------------------------------------------
    # Throttle (429 backoff)
    # ------------------------------------------------------------------
    throttle_max_attempts: int = 10
    throttle_base_wait: float = 30.0
    throttle_max_wait: float = 300.0
    jitter_seconds: float = 0.5

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------
    rate_limit_rps: float = 0.0
    rate_limit_tps: float = 0.0
    rate_limit_window: float = 1.0
    tokens_per_call: int = 850

    # ------------------------------------------------------------------
    # Cycle limiting
    # ------------------------------------------------------------------
    cycle_duration: float = 0.0
    cycle_max_calls: int = 0

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    input_file: str = ""
    text_column: str = "text"
    context_column: str = ""

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    output_dir: str = "output"
    output_format: str = "auto"  # "auto" | "csv" | "xlsx"

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------
    threshold: int = 95
    concurrency: Optional[int] = None

    def __post_init__(self) -> None:
        # Derive category_set from categories for O(1) lookup.
        # NOTE: category_set is computed once at construction time. Do not
        # mutate cfg.categories after construction, as category_set will not
        # reflect the change.
        self.category_set = set(self.categories)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def system_prompt(self) -> str:
        """Build system prompt by injecting categories into prompt_template."""
        return self.prompt_template.format(
            categories="\n".join(f"- {c}" for c in self.categories)
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Path) -> "ClassifyConfig":
        """Load config from a YAML file.

        The API key is never stored in YAML.  It is read from the
        environment in this order:
          1. LLM_API_KEY
          2. OPENAI_API_KEY
        """
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

        categories: list[str] = raw.get("categories", [])

        # ---- prompt ------------------------------------------------
        prompt_section = raw.get("prompt", {})
        system_text: str = prompt_section.get("system", "")
        system_file: str = prompt_section.get("system_file", "")
        if system_file:
            # Resolve relative to the YAML file's directory.
            system_path = Path(path).parent / system_file
            system_text = system_path.read_text(encoding="utf-8")
        user_text: str = prompt_section.get("user", "{text} / {context}")

        # ---- model -------------------------------------------------
        model_section = raw.get("model", {})
        model_name: str = model_section.get("name", "")
        api_base: str = model_section.get("api_base", "")
        temperature: float = float(model_section.get("temperature", 0.1))
        max_tokens: int = int(model_section.get("max_tokens", 500))
        timeout: int = int(model_section.get("timeout", 30))
        max_retries: int = int(model_section.get("max_retries", 3))

        # ---- throttle ----------------------------------------------
        throttle_section = raw.get("throttle", {})
        throttle_max_attempts: int = int(throttle_section.get("max_attempts", 10))
        throttle_base_wait: float = float(throttle_section.get("base_wait", 30.0))
        throttle_max_wait: float = float(throttle_section.get("max_wait", 300.0))
        jitter_seconds: float = float(throttle_section.get("jitter", 0.5))

        # ---- rate limit --------------------------------------------
        rl_section = raw.get("rate_limit", {})
        rate_limit_rps: float = float(rl_section.get("rps", 0.0))
        rate_limit_tps: float = float(rl_section.get("tps", 0.0))
        rate_limit_window: float = float(rl_section.get("window", 1.0))
        tokens_per_call: int = int(rl_section.get("tokens_per_call", 850))

        # ---- cycle -------------------------------------------------
        cycle_section = raw.get("cycle", {})
        cycle_duration: float = float(cycle_section.get("duration", 0.0))
        cycle_max_calls: int = int(cycle_section.get("max_calls", 0))

        # ---- input -------------------------------------------------
        input_section = raw.get("input", {})
        input_file: str = str(input_section.get("file", ""))
        text_column: str = str(input_section.get("text_column", "text"))
        context_column: str = str(input_section.get("context_column", ""))

        # ---- output ------------------------------------------------
        output_section = raw.get("output", {})
        output_dir: str = str(output_section.get("dir", "output"))
        output_format: str = str(output_section.get("format", "auto")).lower()

        # ---- classification ----------------------------------------
        threshold: int = int(raw.get("threshold", 95))
        concurrency_raw = raw.get("concurrency", None)
        concurrency: Optional[int] = int(concurrency_raw) if concurrency_raw is not None else None

        # ---- API key (env-only, never from YAML) -------------------
        api_key: str = (
            os.environ.get("LLM_API_KEY", "")
            or os.environ.get("OPENAI_API_KEY", "")
        )

        cfg = cls(
            categories=categories,
            prompt_template=system_text,
            user_prompt_template=user_text,
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            throttle_max_attempts=throttle_max_attempts,
            throttle_base_wait=throttle_base_wait,
            throttle_max_wait=throttle_max_wait,
            jitter_seconds=jitter_seconds,
            rate_limit_rps=rate_limit_rps,
            rate_limit_tps=rate_limit_tps,
            rate_limit_window=rate_limit_window,
            tokens_per_call=tokens_per_call,
            cycle_duration=cycle_duration,
            cycle_max_calls=cycle_max_calls,
            input_file=input_file,
            text_column=text_column,
            context_column=context_column,
            output_dir=output_dir,
            output_format=output_format,
            threshold=threshold,
            concurrency=concurrency,
        )
        return cfg

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Validate config. Raises ConfigError listing all issues found."""
        errors: list[str] = []

        if not self.categories:
            errors.append("categories must not be empty")

        if not self.model_name:
            errors.append("model.name must not be empty")

        if not self.api_base:
            errors.append("model.api_base must not be empty")

        # rate_limit_window must be positive when limits are active
        if (self.rate_limit_rps > 0 or self.rate_limit_tps > 0) and self.rate_limit_window <= 0:
            errors.append(
                f"rate_limit.window={self.rate_limit_window} — must be > 0 when rate limits are enabled"
            )

        # tokens_per_call must not exceed the per-window token budget
        max_tokens_in_window = (
            int(self.rate_limit_tps * self.rate_limit_window)
            if self.rate_limit_tps > 0
            else 0
        )
        if max_tokens_in_window > 0 and self.tokens_per_call > max_tokens_in_window:
            errors.append(
                f"tokens_per_call ({self.tokens_per_call}) exceeds the per-window token budget "
                f"({max_tokens_in_window}); a single call would never pass token rate limiting"
            )

        # cycle settings must be both set or both zero
        if self.cycle_max_calls > 0 and self.cycle_duration <= 0:
            errors.append(
                f"cycle.max_calls={self.cycle_max_calls} but cycle.duration={self.cycle_duration} — "
                "both must be set together"
            )
        if self.cycle_duration > 0 and self.cycle_max_calls <= 0:
            errors.append(
                f"cycle.duration={self.cycle_duration} but cycle.max_calls={self.cycle_max_calls} — "
                "both must be set together"
            )

        if self.max_retries < 1:
            errors.append(f"model.max_retries={self.max_retries} — must be >= 1")

        if self.throttle_max_attempts < 1:
            errors.append(
                f"throttle.max_attempts={self.throttle_max_attempts} — must be >= 1"
            )

        if self.output_format not in ("auto", "csv", "xlsx"):
            errors.append(
                f"output.format='{self.output_format}' — must be 'auto', 'csv', or 'xlsx'"
            )

        if self.prompt_template and "{categories}" not in self.prompt_template:
            errors.append(
                "prompt.system template must contain the {categories} placeholder; "
                "without it the LLM will not see the category list"
            )
        elif self.prompt_template:
            try:
                _ = self.system_prompt
            except KeyError as exc:
                errors.append(
                    "prompt.system contains an unsupported placeholder "
                    f"{exc!s}; escape literal braces in JSON examples as '{{{{' and '}}}}'"
                )
            except (IndexError, ValueError) as exc:
                errors.append(
                    "prompt.system is not a valid format string; "
                    f"escape literal braces in JSON examples as '{{{{' and '}}}}' ({exc})"
                )

        if self.user_prompt_template:
            try:
                _ = self.user_prompt_template.format(text="TEXT", context="CONTEXT")
            except KeyError as exc:
                errors.append(
                    "prompt.user contains an unsupported placeholder "
                    f"{exc!s}; only {{{{text}}}} and {{{{context}}}} are supported"
                )
            except (IndexError, ValueError) as exc:
                errors.append(
                    "prompt.user is not a valid format string; "
                    f"escape literal braces in examples as '{{{{' and '}}}}' ({exc})"
                )

        if errors:
            raise ConfigError(f"{len(errors)} configuration error(s): {'; '.join(errors)}")
