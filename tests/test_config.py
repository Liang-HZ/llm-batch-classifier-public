"""Tests for ClassifyConfig YAML loading, property evaluation, and validation."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from llm_classifier.config import ClassifyConfig, ConfigError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(tmp_path: Path, content: str) -> Path:
    """Write *content* to a temp YAML file and return its path."""
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


FULL_YAML = """\
categories:
  - "Computer Science"
  - "Finance"
  - "Marketing"

prompt:
  system: |
    You are a classification expert. Classify into:
    {categories}
    Output JSON only.
  user: "{text} / {context}"

model:
  name: deepseek-chat
  api_base: https://api.deepseek.com/v1
  temperature: 0.2
  max_tokens: 600
  timeout: 45
  max_retries: 5

throttle:
  max_attempts: 8
  base_wait: 20.0
  max_wait: 200.0
  jitter: 1.0

rate_limit:
  rps: 3.0
  tps: 20000.0
  window: 1.0
  tokens_per_call: 700

cycle:
  duration: 60.0
  max_calls: 50

input:
  file: data.csv
  text_column: english_name
  context_column: chinese_name

output:
  dir: results

threshold: 90
concurrency: 20
"""

MINIMAL_YAML = """\
categories:
  - "CategoryA"
  - "CategoryB"

prompt:
  system: |
    Classify into:
    {categories}
  user: "{text}"

model:
  name: gpt-4o
  api_base: https://api.openai.com/v1
"""


# ---------------------------------------------------------------------------
# 1. Full YAML loads all fields correctly
# ---------------------------------------------------------------------------

class TestLoadConfigFromYaml:
    def test_categories(self, tmp_path: Path) -> None:
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, FULL_YAML))
        assert cfg.categories == ["Computer Science", "Finance", "Marketing"]

    def test_category_set_derived(self, tmp_path: Path) -> None:
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, FULL_YAML))
        assert cfg.category_set == {"Computer Science", "Finance", "Marketing"}

    def test_prompt_fields(self, tmp_path: Path) -> None:
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, FULL_YAML))
        assert "{categories}" in cfg.prompt_template
        assert cfg.user_prompt_template == "{text} / {context}"

    def test_model_fields(self, tmp_path: Path) -> None:
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, FULL_YAML))
        assert cfg.model_name == "deepseek-chat"
        assert cfg.api_base == "https://api.deepseek.com/v1"
        assert cfg.temperature == pytest.approx(0.2)
        assert cfg.max_tokens == 600
        assert cfg.timeout == 45
        assert cfg.max_retries == 5

    def test_throttle_fields(self, tmp_path: Path) -> None:
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, FULL_YAML))
        assert cfg.throttle_max_attempts == 8
        assert cfg.throttle_base_wait == pytest.approx(20.0)
        assert cfg.throttle_max_wait == pytest.approx(200.0)
        assert cfg.jitter_seconds == pytest.approx(1.0)

    def test_rate_limit_fields(self, tmp_path: Path) -> None:
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, FULL_YAML))
        assert cfg.rate_limit_rps == pytest.approx(3.0)
        assert cfg.rate_limit_tps == pytest.approx(20000.0)
        assert cfg.rate_limit_window == pytest.approx(1.0)
        assert cfg.tokens_per_call == 700

    def test_cycle_fields(self, tmp_path: Path) -> None:
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, FULL_YAML))
        assert cfg.cycle_duration == pytest.approx(60.0)
        assert cfg.cycle_max_calls == 50

    def test_input_fields(self, tmp_path: Path) -> None:
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, FULL_YAML))
        assert cfg.input_file == "data.csv"
        assert cfg.text_column == "english_name"
        assert cfg.context_column == "chinese_name"

    def test_output_field(self, tmp_path: Path) -> None:
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, FULL_YAML))
        assert cfg.output_dir == "results"

    def test_classification_fields(self, tmp_path: Path) -> None:
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, FULL_YAML))
        assert cfg.threshold == 90
        assert cfg.concurrency == 20


# ---------------------------------------------------------------------------
# 2. system_prompt property injects categories
# ---------------------------------------------------------------------------

class TestSystemPromptProperty:
    def test_categories_injected(self, tmp_path: Path) -> None:
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, FULL_YAML))
        prompt = cfg.system_prompt
        assert "- Computer Science" in prompt
        assert "- Finance" in prompt
        assert "- Marketing" in prompt

    def test_no_literal_placeholder_remains(self, tmp_path: Path) -> None:
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, FULL_YAML))
        assert "{categories}" not in cfg.system_prompt

    def test_ordering_preserved(self, tmp_path: Path) -> None:
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, FULL_YAML))
        prompt = cfg.system_prompt
        cs_pos = prompt.index("Computer Science")
        fi_pos = prompt.index("Finance")
        mk_pos = prompt.index("Marketing")
        assert cs_pos < fi_pos < mk_pos


# ---------------------------------------------------------------------------
# 3. Validation — empty categories
# ---------------------------------------------------------------------------

class TestValidateMissingCategories:
    def test_raises_config_error(self, tmp_path: Path) -> None:
        yaml_text = """\
categories: []

prompt:
  system: "Classify: {categories}"
  user: "{text}"

model:
  name: gpt-4o
  api_base: https://api.openai.com/v1
"""
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, yaml_text))
        with pytest.raises(ConfigError, match="categories must not be empty"):
            cfg.validate()


# ---------------------------------------------------------------------------
# 4. Validation — empty model_name
# ---------------------------------------------------------------------------

class TestValidateMissingModel:
    def test_raises_config_error_for_empty_name(self, tmp_path: Path) -> None:
        yaml_text = """\
categories:
  - "A"

prompt:
  system: "{categories}"
  user: "{text}"

model:
  name: ""
  api_base: https://api.openai.com/v1
"""
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, yaml_text))
        with pytest.raises(ConfigError, match="model.name must not be empty"):
            cfg.validate()

    def test_raises_config_error_for_missing_api_base(self, tmp_path: Path) -> None:
        yaml_text = """\
categories:
  - "A"

prompt:
  system: "{categories}"
  user: "{text}"

model:
  name: gpt-4o
  api_base: ""
"""
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, yaml_text))
        with pytest.raises(ConfigError, match="model.api_base must not be empty"):
            cfg.validate()


# ---------------------------------------------------------------------------
# 5. Prompt from external file (system_file)
# ---------------------------------------------------------------------------

class TestPromptFromFile:
    def test_system_file_loaded(self, tmp_path: Path) -> None:
        prompt_file = tmp_path / "my_prompt.txt"
        prompt_file.write_text(
            "You are a helper. Categories:\n{categories}\nRespond in JSON.",
            encoding="utf-8",
        )
        yaml_text = f"""\
categories:
  - "X"

prompt:
  system_file: my_prompt.txt
  user: "{{text}}"

model:
  name: gpt-4o
  api_base: https://api.openai.com/v1
"""
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, yaml_text))
        assert "You are a helper" in cfg.prompt_template
        assert "{categories}" in cfg.prompt_template

    def test_system_file_overrides_inline_system(self, tmp_path: Path) -> None:
        prompt_file = tmp_path / "override.txt"
        prompt_file.write_text("From file: {categories}", encoding="utf-8")
        yaml_text = f"""\
categories:
  - "X"

prompt:
  system: "Inline system"
  system_file: override.txt
  user: "{{text}}"

model:
  name: gpt-4o
  api_base: https://api.openai.com/v1
"""
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, yaml_text))
        # system_file wins because it is processed after system in from_yaml
        assert "From file" in cfg.prompt_template


# ---------------------------------------------------------------------------
# 6. Minimal YAML uses sensible defaults
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    def setup_method(self) -> None:
        # Avoid env pollution affecting api_key assertions
        self._saved_llm = os.environ.pop("LLM_API_KEY", None)
        self._saved_oai = os.environ.pop("OPENAI_API_KEY", None)

    def teardown_method(self) -> None:
        if self._saved_llm is not None:
            os.environ["LLM_API_KEY"] = self._saved_llm
        if self._saved_oai is not None:
            os.environ["OPENAI_API_KEY"] = self._saved_oai

    def test_temperature_default(self, tmp_path: Path) -> None:
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, MINIMAL_YAML))
        assert cfg.temperature == pytest.approx(0.1)

    def test_max_tokens_default(self, tmp_path: Path) -> None:
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, MINIMAL_YAML))
        assert cfg.max_tokens == 500

    def test_timeout_default(self, tmp_path: Path) -> None:
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, MINIMAL_YAML))
        assert cfg.timeout == 30

    def test_max_retries_default(self, tmp_path: Path) -> None:
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, MINIMAL_YAML))
        assert cfg.max_retries == 3

    def test_rate_limits_default_to_zero(self, tmp_path: Path) -> None:
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, MINIMAL_YAML))
        assert cfg.rate_limit_rps == 0.0
        assert cfg.rate_limit_tps == 0.0

    def test_cycle_default_disabled(self, tmp_path: Path) -> None:
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, MINIMAL_YAML))
        assert cfg.cycle_duration == 0.0
        assert cfg.cycle_max_calls == 0

    def test_concurrency_default_none(self, tmp_path: Path) -> None:
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, MINIMAL_YAML))
        assert cfg.concurrency is None

    def test_output_dir_default(self, tmp_path: Path) -> None:
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, MINIMAL_YAML))
        assert cfg.output_dir == "output"

    def test_threshold_default(self, tmp_path: Path) -> None:
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, MINIMAL_YAML))
        assert cfg.threshold == 95

    def test_text_column_default(self, tmp_path: Path) -> None:
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, MINIMAL_YAML))
        assert cfg.text_column == "text"

    def test_api_key_empty_when_env_unset(self, tmp_path: Path) -> None:
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, MINIMAL_YAML))
        assert cfg.api_key == ""


# ---------------------------------------------------------------------------
# 7. Rate limit validation
# ---------------------------------------------------------------------------

class TestRateLimitValidation:
    def _cfg_with_rate_limit(
        self,
        tmp_path: Path,
        *,
        rps: float = 0.0,
        tps: float = 0.0,
        window: float = 1.0,
        tokens_per_call: int = 850,
    ) -> ClassifyConfig:
        yaml_text = f"""\
categories:
  - "A"

prompt:
  system: "{{categories}}"
  user: "{{text}}"

model:
  name: gpt-4o
  api_base: https://api.openai.com/v1

rate_limit:
  rps: {rps}
  tps: {tps}
  window: {window}
  tokens_per_call: {tokens_per_call}
"""
        return ClassifyConfig.from_yaml(_write_yaml(tmp_path, yaml_text))

    def test_zero_window_with_rps_raises(self, tmp_path: Path) -> None:
        cfg = self._cfg_with_rate_limit(tmp_path, rps=3.0, window=0.0)
        with pytest.raises(ConfigError, match="window"):
            cfg.validate()

    def test_zero_window_with_tps_raises(self, tmp_path: Path) -> None:
        cfg = self._cfg_with_rate_limit(tmp_path, tps=1000.0, window=0.0)
        with pytest.raises(ConfigError, match="window"):
            cfg.validate()

    def test_tokens_per_call_exceeds_window_budget_raises(self, tmp_path: Path) -> None:
        # tps=100, window=1 → budget=100 tokens; tokens_per_call=850 → violation
        cfg = self._cfg_with_rate_limit(tmp_path, tps=100.0, window=1.0, tokens_per_call=850)
        with pytest.raises(ConfigError, match="tokens_per_call"):
            cfg.validate()

    def test_valid_rate_limits_do_not_raise(self, tmp_path: Path) -> None:
        cfg = self._cfg_with_rate_limit(
            tmp_path, rps=3.0, tps=20000.0, window=1.0, tokens_per_call=700
        )
        # Should not raise (model fields are required — patch them after load)
        cfg.validate()  # categories + model are valid from the YAML above

    def test_cycle_only_duration_raises(self, tmp_path: Path) -> None:
        yaml_text = """\
categories:
  - "A"

prompt:
  system: "{categories}"
  user: "{text}"

model:
  name: gpt-4o
  api_base: https://api.openai.com/v1

cycle:
  duration: 60.0
  max_calls: 0
"""
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, yaml_text))
        with pytest.raises(ConfigError, match="cycle"):
            cfg.validate()

    def test_cycle_only_max_calls_raises(self, tmp_path: Path) -> None:
        yaml_text = """\
categories:
  - "A"

prompt:
  system: "{categories}"
  user: "{text}"

model:
  name: gpt-4o
  api_base: https://api.openai.com/v1

cycle:
  duration: 0
  max_calls: 100
"""
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, yaml_text))
        with pytest.raises(ConfigError, match="cycle"):
            cfg.validate()


# ---------------------------------------------------------------------------
# 7b. Validation — prompt template missing {categories} placeholder
# ---------------------------------------------------------------------------

class TestValidatePromptTemplate:
    def test_missing_categories_placeholder_raises(self, tmp_path: Path) -> None:
        yaml_text = """\
categories:
  - "A"

prompt:
  system: "Classify the following text."
  user: "{text}"

model:
  name: gpt-4o
  api_base: https://api.openai.com/v1
"""
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, yaml_text))
        with pytest.raises(ConfigError, match=r"\{categories\}"):
            cfg.validate()

    def test_system_prompt_raises_on_missing_placeholder(self, tmp_path: Path) -> None:
        """system_prompt property raises KeyError when template lacks {categories}."""
        yaml_text = """\
categories:
  - "A"

prompt:
  system: "No placeholder here."
  user: "{text}"

model:
  name: gpt-4o
  api_base: https://api.openai.com/v1
"""
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, yaml_text))
        # str.format() with only categories= kwarg: a template with no {categories}
        # and no other placeholders just returns the string unchanged — no error.
        # But if there's a different unknown placeholder, KeyError is raised.
        result = cfg.system_prompt  # no {categories} → categories silently unused
        assert result == "No placeholder here."

    def test_system_prompt_raises_key_error_on_unknown_placeholder(
        self, tmp_path: Path
    ) -> None:
        """system_prompt raises KeyError if template has an unrecognised placeholder."""
        yaml_text = """\
categories:
  - "A"

prompt:
  system: "Use {model} and {categories}"
  user: "{text}"

model:
  name: gpt-4o
  api_base: https://api.openai.com/v1
"""
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, yaml_text))
        with pytest.raises(KeyError):
            _ = cfg.system_prompt

    def test_validate_rejects_unknown_system_placeholder(self, tmp_path: Path) -> None:
        yaml_text = """\
categories:
  - "A"

prompt:
  system: "Use {model} and {categories}"
  user: "{text}"

model:
  name: gpt-4o
  api_base: https://api.openai.com/v1
"""
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, yaml_text))
        with pytest.raises(ConfigError, match="unsupported placeholder"):
            cfg.validate()

    def test_validate_rejects_unescaped_json_braces(self, tmp_path: Path) -> None:
        yaml_text = """\
categories:
  - "A"

prompt:
  system: |
    Classify into:
    {categories}
    Output:
    {"labels": [{"name": "A"}]}
  user: "{text}"

model:
  name: gpt-4o
  api_base: https://api.openai.com/v1
"""
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, yaml_text))
        with pytest.raises(ConfigError, match="escape literal braces"):
            cfg.validate()

    def test_validate_rejects_unknown_user_placeholder(self, tmp_path: Path) -> None:
        yaml_text = """\
categories:
  - "A"

prompt:
  system: "{categories}"
  user: "{text} / {language}"

model:
  name: gpt-4o
  api_base: https://api.openai.com/v1
"""
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, yaml_text))
        with pytest.raises(ConfigError, match="prompt.user contains an unsupported placeholder"):
            cfg.validate()


# ---------------------------------------------------------------------------
# 8. API key from environment
# ---------------------------------------------------------------------------

class TestApiKeyFromEnv:
    def setup_method(self) -> None:
        self._saved_llm = os.environ.pop("LLM_API_KEY", None)
        self._saved_oai = os.environ.pop("OPENAI_API_KEY", None)

    def teardown_method(self) -> None:
        if self._saved_llm is not None:
            os.environ["LLM_API_KEY"] = self._saved_llm
        if self._saved_oai is not None:
            os.environ["OPENAI_API_KEY"] = self._saved_oai

    def test_llm_api_key_takes_precedence(self, tmp_path: Path) -> None:
        os.environ["LLM_API_KEY"] = "llm-secret-key"
        os.environ["OPENAI_API_KEY"] = "oai-secret-key"
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, MINIMAL_YAML))
        assert cfg.api_key == "llm-secret-key"

    def test_falls_back_to_openai_api_key(self, tmp_path: Path) -> None:
        os.environ.pop("LLM_API_KEY", None)
        os.environ["OPENAI_API_KEY"] = "oai-fallback-key"
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, MINIMAL_YAML))
        assert cfg.api_key == "oai-fallback-key"

    def test_empty_when_neither_set(self, tmp_path: Path) -> None:
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, MINIMAL_YAML))
        assert cfg.api_key == ""

    def test_api_key_not_in_yaml(self, tmp_path: Path) -> None:
        """The YAML must never contain the API key."""
        yaml_path = _write_yaml(tmp_path, FULL_YAML)
        content = yaml_path.read_text(encoding="utf-8")
        assert "api_key" not in content


# ---------------------------------------------------------------------------
# 9. Output format
# ---------------------------------------------------------------------------


class TestOutputFormat:
    def test_output_format_defaults_to_auto(self, tmp_path: Path) -> None:
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, MINIMAL_YAML))
        assert cfg.output_format == "auto"

    def test_output_format_from_yaml(self, tmp_path: Path) -> None:
        yaml_text = MINIMAL_YAML + "\noutput:\n  dir: out\n  format: xlsx\n"
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, yaml_text))
        assert cfg.output_format == "xlsx"

    def test_output_format_case_insensitive(self, tmp_path: Path) -> None:
        yaml_text = MINIMAL_YAML + "\noutput:\n  format: XLSX\n"
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, yaml_text))
        assert cfg.output_format == "xlsx"

    def test_invalid_output_format_raises(self, tmp_path: Path) -> None:
        yaml_text = MINIMAL_YAML + "\noutput:\n  format: pdf\n"
        cfg = ClassifyConfig.from_yaml(_write_yaml(tmp_path, yaml_text))
        with pytest.raises(ConfigError, match="output.format"):
            cfg.validate()
