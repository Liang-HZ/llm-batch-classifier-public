# Contributing

This document is for readers who want to contribute code, tests, examples, or documentation to this repository.

## Before You Open a Pull Request

- Small fixes are fine to send directly.
- For larger features or behavior changes, open an issue first so the scope is clear before implementation.
- Do not include API keys, customer data, or private business taxonomies in examples, tests, docs, or git history.

## Local Setup

```bash
git clone https://github.com/Liang-HZ/llm-batch-classifier-public.git
cd llm-batch-classifier-public
python -m pip install -e ".[dev]"
```

## Development Expectations

- Keep the project focused on batch classification against a fixed label set.
- Preserve existing behavior unless the change is intentionally breaking and documented.
- Add or update tests for behavior changes.
- Update README or example docs when the user-facing workflow changes.
- Keep examples and fixtures safe for public release.

## Validation

Run the standard checks before opening a pull request:

```bash
pytest -q
python -m compileall src
```

If your change only touches documentation, say so in the pull request.

## Pull Request Checklist

- Explain what changed and why.
- Describe any compatibility risk or migration impact.
- Include test coverage or explain why tests were not needed.
- Keep commits scoped and readable.

## Review Notes

- Behavior regressions and data safety issues take priority over style feedback.
- Changes that broaden scope from classification into generic extraction should be discussed before implementation.

## Community Standards

By participating, you agree to follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
