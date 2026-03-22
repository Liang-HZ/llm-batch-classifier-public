# Security Policy

## Supported Versions

Security fixes are provided on a best-effort basis for:

- The latest commit on `main`
- The latest published PyPI release

Older versions may not receive patches.

## Reporting a Vulnerability

Please do not open a public issue for a suspected vulnerability.

Use one of these paths instead:

- GitHub Private Vulnerability Reporting for this repository, if it is enabled
- A direct maintainer contact method exposed on the maintainer's GitHub profile

Include the following details:

- A clear description of the issue
- Reproduction steps or a minimal proof of concept
- Affected version, commit, or environment
- Impact assessment, if known

## Disclosure Expectations

- Please give the maintainer reasonable time to investigate and prepare a fix before public disclosure.
- Do not include secrets, customer data, or private datasets in reports.
- Best-effort acknowledgement and follow-up will be provided, but no formal SLA is promised.

## Scope Notes

The most relevant security concerns for this project are:

- Secret leakage in examples, tests, configs, or git history
- Unsafe handling of untrusted files used as batch inputs
- Dependency issues that create code execution or data exposure risk
