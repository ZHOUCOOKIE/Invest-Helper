# Security Policy

## Supported Versions

Security fixes are provided for active development targets only.

| Version/Branch | Supported |
| --- | --- |
| `main` (latest commit) | :white_check_mark: |
| Latest release tag | :white_check_mark: |
| Older release tags | :x: |
| Unreleased forks/custom patches | :x: |

Notes:
- If there is no formal release tag yet, `main` is the only supported target.
- Security patches are applied forward; backports are not guaranteed.

## Reporting a Vulnerability

Please do **not** open public issues for suspected vulnerabilities.

Preferred channel:
- Use GitHub private reporting: `Security` tab -> `Report a vulnerability`.

If private reporting is unavailable in your environment:
- Open a normal issue with the title prefix `[SECURITY-REQUEST]` and include only minimal, non-sensitive context.
- Ask maintainers to move the report to a private channel before sharing details.

## What To Include

Provide as much of the following as possible:
- Affected component (API/Web/DB/migration/script).
- Reproduction steps and prerequisites.
- Impact assessment (data exposure, auth bypass, integrity, availability).
- Proof-of-concept details (sanitized).
- Suggested fix or mitigation (if available).

## Response SLA (Best Effort)

- Initial triage acknowledgment: within **3 business days**.
- Status update after triage: within **7 business days**.
- Fix timeline depends on severity and release readiness.

Severity guidelines:
- Critical/High: prioritized for immediate patch planning.
- Medium/Low: scheduled into upcoming maintenance work.

## Disclosure Policy

- Coordinated disclosure is preferred.
- Public disclosure should wait until a fix or mitigation is available.
- If a report is accepted, release notes/changelog should reference the security fix.

## Scope and Operational Notes

In-scope areas include:
- FastAPI routes, extraction workflow, normalization, review paths.
- Traceability/replay data handling (`raw_posts`, `post_extractions`, `daily_digests`).
- Auth/configuration mistakes that may expose sensitive data.
- Dependency and container configuration risks.

Out-of-scope:
- Best-practice suggestions without a concrete exploit path.
- Issues only reproducible on heavily modified forks.

## Safe Harbor

We support good-faith security research and will not pursue action for:
- Non-destructive testing.
- Responsible, private disclosure.
- Avoiding privacy violations, data destruction, and service disruption.

