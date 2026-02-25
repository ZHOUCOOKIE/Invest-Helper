# InvestPulse Public Repo Checklist

Use this checklist before changing repository visibility to Public.

## 1) Secrets and privacy scan

- Search tracked files for secret patterns:
  - `git grep -n -I -E "OPENAI_API_KEY|OPENAI_BASE_URL|ADMIN_TOKEN|Bearer [A-Za-z0-9._-]{10,}|sk-[A-Za-z0-9_-]{10,}|or-[A-Za-z0-9_-]{10,}|xoxb-[A-Za-z0-9-]{10,}"`
- Search for privacy/data exports and local files:
  - `git ls-files | rg -n "(^|/)\\.env($|\\.)|RUNBOOK_LOCAL|twitter-.*\\.json|raw_posts|\\.db$|\\.sqlite3?$|\\.log$|dump"`
- Confirm nothing sensitive is staged:
  - `git status --short`

Expected: no real secrets, no personal/local data exports, no local-only runbooks in tracked files.

## 2) Required ignore policy

Verify root `.gitignore` includes:

- `.env`, `.env.*`, keep only `.env.example`
- local runbook files (`RUNBOOK_LOCAL*`, keep only example template)
- local datasets/exports (`twitter-*.json`, `raw_posts*.json`)
- local DB/log/runtime files (`*.db`, `*.sqlite*`, `*.log`, `tmp/`, `logs/`)

## 3) Secrets management policy

- Keep real values in local environment variables or local `.env` files only.
- Commit only templates such as `apps/api/.env.example`.
- Never commit tokens, API keys, cookies, session values, or personal account exports.
- Rotate any key that has ever been committed.

## 4) If sensitive files exist in git history

Use history rewrite (example with `git filter-repo`) to remove tracked sensitive paths from all commits, then force push.

- Paths typically removed:
  - `apps/api/.env`
  - `RUNBOOK_LOCAL.txt`
  - `twitter-*.json`

After rewrite, verify:

- Local history no longer contains files:
  - `git log --all -- apps/api/.env RUNBOOK_LOCAL.txt --oneline`
- Remote branches and tags are replaced:
  - `git push --force --all origin`
  - `git push --force --tags origin`
- GitHub UI/repo search no longer finds old secrets/data.

## 5) Final public gate

- `git status` clean (or only intentional code changes).
- CI/tests pass.
- GitHub secret scanning and push protection enabled.
- Repository visibility changed to Public only after all checks pass.
