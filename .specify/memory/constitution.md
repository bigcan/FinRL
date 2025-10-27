# FinRL Constitution

## Core Principles

### I. Reproducible Quant Research
Every trading experiment must be reproducible end-to-end: deterministic seeds, versioned datasets, locked dependency manifests, and documented CLI invocations (`train.py`, `test.py`, `trade.py`). Any change that threatens reproducibility demands an accompanying regression test or experiment log.

### II. CLI-First Automation
All user workflows are driven through the project CLIs (`poetry run python finrl/...`). Scripts, notebooks, and docs must reference CLI usage, accept configuration via arguments or `.env`, and emit machine-readable logs where feasible.

### III. Test-Gated Trading Logic
Reward shaping, agent policies, and downloader integrations require pytest coverage in `unit_tests/`. Failing to add or update tests when behavior changes is non-negotiable; green tests are the gate for merging.

### IV. Secure Data Stewardship
Market data, API keys, and credentials stay out of version control. Use `.env` templates, environment variables, or secret managers. Any new integration documents required secrets and sanitizes logs.

### V. Incremental Simplicity
Prefer minimal, composable abstractions that mirror the package layout (`finrl/applications`, `agents`, `meta`). Build features iteratively, removing dead code and resisting premature optimization.

## Delivery Constraints
- Dependency management flows through Poetry; contributors must rely on the managed virtualenv (`poetry install`, `poetry run`).
- Pre-commit hooks (`black`, `flake8`, `reorder-python-imports`, `pyupgrade`) run clean before code review.
- Docs, examples, and figures mirror functionality changes; failing to update user-facing assets is a release blocker.
- Offline-friendly development is expected: mock external services and document fixtures for CI use.

## Development Workflow
- Begin work by reviewing the relevant spec or CLI help, then produce or update tasks checklists under `.specify/`.
- Maintain feature branches named with the issue/story number and slug (e.g., `123-add-new-agent`); keep commits scoped and present tense.
- Code review focuses on trading impact, data integrity, and reproducibility. Reviewers verify tests, docs, and configuration notes before approval.
- Before handing off or requesting review, run `poetry run pytest unit_tests` and `poetry run pre-commit run --all-files`; share results in PR descriptions.

## Governance
This constitution supersedes ad-hoc practices. Amendments require consensus from maintainers, documentation of rationale, and simultaneous updates to specs/checklists reflecting the new expectations.

**Version**: 1.0.0 | **Ratified**: 2024-05-09 | **Last Amended**: 2024-05-09
