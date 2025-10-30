# Repository Guidelines

## Project Structure & Module Organization
- Core library in `finrl/` with `applications/`, `agents/`, and `meta/`; `train.py`, `test.py`, `trade.py`, `main.py` drive the train-test-trade flow.
- Reusable notebooks live in `examples/`; docs sources in `docs/`; shared figures in `figs/`.
- Regression checks sit in `unit_tests/` (`environments/`, `downloaders/`). Add new cases next to the code they validate.

## Development Log
- Use `analysis/dev_log.md` as the shared notebook for Codex and Claude to record experiments, fixes, and decisions.
- Append new entries at the top, include date, agent, summary, and link to related artifacts instead of pasting large outputs.
- Do not delete prior notes; add follow-up bullets if results change.

## Environment Setup & Init
- Clone the repo, install Poetry, then run `poetry install` from the repository root to create the managed virtualenv.
- Activate with `poetry shell` or prefix commands with `poetry run`; pip users can call `pip install -e .` for editable imports.
- Install hooks once via `poetry run pre-commit install`; copy `.env.example` artifacts if the feature needs API keys.

## Build, Test, and Development Commands
- `poetry run python finrl/train.py --help` — inspect CLI options before launching experiments.
- `poetry run pytest unit_tests` — run the regression suite; add `-k <pattern>` for focused runs.
- `poetry run pre-commit run --all-files` — apply formatting (`black`, `reorder-python-imports`, `flake8`, `pyupgrade`) locally before committing.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation; `flake8` enforces a 127 char soft limit and ignores `E203`, `W503`. Prefer expressive, lower_snake_case for modules/functions, UpperCamelCase for classes, and ALL_CAPS for constants.
- Run `black` via pre-commit; do not hand-edit to bypass formatting.
- Keep new agents/configs under descriptive subpackages (`finrl/agents/<backend>`); mirror filenames (e.g., `*_config.py` for settings).

## Testing Guidelines
- Tests use `pytest`; name files `test_<feature>.py` and functions `test_<behavior>`.
- Mock external data sources and guard credentials with environment variables. Provide lightweight fixtures so CI can run without network access.
- Add regression tests when changing trading logic, reward shaping, or downloaders; update documentation with expected metrics.

## Commit & Pull Request Guidelines
- Commit messages are short, present-tense summaries (e.g., `fix downloader retry logic`); reference issues or PRs with `(#123)` when applicable.
- For PRs, include: summary of behavior change, testing evidence (`pytest` output or notebook screenshots), and links to related issues/docs. Record any required API keys or configs in the PR description.
- Ensure CI passes and reviewers can reproduce results with the commands above before requesting review.

## Security & Configuration Tips
- Store API tokens in `.env` or secret managers; never commit keys or cached market data.
- Document new environment variables in `docs/` and refresh sample configs in `finrl/config*.py`.
