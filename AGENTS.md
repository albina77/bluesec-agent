# Repository Guidelines

## Project Structure & Module Organization

Application code uses a `src/` layout. `src/bluesec1_agent/` contains the SOC investigation loop, CLI, settings, LLM integration, state, and tracing. Treat `src/bluesec1_client/` as the runtime client and transport layer; generated gRPC modules live under `src/bluesec1_client/proto/`. The source schema is `proto/investigation/runtime/v1/runtime.proto`, and `scripts/generate_remote_proto.py` regenerates its Python artifacts. Tests are in `tests/`, with files grouped by component (`test_agent.py`, `test_state.py`, and similar). Runtime traces are written to ignored `runs/traces/`.

## Build, Test, and Development Commands

- `uv sync --dev`: create or update the environment from `uv.lock`, including test and lint tools.
- `uv run pytest`: run the complete pytest suite; configuration in `pyproject.toml` adds `src/` to the import path.
- `uv run pytest tests/test_agent.py`: run one focused test module.
- `uv run ruff check .`: enforce formatting-independent style, import ordering, and lint rules.
- `uv run --env-file .env bluesec1-agent --max-tasks 1`: perform a one-task remote smoke test and close the run cleanly.
- `uv run python scripts/generate_remote_proto.py`: regenerate committed protobuf code after changing the `.proto` contract.

## Coding Style & Naming Conventions

Use Python 3.11 syntax, four-space indentation, and a 100-character line limit. Ruff enables `E`, `F`, `I`, `B`, `UP`, and `SIM`; run it before submitting. Use `snake_case` for functions, modules, and variables, `PascalCase` for classes and Pydantic models, and uppercase names for constants. Add type annotations and concise docstrings to public behavior. Do not hand-edit generated files under `src/bluesec1_client/proto/`.

## Testing Guidelines

Pytest discovers `tests/test_*.py` and functions named `test_*`. Keep tests deterministic and isolated; use fakes for LLM and remote-session behavior rather than live services. Add regression coverage beside the component affected. No coverage threshold is configured, so prioritize meaningful success, validation, retry, and failure-path assertions.

## Commit & Pull Request Guidelines

History currently uses a short, imperative, sentence-style subject (for example, `Build efficient evidence-driven BlueSec agent`). Keep commits focused and avoid committing `.env`, traces, caches, or credentials. Pull requests should explain the behavioral change, identify affected agent/client contracts, link the relevant issue, and list exact checks run. Include sample CLI output or trace excerpts when runtime behavior changes, with secrets and task-sensitive data removed.

## Security & Configuration

Copy `.env.example` to `.env`; never commit real runtime tokens or LLM keys. Use the `practice` arena for development and explicitly verify the arena before a scored run.
