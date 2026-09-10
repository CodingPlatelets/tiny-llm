# AGENTS.md

## Scope

- This file applies to the entire repository.
- Use this as the default test-running policy for coding agents.

## Learning Workflow

- Before helping with course exercises, read and follow `CLAUDE.md` for
  tutoring rules. Those rules apply to Codex as well as Claude.
- Read `LEARN.md` for learning progress, past difficulties, and recorded
  test results. Verify the handoff against current code and relevant tests;
  the log may lag behind the implementation.
- Guide the learner through reasoning and debugging rather than writing
  exercise implementations or copying reference solutions. Follow the
  exceptions in `CLAUDE.md` for explicitly requested fixes and non-exercise work.
- Whenever naming an MLX API, provide its PyTorch counterpart and explain
  meaningful differences, as required by `CLAUDE.md`.
- Keep learning history in `LEARN.md`, not in this instruction file. When
  updating the log, distinguish verified test results from untested code,
  and do not invent the learner's reasoning or claim unobserved progress.
- Before model-dependent checks or downloads, verify the external-volume
  Hugging Face cache path and pass it explicitly when needed; do not
  download model assets to the Mac's main disk.

## Objective

- Run and verify tests in a way that matches the book workflow (`book/src/*.md`).
- Prefer `pdm` entrypoints defined in `pyproject.toml`.

## Environment Requirements

- macOS on Apple Silicon is expected by the project.
- Install dependencies first:

```bash
pdm install -v
pdm run check-installation
```

- Optional baseline check from the setup chapter (reference solution, Week 1):

```bash
pdm run test-refsol -- -- -k week_1
```

## Agent Test Workflow

1. Start with the smallest relevant scope (`--week` + `--day`).
2. Use pytest filters via `-- -k ...` to isolate failing tasks.
3. Run broader suites only after targeted tests pass.
4. If extension code changed, rebuild extensions before testing.

## Canonical Commands

Run all tests:

```bash
pdm run test
```

Run a specific chapter/day:

```bash
pdm run test --week <WEEK> --day <DAY>
```

Run with pytest filters:

```bash
pdm run test --week 1 --day 3 -- -k task_2
pdm run test --week 2 --day 3 -- -k task_3
pdm run test --week 2 --day 3 -- -k gpu
```

Run reference-solution tests:

```bash
pdm run test-refsol
pdm run test-refsol --week 2 --day 3 -- -k task_3
```

## Extension Rebuild Rule

Rebuild before tests if these changed:

- `src/extensions/src/*`

Commands:

```bash
pdm run build-ext
```

## Guardrails

- Use `--` before pytest args (`-k`, `-q`, `--collect-only`, etc.).
- `pdm run test --week X --day Y` auto-copies `tests_refsol/test_week_X_day_Y.py` into `tests/`.
- Model-dependent tests (0.5B/1.5B/7B) skip when models are not downloaded locally.
## GitHub CLI

- The `gh` CLI is installed, authenticated, and expected to work in this
  repository.
- If a `gh` command fails because of sandbox or network restrictions, retry it
  with the required authorization. Do not infer that GitHub or `gh` is
  unavailable from a sandboxed failure.
