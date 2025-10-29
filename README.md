# create-commit

AI-assisted git commit helper for generating well-structured commit messages directly from your working tree.

## Features

- Infers commit style from recent history and branch naming conventions (e.g. ticket IDs).
- Summarizes the current diff per file and asks an OpenAI GPT-5 family model to propose a commit plan.
- Can split changes into multiple commits when appropriate and apply them automatically.
- Supports a dry-run mode for reviewing the generated plan before any commits are created.

## Prerequisites

- Rust toolchain (edition 2024 compatible).
- Access to the OpenAI API with the `OPENAI_API_KEY` environment variable set.
- Optional environment overrides:
  - `OPENAI_BASE_URL` to point at a different API endpoint.
  - `OPENAI_MODEL` to select a different model (defaults to `gpt-5-mini`).

## Usage

```bash
cargo run -- --dry-run
# Review the suggested commits, then run without --dry-run:
cargo run --
```

Key flags:

- `--dry-run`: only prints the plan produced by the model.
- `--allow-dirty-index`: proceed even if you already have staged changes.
- `--history-limit <n>`: adjust how many recent commit messages are sampled.
- `--diff-char-limit <n>`: cap the characters captured per file diff.
- `--model <name>` / `--openai-base-url <url>`: override model configuration without touching env vars.

## How it works

1. Collects git metadata (branch name, recent commit subjects, working tree status, per-file diffs).
2. Sends that context to an OpenAI chat completion request, instructing the model to emit JSON describing the commit plan.
3. Optionally prints the plan (`--dry-run`) or stages the suggested files and performs commits accordingly.

If the model suggests a commit containing files that have no changes, the tool skips that commit and emits a warning.

## Notes

- The tool expects to run inside an existing git repository.
- Ensure you have a clean index unless you pass `--allow-dirty-index`.
- Review generated commits before pushing. The model responses should be treated as suggestions and verified in your workflow.
