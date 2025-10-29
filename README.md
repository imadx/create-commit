# create-commit

AI-assisted git commit helper for generating well-structured commit messages directly from your working tree.

## Features

- Infers commit style from recent history and branch naming conventions (e.g. ticket IDs).
- Summarizes the current diff per file and asks an AI model (OpenAI GPT-5 family or local Ollama) to propose a commit plan.
- Can split changes into multiple commits when appropriate and apply them automatically.
- Supports a dry-run mode for reviewing the generated plan before any commits are created.
- Works with OpenAI or a local Ollama model so you can stay online or offline. Ollama runs an extra diff-based refinement pass to tighten commit summaries and enforce ticket prefixes when required.

## Prerequisites

- Rust toolchain (edition 2024 compatible).
- For OpenAI: set `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`, `OPENAI_MODEL`).
- For Ollama: run a local Ollama daemon with a chat-capable model pulled (`ollama pull llama3.2`, etc.). Optional overrides via `OLLAMA_BASE_URL` and `OLLAMA_MODEL`.

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
- `--provider {openai|ollama}`: choose the AI backend (default `openai`).
- `--model <name>` / `--openai-base-url <url>`: override OpenAI configuration without touching env vars.
- `--ollama-model <name>` / `--ollama-base-url <url>`: override Ollama configuration when using the local provider.

### Using Ollama

```bash
# Ensure Ollama is running and your model is available
ollama pull llama3.2

cargo run -- --provider ollama --ollama-model llama3.2 --dry-run
```

## How it works

1. Collects git metadata (branch name, recent commit subjects, working tree status, per-file diffs).
2. Sends that context to the selected AI provider (OpenAI chat completions or Ollama) and asks for a JSON commit plan.
3. Optionally prints the plan (`--dry-run`) or stages the suggested files and performs commits accordingly.

If the model suggests a commit containing files that have no changes, the tool skips that commit and emits a warning.

## Notes

- The tool expects to run inside an existing git repository.
- Ensure you have a clean index unless you pass `--allow-dirty-index`.
- Review generated commits before pushing. The model responses should be treated as suggestions and verified in your workflow.
