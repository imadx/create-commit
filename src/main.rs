use std::ffi::OsStr;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use once_cell::sync::Lazy;
use regex::Regex;
use reqwest::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tempfile::NamedTempFile;

#[derive(Parser, Debug)]
#[command(
    name = "create-commit",
    version,
    about = "AI-assisted commit generator"
)]
struct Cli {
    /// Only print the suggested commits instead of creating them
    #[arg(long)]
    dry_run: bool,

    /// Allow running even when the git index already has staged changes
    #[arg(long)]
    allow_dirty_index: bool,

    /// Maximum number of recent commit messages to sample
    #[arg(long, default_value_t = 20)]
    history_limit: usize,

    /// Limit for captured diff text per file (characters)
    #[arg(long, default_value_t = 6000)]
    diff_char_limit: usize,

    /// Override the OpenAI base URL (defaults to https://api.openai.com/v1)
    #[arg(long)]
    openai_base_url: Option<String>,

    /// Override the model name (defaults to env OPENAI_MODEL or gpt-5-mini)
    #[arg(long)]
    model: Option<String>,
}

#[derive(Debug)]
struct GitRunner {
    root: PathBuf,
}

#[derive(Debug, Serialize)]
struct FileDiff {
    path: String,
    summary: String,
    diff: String,
}

#[derive(Debug, Serialize)]
struct CommitContext {
    branch: Option<String>,
    ticket_hint: Option<String>,
    recent_commit_examples: Vec<String>,
    changes: Vec<FileDiff>,
}

#[derive(Debug, Deserialize)]
struct AiPlan {
    commits: Vec<AiCommit>,
    #[serde(default)]
    notes: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AiCommit {
    title: String,
    #[serde(default)]
    body: Option<String>,
    files: Vec<String>,
    #[serde(default)]
    rationale: Option<String>,
}

struct OpenAiClient {
    http: Client,
    base_url: String,
    model: String,
}

#[tokio::main]
async fn main() {
    if let Err(err) = run().await {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let cli = Cli::parse();
    progress("Collecting repository context...");
    let repo_root = get_repo_root()?;
    let git = GitRunner { root: repo_root };

    if !cli.allow_dirty_index && !git.index_is_clean()? {
        bail!("staged changes detected; run with --allow-dirty-index to override");
    }

    let recent_commits = git.recent_commit_messages(cli.history_limit)?;
    let branch = git.current_branch()?;
    let ticket_hint = branch
        .as_ref()
        .and_then(|name| extract_ticket(name))
        .map(|s| s.to_string());

    progress("Analyzing working tree changes...");
    let status_entries = git.status_entries()?;
    if status_entries.is_empty() {
        bail!("no changes detected in the working tree");
    }

    let diffs = git.collect_diffs(&status_entries, cli.diff_char_limit)?;
    let context = CommitContext {
        branch: branch.clone(),
        ticket_hint,
        recent_commit_examples: recent_commits,
        changes: diffs,
    };

    progress("Contacting OpenAI for commit plan (this may take a moment)...");
    let openai = OpenAiClient::new(cli.openai_base_url, cli.model).await?;
    let plan = openai.build_plan(&context).await?;
    progress(&format!(
        "Received AI plan with {} proposed commit(s)",
        plan.commits.len()
    ));

    if cli.dry_run {
        progress("Dry run enabled; no commits will be created.");
        print_plan(&plan)?;
        return Ok(());
    }

    progress("Ensuring repository is in sync...");
    git.ensure_head_in_sync()?;
    progress("Applying commit plan...");
    let applied = git.apply_plan(&plan)?;
    if applied.is_empty() {
        bail!("no commits were created; check plan output or run with --dry-run");
    }

    println!("Created {} commit(s):", applied.len());
    for commit in applied {
        println!("  - {}", commit);
    }

    if let Some(notes) = plan.notes.as_ref() {
        println!("\nAdditional notes:\n{notes}");
    }

    Ok(())
}

impl GitRunner {
    fn run_git<I, S>(&self, args: I) -> Result<CommandOutput>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        let output = Command::new("git")
            .args(args)
            .current_dir(&self.root)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .with_context(|| "failed to execute git command")?;
        Ok(CommandOutput::new(output))
    }

    fn index_is_clean(&self) -> Result<bool> {
        let status = Command::new("git")
            .args(["diff", "--cached", "--quiet"])
            .current_dir(&self.root)
            .status()
            .with_context(|| "failed to execute git diff --cached")?;
        Ok(status.success())
    }

    fn recent_commit_messages(&self, limit: usize) -> Result<Vec<String>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let output = self.run_git(["log", &format!("-n{limit}"), "--pretty=format:%s"])?;
        if !output.status.success() {
            bail!(
                "git log failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let mut messages = Vec::new();
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                messages.push(trimmed.to_string());
            }
        }
        Ok(messages)
    }

    fn current_branch(&self) -> Result<Option<String>> {
        let output = self.run_git(["rev-parse", "--abbrev-ref", "HEAD"])?;
        if !output.status.success() {
            return Ok(None);
        }
        let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if value == "HEAD" {
            Ok(None)
        } else if value.is_empty() {
            Ok(None)
        } else {
            Ok(Some(value))
        }
    }

    fn status_entries(&self) -> Result<Vec<StatusEntry>> {
        let output = self.run_git(["status", "--porcelain=1"])?;
        if !output.status.success() {
            bail!(
                "git status failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        let mut entries = Vec::new();
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            if line.is_empty() {
                continue;
            }
            if line.starts_with("?? ") {
                let path = line[3..].to_string();
                entries.push(StatusEntry {
                    staged: '?',
                    unstaged: '?',
                    path,
                    original_path: None,
                });
                continue;
            }

            if line.len() < 4 {
                continue;
            }

            let staged = line.chars().nth(0).unwrap_or(' ');
            let unstaged = line.chars().nth(1).unwrap_or(' ');
            let remainder = line[3..].to_string();
            let (path, original_path) = if remainder.contains(" -> ") {
                let mut parts = remainder.splitn(2, " -> ");
                let orig = parts.next().map(|s| s.to_string());
                let new = parts
                    .next()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| remainder.clone());
                (new, orig)
            } else {
                (remainder, None)
            };
            entries.push(StatusEntry {
                staged,
                unstaged,
                path,
                original_path,
            });
        }
        Ok(entries)
    }

    fn collect_diffs(&self, entries: &[StatusEntry], limit: usize) -> Result<Vec<FileDiff>> {
        let mut diffs = Vec::new();
        for entry in entries {
            let summary = format!(
                "staged={} unstaged={}{}",
                entry.staged,
                entry.unstaged,
                entry
                    .original_path
                    .as_ref()
                    .map(|orig| format!(" (from {orig})"))
                    .unwrap_or_default()
            );

            let diff_text = self.diff_for_entry(entry)?;
            let truncated = truncate(&diff_text, limit);
            diffs.push(FileDiff {
                path: entry.path.clone(),
                summary,
                diff: truncated,
            });
        }
        Ok(diffs)
    }

    fn diff_for_entry(&self, entry: &StatusEntry) -> Result<String> {
        let path = &entry.path;
        let mut tried_head = false;

        if entry.staged != '?' || entry.unstaged != '?' {
            tried_head = true;
            if let Some(diff) = self.try_diff_against_head(path)? {
                return Ok(diff);
            }
        }

        if !tried_head || entry.staged == '?' {
            if let Some(diff) = self.try_diff_no_index(path)? {
                return Ok(diff);
            }
        }

        Ok(String::from("diff not available for this file"))
    }

    fn try_diff_against_head(&self, path: &str) -> Result<Option<String>> {
        let output = self.run_git(["diff", "--binary", "HEAD", "--", path])?;
        let code = output.status.code().unwrap_or(-1);
        if code == 0 || code == 1 {
            let text = String::from_utf8_lossy(&output.stdout).to_string();
            if text.trim().is_empty() {
                Ok(None)
            } else {
                Ok(Some(text))
            }
        } else {
            Ok(None)
        }
    }

    fn try_diff_no_index(&self, path: &str) -> Result<Option<String>> {
        let abs = self.root.join(path);
        if !abs.exists() {
            return Ok(None);
        }

        let output = Command::new("git")
            .args(["diff", "--binary", "--no-index", "/dev/null", path])
            .current_dir(&self.root)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .with_context(|| "failed to produce diff for untracked file")?;

        let code = output.status.code().unwrap_or(-1);
        if code == 0 || code == 1 {
            let text = String::from_utf8_lossy(&output.stdout).to_string();
            if text.trim().is_empty() {
                Ok(None)
            } else {
                Ok(Some(text))
            }
        } else {
            Ok(None)
        }
    }

    fn ensure_head_in_sync(&self) -> Result<()> {
        let status = Command::new("git")
            .args(["status", "--short", "--branch"])
            .current_dir(&self.root)
            .stdout(Stdio::piped())
            .output()
            .with_context(|| "failed to read git status")?;
        if !status.status.success() {
            bail!(
                "git status failed: {}",
                String::from_utf8_lossy(&status.stderr)
            );
        }
        Ok(())
    }

    fn apply_plan(&self, plan: &AiPlan) -> Result<Vec<String>> {
        let mut created = Vec::new();
        for commit in &plan.commits {
            progress(&format!("Preparing commit '{}'", commit.title));
            self.stage_files(&commit.files)?;
            if !self.has_staged_changes()? {
                eprintln!(
                    "warning: skipped commit '{}' because no staged changes matched",
                    commit.title
                );
                continue;
            }
            progress(&format!("Creating commit '{}'", commit.title));
            let message = compose_commit_message(commit);
            self.create_commit(&message)?;
            created.push(commit.title.clone());
        }
        Ok(created)
    }

    fn stage_files(&self, files: &[String]) -> Result<()> {
        for path in files {
            let output = self.run_git(["add", "--", path])?;
            if !output.status.success() {
                bail!(
                    "failed to stage '{}': {}",
                    path,
                    String::from_utf8_lossy(&output.stderr)
                );
            }
        }
        Ok(())
    }

    fn has_staged_changes(&self) -> Result<bool> {
        let status = Command::new("git")
            .args(["diff", "--cached", "--quiet"])
            .current_dir(&self.root)
            .status()
            .with_context(|| "failed to inspect staged changes")?;
        Ok(!status.success())
    }

    fn create_commit(&self, message: &str) -> Result<()> {
        let mut temp = NamedTempFile::new().context("failed to create temp commit file")?;
        temp.write_all(message.as_bytes())
            .context("failed to write commit message")?;

        let status = Command::new("git")
            .arg("commit")
            .arg("--file")
            .arg(temp.path())
            .current_dir(&self.root)
            .status()
            .with_context(|| "git commit failed to start")?;

        if !status.success() {
            bail!("git commit failed");
        }
        Ok(())
    }
}

struct CommandOutput {
    status: std::process::ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

impl CommandOutput {
    fn new(output: std::process::Output) -> Self {
        Self {
            status: output.status,
            stdout: output.stdout,
            stderr: output.stderr,
        }
    }
}

#[derive(Debug)]
struct StatusEntry {
    staged: char,
    unstaged: char,
    path: String,
    original_path: Option<String>,
}

impl OpenAiClient {
    async fn new(base: Option<String>, model: Option<String>) -> Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| anyhow!("OPENAI_API_KEY environment variable is required"))?;
        let base_url = base
            .or_else(|| std::env::var("OPENAI_BASE_URL").ok())
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
        let model = model
            .or_else(|| std::env::var("OPENAI_MODEL").ok())
            .unwrap_or_else(|| "gpt-5-mini".to_string());

        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", api_key))
                .context("invalid API key header value")?,
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let http = Client::builder()
            .default_headers(headers)
            .build()
            .context("failed to initialize HTTP client")?;

        Ok(Self {
            http,
            base_url,
            model,
        })
    }

    async fn build_plan(&self, context: &CommitContext) -> Result<AiPlan> {
        let schema = json!({
            "name": "commit_plan",
            "schema": {
                "type": "object",
                "properties": {
                    "commits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "body": {"type": "string"},
                                "files": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 1
                                },
                                "rationale": {"type": "string"}
                            },
                            "required": ["title", "files"]
                        }
                    },
                    "notes": {"type": "string"}
                },
                "required": ["commits"],
                "additionalProperties": false
            }
        });

        let system_prompt = "You are an assistant that writes git commit plans. \
Respect project conventions, include ticket identifiers when appropriate, \
and prefer smaller, focused commits.";

        let context_json = serde_json::to_string_pretty(context)?;
        let user_prompt = format!(
            "Generate a set of clean commits for the current working tree. \
Use the observed commit history to infer the preferred style. \
If the branch name includes a ticket reference, include it in each relevant commit title. \
Return JSON that matches the provided schema. \
Repository context:\n```json\n{}\n```",
            context_json
        );

        let body = json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": schema
            }
        });

        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let response = self
            .http
            .post(url)
            .json(&body)
            .send()
            .await
            .context("failed to call OpenAI API")?;

        if !response.status().is_success() {
            bail!(
                "OpenAI API error: {}",
                response
                    .text()
                    .await
                    .unwrap_or_else(|_| "<unavailable>".to_string())
            );
        }

        let payload: Value = response.json().await.context("invalid OpenAI response")?;
        let Some(choice) = payload["choices"].get(0) else {
            bail!("OpenAI response missing choices");
        };
        let content = choice["message"]["content"]
            .as_str()
            .ok_or_else(|| anyhow!("OpenAI response missing content"))?;
        let plan: AiPlan =
            serde_json::from_str(content).context("failed to parse plan JSON from OpenAI")?;
        Ok(plan)
    }
}

fn progress(message: impl AsRef<str>) {
    println!("[create-commit] {}", message.as_ref());
    let _ = io::stdout().flush();
}

fn compose_commit_message(commit: &AiCommit) -> String {
    let mut message = String::new();
    message.push_str(commit.title.trim());
    if let Some(body) = commit.body.as_ref() {
        let trimmed = body.trim();
        if !trimmed.is_empty() {
            message.push_str("\n\n");
            message.push_str(trimmed);
        }
    }
    message
}

fn extract_ticket(name: &str) -> Option<&str> {
    // simple pattern like ABC-123
    static TICKET_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"([A-Z]{2,}-\d{1,6})").unwrap());
    TICKET_RE
        .captures(name)
        .and_then(|caps| caps.get(1).map(|m| m.as_str()))
}

fn truncate(value: &str, limit: usize) -> String {
    if value.len() <= limit {
        value.to_string()
    } else {
        let mut truncated = value[..limit].to_string();
        truncated.push_str("\n...<truncated>...");
        truncated
    }
}

fn get_repo_root() -> Result<PathBuf> {
    let output = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .context("failed to locate git repository")?;
    if !output.status.success() {
        bail!(
            "not inside a git repository: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(PathBuf::from(
        String::from_utf8_lossy(&output.stdout).trim(),
    ))
}

fn print_plan(plan: &AiPlan) -> Result<()> {
    println!("Planned commits:");
    for commit in &plan.commits {
        println!("- {}", commit.title);
        if let Some(body) = commit.body.as_ref() {
            if !body.trim().is_empty() {
                println!("  body: {}", body.trim().replace('\n', " "));
            }
        }
        println!("  files: {}", commit.files.join(", "));
        if let Some(reason) = commit.rationale.as_ref() {
            println!("  rationale: {}", reason.trim());
        }
    }
    if let Some(notes) = plan.notes.as_ref() {
        println!("\nNotes:\n{notes}");
    }
    Ok(())
}
