use std::collections::{HashMap, HashSet};
use std::ffi::OsStr;
use std::fmt;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, ValueEnum};
use once_cell::sync::Lazy;
use regex::Regex;
use reqwest::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tempfile::NamedTempFile;

#[derive(Debug, Copy, Clone, Eq, PartialEq, ValueEnum)]
enum Provider {
    #[clap(alias = "openai")]
    Openai,
    #[clap(alias = "ollama")]
    Ollama,
}

impl fmt::Display for Provider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Provider::Openai => write!(f, "OpenAI"),
            Provider::Ollama => write!(f, "Ollama"),
        }
    }
}

const DEFAULT_OLLAMA_BASE_URL: &str = "http://localhost:11434";
const DEFAULT_OLLAMA_MODEL: &str = "llama3.2";

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

    /// Select the AI provider to use (openai or ollama)
    #[arg(long, value_enum, default_value_t = Provider::Openai)]
    provider: Provider,

    /// Override the OpenAI base URL (defaults to https://api.openai.com/v1)
    #[arg(long)]
    openai_base_url: Option<String>,

    /// Override the OpenAI model name (defaults to env OPENAI_MODEL or gpt-5-mini)
    #[arg(long)]
    model: Option<String>,

    /// Override the Ollama model name (defaults to env OLLAMA_MODEL or llama3.2)
    #[arg(long)]
    ollama_model: Option<String>,

    /// Override the Ollama base URL (defaults to http://localhost:11434)
    #[arg(long)]
    ollama_base_url: Option<String>,
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
    ticket_required: bool,
    recent_commit_examples: Vec<String>,
    changes: Vec<FileDiff>,
}

#[derive(Debug, Clone)]
struct TicketFormat {
    prefix: String,
    suffix: String,
    separator: String,
}

impl TicketFormat {
    fn with_summary(&self, ticket: &str, summary: &str) -> String {
        format!(
            "{}{}{}{}{}",
            self.prefix, ticket, self.suffix, self.separator, summary
        )
    }

    fn generic_example(&self, summary: &str) -> String {
        self.with_summary("TICKET-123", summary)
    }
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

#[derive(Debug, Clone)]
struct PromptBundle {
    system: String,
    user: String,
    schema: Value,
}

#[derive(Debug, Deserialize)]
struct RefinedCommit {
    title: String,
    #[serde(default)]
    body: Option<String>,
}

enum AiClient {
    OpenAi(OpenAiClient),
    Ollama(OllamaClient),
}

impl AiClient {
    async fn new(cli: &Cli) -> Result<Self> {
        match cli.provider {
            Provider::Openai => {
                let client =
                    OpenAiClient::new(cli.openai_base_url.clone(), cli.model.clone()).await?;
                Ok(Self::OpenAi(client))
            }
            Provider::Ollama => {
                let client =
                    OllamaClient::new(cli.ollama_base_url.clone(), cli.ollama_model.clone())
                        .await?;
                Ok(Self::Ollama(client))
            }
        }
    }

    async fn build_plan(&self, prompts: &PromptBundle) -> Result<AiPlan> {
        match self {
            AiClient::OpenAi(client) => client.build_plan(prompts).await,
            AiClient::Ollama(client) => client.build_plan(prompts).await,
        }
    }

    async fn refine_commits(&self, plan: &mut AiPlan, context: &CommitContext) -> Result<()> {
        match self {
            AiClient::OpenAi(_) => Ok(()),
            AiClient::Ollama(client) => client.refine_commits(plan, context).await,
        }
    }
}

struct OpenAiClient {
    http: Client,
    base_url: String,
    model: String,
}

struct OllamaClient {
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
    progress(&format!("Using {} provider", cli.provider));
    progress("Collecting repository context...");
    let repo_root = get_repo_root()?;
    let git = GitRunner { root: repo_root };

    if !cli.allow_dirty_index && !git.index_is_clean()? {
        bail!("staged changes detected; run with --allow-dirty-index to override");
    }

    let mut pass = 1usize;
    let mut all_commits = Vec::new();
    let mut aggregated_notes = Vec::new();

    loop {
        if pass == 1 {
            progress("Analyzing working tree changes...");
        } else {
            progress(&format!("Analyzing remaining changes (pass #{pass})..."));
        }

        let status_entries = git.status_entries()?;
        if status_entries.is_empty() {
            if pass == 1 {
                bail!("no changes detected in the working tree");
            } else {
                break;
            }
        }

        let diffs = git.collect_diffs(&status_entries, cli.diff_char_limit)?;
        let recent_commits = git.recent_commit_messages(cli.history_limit)?;
        let branch = git.current_branch()?;
        let ticket_hint = branch
            .as_ref()
            .and_then(|name| extract_ticket(name))
            .map(|s| s.to_string());
        let ticket_required = ticket_hint.as_ref().map_or(false, |ticket| {
            recent_commits.iter().any(|msg| msg.contains(ticket))
        });

        let context = CommitContext {
            branch: branch.clone(),
            ticket_hint,
            ticket_required,
            recent_commit_examples: recent_commits,
            changes: diffs,
        };

        let prompts = build_prompts(&context)?;
        debug_log(&format!("System prompt: {}", prompts.system));
        debug_log(&format!(
            "User prompt snippet: {}",
            truncate(&prompts.user, 500)
        ));

        let plan = {
            debug_log(&format!("Opening AI planning session for pass #{}", pass));
            let ai_client = AiClient::new(&cli).await?;
            progress(&format!(
                "Contacting {} for commit plan (pass #{pass}, this may take a moment)...",
                cli.provider
            ));
            let mut plan = ai_client.build_plan(&prompts).await?;
            sanitize_plan(&mut plan, &context);
            ai_client.refine_commits(&mut plan, &context).await?;
            progress(&format!(
                "Received AI plan with {} proposed commit(s) in pass #{}",
                plan.commits.len(),
                pass
            ));
            plan
        };

        if cli.dry_run {
            progress("Dry run enabled; no commits will be created.");
            print_plan(&plan)?;
            return Ok(());
        }

        if pass == 1 {
            progress("Ensuring repository is in sync...");
            git.ensure_head_in_sync()?;
        }

        progress("Applying commit plan...");
        let applied = git.apply_plan(&plan)?;
        if applied.is_empty() {
            bail!("no commits were created; check plan output or run with --dry-run");
        }
        progress(&format!(
            "Completed pass #{} with {} commit(s).",
            pass,
            applied.len()
        ));
        all_commits.extend(applied);
        if let Some(notes) = plan.notes.as_ref() {
            aggregated_notes.push(notes.clone());
        }

        pass += 1;
    }

    if all_commits.is_empty() {
        bail!("no commits were created; check plan output or run with --dry-run");
    }

    println!("Created {} commit(s):", all_commits.len());
    for commit in &all_commits {
        println!("  - {}", commit);
    }

    if !aggregated_notes.is_empty() {
        println!("\nAdditional notes:");
        for notes in aggregated_notes {
            println!("{notes}");
        }
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

    async fn build_plan(&self, prompts: &PromptBundle) -> Result<AiPlan> {
        let schema = prompts.schema.clone();
        let body = json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompts.system.as_str()},
                {"role": "user", "content": prompts.user.as_str()}
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
        debug_log(&format!("OpenAI raw payload: {}", payload));
        let Some(choice) = payload["choices"].get(0) else {
            bail!("OpenAI response missing choices");
        };
        let content = choice["message"]["content"]
            .as_str()
            .ok_or_else(|| anyhow!("OpenAI response missing content"))?;
        debug_log(&format!("OpenAI content: {}", content));
        parse_json_str(content)
    }
}

impl OllamaClient {
    async fn new(base: Option<String>, model: Option<String>) -> Result<Self> {
        let base_url = base
            .or_else(|| std::env::var("OLLAMA_BASE_URL").ok())
            .unwrap_or_else(|| DEFAULT_OLLAMA_BASE_URL.to_string());
        let model = model
            .or_else(|| std::env::var("OLLAMA_MODEL").ok())
            .unwrap_or_else(|| DEFAULT_OLLAMA_MODEL.to_string());

        let http = Client::builder()
            .no_proxy()
            .build()
            .context("failed to initialize HTTP client for Ollama")?;

        Ok(Self {
            http,
            base_url,
            model,
        })
    }

    async fn build_plan(&self, prompts: &PromptBundle) -> Result<AiPlan> {
        let base_prompt = format!(
            "{}\nRespond ONLY with a single JSON object matching the schema. \
Do not include markdown code fences or additional narration.",
            prompts.user
        );

        let example_plan = json!({
            "commits": [
                {
                    "title": "TICKET-123: Describe the change",
                    "files": ["path/to/file"]
                }
            ]
        });
        let mut example_text = serde_json::to_string_pretty(&example_plan)?;
        example_text.push_str(
            "\nOptional fields (`body`, `rationale`, `notes`) may be included when needed, but the structure must stay the same."
        );

        let prompt_variants = [
            base_prompt.clone(),
            format!(
                "{}\nThe JSON MUST include a top-level `commits` array that follows the provided schema. \
Avoid returning unrelated JSON structures.",
                base_prompt
            ),
            format!(
                "{}\nReturn JSON that mirrors this template exactly:\n{}",
                base_prompt, example_text
            ),
        ];

        let mut last_error: Option<anyhow::Error> = None;
        for user_prompt in prompt_variants {
            match self
                .request_plan_content(&prompts.system, &user_prompt)
                .await
            {
                Ok(content) => match parse_json_str::<AiPlan>(&content) {
                    Ok(plan) if !plan.commits.is_empty() => return Ok(plan),
                    Ok(_) => {
                        last_error = Some(anyhow!(
                            "Ollama returned a plan without any commits; content snippet: {}",
                            truncate(&content, 500)
                        ));
                    }
                    Err(err) => {
                        last_error = Some(anyhow!("failed to parse plan JSON from Ollama: {err}"));
                    }
                },
                Err(err) => {
                    last_error = Some(err);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow!("failed to obtain plan from Ollama")))
    }

    async fn request_plan_content(&self, system_prompt: &str, user_prompt: &str) -> Result<String> {
        let body = json!({
            "model": self.model,
            "format": "json",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": false,
            "options": {
                "temperature": 0.15
            }
        });

        let url = format!("{}/api/chat", self.base_url.trim_end_matches('/'));
        let response = self
            .http
            .post(url)
            .json(&body)
            .send()
            .await
            .context("failed to call Ollama API")?;

        if !response.status().is_success() {
            bail!(
                "Ollama API error: {}",
                response
                    .text()
                    .await
                    .unwrap_or_else(|_| "<unavailable>".to_string())
            );
        }

        let payload: Value = response.json().await.context("invalid Ollama response")?;
        debug_log(&format!("Ollama raw payload: {}", payload));
        let content = Self::extract_content(payload)?;
        debug_log(&format!("Ollama content: {}", content));
        if content.trim().is_empty() {
            bail!("Ollama returned an empty message");
        }
        Ok(content)
    }

    async fn refine_commits(&self, plan: &mut AiPlan, context: &CommitContext) -> Result<()> {
        let diff_map: HashMap<&str, &FileDiff> = context
            .changes
            .iter()
            .map(|diff| (diff.path.as_str(), diff))
            .collect();

        for commit in &mut plan.commits {
            let mut sections = Vec::new();
            for file in &commit.files {
                if let Some(diff) = diff_map.get(file.as_str()) {
                    sections.push(format!(
                        "File: {}\nSummary: {}\nDiff:\n{}\n",
                        diff.path, diff.summary, diff.diff
                    ));
                } else {
                    debug_log(&format!(
                        "No diff captured for file '{}' during refinement",
                        file
                    ));
                }
            }

            if sections.is_empty() {
                continue;
            }

            let combined = truncate(&sections.join("\n"), 12000);
            let current_body = commit.body.as_deref().unwrap_or("").trim();
            let guidance = match (context.ticket_hint.as_deref(), context.ticket_required) {
                (Some(ticket), true) => format!(
                    "Craft a title that begins with '{ticket}: ' followed by a concise summary of the change. \
If the summary already mentions the ticket, do not repeat it, but the prefix is mandatory."
                ),
                (Some(ticket), false) => format!(
                    "If it improves clarity, include the ticket identifier '{ticket}' in the title, but only when it reads naturally."
                ),
                (None, _) => String::from(
                    "Craft a title that clearly summarizes the logical change. \
Expand the body only if additional context is necessary.",
                ),
            };

            let refinement_prompt = format!(
                "You are refining a git commit message based on the diff of the files being committed.\n\
Existing commit suggestion:\n  Title: {}\n  Body: {}\n\n\
{}\n\
Diff excerpt (truncated if large):\n{}\n\
Return a single JSON object that looks like:\n{{\"title\": \"...\", \"body\": \"...\"}}\n\
- Always include `title`.\n\
- Include `body` only when extra explanation beyond the title is useful; otherwise omit it.\n\
- Do not return schemas, code fences, or additional prose.\n\
- Ensure the title summarizes the change, honors ticket requirements, and avoids repeating the branch name or duplicating the body.",
                commit.title.trim(),
                if current_body.is_empty() {
                    "<none>"
                } else {
                    current_body
                },
                guidance,
                combined
            );

            let body = json!({
                "model": self.model,
                "format": "json",
                "messages": [
                    {"role": "system", "content": "You improve git commit messages so they precisely summarize the associated diff."},
                    {"role": "user", "content": refinement_prompt}
                ],
                "stream": false,
                "options": {
                    "temperature": 0.1
                }
            });

            let url = format!("{}/api/chat", self.base_url.trim_end_matches('/'));
            let response = self
                .http
                .post(&url)
                .json(&body)
                .send()
                .await
                .context("failed to call Ollama API for commit refinement")?;

            if !response.status().is_success() {
                debug_log(&format!(
                    "Ollama refinement error status {} for commit '{}'",
                    response.status(),
                    commit.title
                ));
                continue;
            }

            let payload: Value = response
                .json()
                .await
                .context("invalid Ollama refinement response")?;
            debug_log(&format!("Ollama refinement payload: {}", payload));
            let content = Self::extract_content(payload)?;
            debug_log(&format!("Ollama refinement content: {}", content));

            if content.trim().is_empty() {
                continue;
            }

            match parse_json_str::<RefinedCommit>(&content) {
                Ok(refined) => {
                    commit.title = refined.title.trim().to_string();
                    let title_lower = commit.title.to_lowercase();
                    let ticket_prefix_lower = context
                        .ticket_hint
                        .as_ref()
                        .map(|ticket| format!("{}:", ticket.to_lowercase()));
                    commit.body = refined.body.and_then(|b| {
                        let trimmed = b.trim();
                        let body_lower = trimmed.to_lowercase();
                        let starts_with_ticket = ticket_prefix_lower
                            .as_ref()
                            .map(|prefix| body_lower.starts_with(prefix))
                            .unwrap_or(false);
                        if trimmed.is_empty() || body_lower == title_lower || starts_with_ticket {
                            None
                        } else {
                            Some(trimmed.to_string())
                        }
                    });
                }
                Err(err) => {
                    debug_log(&format!(
                        "Failed to parse refinement JSON for commit '{}': {err}",
                        commit.title
                    ));
                }
            }
        }

        Ok(())
    }

    fn extract_content(payload: Value) -> Result<String> {
        if let Some(msg) = payload
            .get("message")
            .and_then(|message| message.get("content"))
            .and_then(|c| c.as_str())
        {
            return Ok(msg.to_string());
        }

        if let Some(messages) = payload.get("messages").and_then(|m| m.as_array()) {
            if let Some(text) = messages
                .iter()
                .rev()
                .find(|msg| msg.get("role").and_then(|r| r.as_str()) == Some("assistant"))
                .and_then(|msg| {
                    msg.get("content").map(|c| {
                        if c.is_string() {
                            c.as_str().map(|s| s.to_string())
                        } else if let Some(array) = c.as_array() {
                            Some(
                                array
                                    .iter()
                                    .filter_map(|chunk| chunk.as_str())
                                    .collect::<Vec<_>>()
                                    .join(""),
                            )
                        } else {
                            None
                        }
                    })
                })
                .flatten()
            {
                return Ok(text);
            }
        }

        bail!("Ollama response missing assistant message content")
    }
}

fn parse_json_str<T>(content: &str) -> Result<T>
where
    T: DeserializeOwned,
{
    match serde_json::from_str(content) {
        Ok(value) => Ok(value),
        Err(primary_err) => {
            if let Some(extracted) = extract_json_object(content) {
                debug_log(&format!(
                    "Attempting to parse extracted JSON snippet: {}",
                    truncate(&extracted, 200)
                ));
                match serde_json::from_str(&extracted) {
                    Ok(value) => Ok(value),
                    Err(err) => Err(anyhow!(
                        "failed to parse JSON after extraction: {err}; original content snippet: {}",
                        truncate(content, 500)
                    )),
                }
            } else {
                Err(anyhow!(
                    "failed to parse JSON: {primary_err}; content snippet: {}",
                    truncate(content, 500)
                ))
            }
        }
    }
}

fn extract_json_object(content: &str) -> Option<String> {
    let mut in_string = false;
    let mut escape = false;
    let mut depth = 0usize;
    let mut start_idx: Option<usize> = None;

    for (idx, ch) in content.char_indices() {
        if escape {
            escape = false;
            continue;
        }

        match ch {
            '\\' if in_string => {
                escape = true;
            }
            '"' => {
                in_string = !in_string;
            }
            '{' if !in_string => {
                if depth == 0 {
                    start_idx = Some(idx);
                }
                depth += 1;
            }
            '}' if !in_string => {
                if depth > 0 {
                    depth -= 1;
                    if depth == 0 {
                        if let Some(start) = start_idx {
                            let end = idx + ch.len_utf8();
                            return Some(content[start..end].to_string());
                        }
                    }
                }
            }
            _ => {}
        }
    }

    None
}

fn progress(message: impl AsRef<str>) {
    println!("[create-commit] {}", message.as_ref());
    let _ = io::stdout().flush();
}

#[cfg(debug_assertions)]
fn debug_log(message: impl AsRef<str>) {
    eprintln!("[create-commit:debug] {}", message.as_ref());
}

#[cfg(not(debug_assertions))]
fn debug_log(_message: impl AsRef<str>) {}

fn build_prompts(context: &CommitContext) -> Result<PromptBundle> {
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

    let schema_text = serde_json::to_string_pretty(
        schema
            .get("schema")
            .ok_or_else(|| anyhow!("prompt schema missing definition"))?,
    )?;

    let system_prompt = "You are an assistant that writes git commit plans. \
Respect project conventions, include ticket identifiers when appropriate, \
and prefer smaller, focused commits.";

    let context_json = serde_json::to_string_pretty(context)?;
    let summary_placeholder = "Describe the change";
    let detected_format = detect_ticket_format(&context.recent_commit_examples);
    let ticket_line = match (context.ticket_hint.as_deref(), context.ticket_required) {
        (Some(ticket), true) => {
            if let Some(format) = detected_format.as_ref() {
                let example = format.with_summary(ticket, summary_placeholder);
                format!(
                    "Every commit title must begin with \"{example}\" (replace the summary text with an accurate description). \
This convention is mandatory because recent commits use it."
                )
            } else {
                format!(
                    "Every commit title must begin with \"{ticket}: {summary_placeholder}\" (adjust the summary to fit). \
This convention is mandatory because recent commits use it."
                )
            }
        }
        (Some(ticket), false) => {
            if let Some(format) = detected_format.as_ref() {
                let example = format.with_summary(ticket, summary_placeholder);
                format!(
                    "Prefer starting the title with \"{example}\" when it suits the change, mirroring the existing history."
                )
            } else {
                format!(
                    "Prefer including the ticket identifier \"{ticket}\" in titles when it naturally fits."
                )
            }
        }
        (None, _) => String::new(),
    };

    let optional_ticket = if ticket_line.is_empty() {
        String::new()
    } else {
        format!("{ticket_line}\n")
    };

    let ticket_general = if context.ticket_hint.is_some() {
        let example = detected_format
            .as_ref()
            .map(|format| format.generic_example(summary_placeholder))
            .unwrap_or_else(|| format!("TICKET-123: {summary_placeholder}"));
        format!(
            "When a ticket reference is available, mirror the repository style (e.g. \"{example}\").\n"
        )
    } else {
        String::new()
    };

    let user_prompt = format!(
        "Generate a set of focused commits for the current working tree. \
Use the observed commit history to infer the preferred style. \
{optional_ticket}\
Return a JSON object that matches the provided schema exactly.\n\
Commit titles must summarize the changes and must not repeat the branch name verbatim.\n\
{ticket_general}\
- Use only files listed in the repository context; never invent new paths.\n\
- Each commit's `files` array must reference the relevant paths from the context.\n\
- Provide a `body` only when you have extra detail that is not already in the title; otherwise omit it.\n\
- Avoid duplicating the title text inside the body.\n\
Schema (JSON):\n{schema_text}\n\n\
Repository context (JSON):\n{context_json}\n\n\
Respond only with JSON representing the commits. Do not wrap the response in code fences, markdown, or extra commentary."
    );

    Ok(PromptBundle {
        system: system_prompt.to_string(),
        user: user_prompt,
        schema,
    })
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

fn detect_ticket_format(messages: &[String]) -> Option<TicketFormat> {
    messages
        .iter()
        .find_map(|message| parse_ticket_format(message))
}

fn parse_ticket_format(message: &str) -> Option<TicketFormat> {
    if let Some(format) = parse_bracketed_ticket(message) {
        return Some(format);
    }
    parse_leading_ticket(message)
}

fn parse_bracketed_ticket(message: &str) -> Option<TicketFormat> {
    if !message.starts_with('[') {
        return None;
    }
    let closing = message.find(']')?;
    let candidate = &message[1..closing];
    if !is_ticket_identifier(candidate) {
        return None;
    }
    let separator = separator_after(&message[closing + 1..]);
    Some(TicketFormat {
        prefix: "[".to_string(),
        suffix: "]".to_string(),
        separator,
    })
}

fn parse_leading_ticket(message: &str) -> Option<TicketFormat> {
    let mut end = 0;
    for (idx, ch) in message.char_indices() {
        if ch.is_ascii_uppercase() || ch.is_ascii_digit() || ch == '-' {
            end = idx + ch.len_utf8();
            continue;
        }
        break;
    }
    if end == 0 {
        return None;
    }
    let candidate = &message[..end];
    if !is_ticket_identifier(candidate) {
        return None;
    }
    let separator = separator_after(&message[end..]);
    Some(TicketFormat {
        prefix: String::new(),
        suffix: String::new(),
        separator,
    })
}

fn separator_after(segment: &str) -> String {
    if segment.is_empty() {
        return " ".to_string();
    }
    let mut end = 0;
    for (idx, ch) in segment.char_indices() {
        if ch.is_alphanumeric() || ch == '\n' || ch == '\r' {
            break;
        }
        end = idx + ch.len_utf8();
    }
    if end == 0 {
        " ".to_string()
    } else {
        segment[..end].to_string()
    }
}

fn is_ticket_identifier(candidate: &str) -> bool {
    static TICKET_ONLY_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^[A-Z]{2,}-\d{1,6}$").unwrap());
    TICKET_ONLY_RE.is_match(candidate)
}

fn sanitize_plan(plan: &mut AiPlan, context: &CommitContext) {
    let valid_files: HashSet<&str> = context
        .changes
        .iter()
        .map(|diff| diff.path.as_str())
        .collect();

    for commit in &mut plan.commits {
        let original = commit.files.len();
        commit.files.retain(|file| {
            let keep = valid_files.contains(file.as_str());
            if !keep {
                debug_log(&format!(
                    "Dropping unknown file '{}' from commit '{}'",
                    file, commit.title
                ));
            }
            keep
        });

        if original > 0 && commit.files.is_empty() {
            debug_log(&format!(
                "Commit '{}' has no valid files after filtering",
                commit.title
            ));
        }

        if let Some(body) = commit.body.as_ref() {
            if body.trim().eq_ignore_ascii_case(commit.title.trim()) {
                debug_log(&format!(
                    "Clearing body for commit '{}' because it duplicates the title",
                    commit.title
                ));
                commit.body = None;
            }
        }
    }

    plan.commits.retain(|commit| {
        let keep = !commit.files.is_empty();
        if !keep {
            debug_log(&format!(
                "Removing commit '{}' because it has no files after filtering",
                commit.title
            ));
        }
        keep
    });
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
