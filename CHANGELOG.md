# Changelog

## v0.4.1

- **Fix: accept the `kimi` runtime in template `runtime_compatibility`.** gitmoot
  added Kimi Code as a supported runtime and its default agent-template scaffold
  lists `kimi`, but the contract validator's `_VALID_RUNTIMES` whitelist only had
  `codex`/`claude`/`shell` — so the optimizer crashed with
  `ContractError: template frontmatter has invalid runtime_compatibility 'kimi'`
  on any real gitmoot template that declares kimi compatibility. Added `kimi` to
  the whitelist. Found by a live end-to-end codex optimization run.

## v0.4.0

- **Trajectory digest for the judge** (#348 Phase 1). A budgeted, secrets-redacted
  `TrajectoryDigest` (reusing the previously-unused `parse_codex_raw`) is injected
  into the gitmoot judge prompt as a `## Process Summary` section, so the judge can
  see *what the agent did* (tool calls, commands, tests + outcomes, files touched,
  retries/errors) — not only the final artifact. **Off by default** (judge prompt
  byte-identical until enabled via `evaluator_config`), exec-backends only,
  fail-closed on any missing/garbage trace, no contract bump, no raw chain-of-thought.
- **Live-pairwise evaluation mode** (#77a). New opt-in `live-pairwise` mode reruns
  both the promoted and candidate templates live over the validation set and emits a
  **blinded paired review packet** (the secret A/B map kept in a separate artifact) +
  per-item token/cost/failure artifacts. No optimizer rewrite, no score-gate, manual
  promotion, additive contract; the default saved-baseline path is unchanged. The Go
  side ingests the blinded packet into canonical feedback events (gitmoot #508).

## v0.3.1

- Default-on deterministic **hard-verifier floor** in the gitmoot evaluator
  (#346). `evaluate_response` now runs `_run_hard_verifiers` before the LLM
  judge and short-circuits with a `hard=0` failure packet, shrinking the
  gameable surface and giving the optimizer crisp, actionable failures.
  - Built-in checks keyed by `task_kind`: **`agent_template`** (valid YAML
    frontmatter, required "update format" section, fenced ```json blocks must
    parse, no secrets / absolute paths / remote-mutation·auto-promote language,
    size bounds) and **`package`** (valid JSON, strict `contract_version == 1`).
  - Also honors declared `evaluator_profile.checks`; unknown task kinds with no
    declared checks are a no-op (behavior unchanged).
  - **Fail-closed:** every check runs under a guard that converts any crash
    (e.g. `RecursionError` from adversarial deeply-nested YAML/JSON) into a
    clean `hard=0` failure instead of taking down the evaluator.

## v0.3.0

- Wire judge-prompt optimization end-to-end (#345 Phase 2, via #70 + #71). The
  optimizer can now tune the **judge prompt** against a held-out human-labeled
  set instead of the skill — the freeze-and-alternate counterpart to skill
  optimization, gated on held-out human agreement.
  - New `optimize` flags: `--judge-prompt-optimization`,
    `--judge-human-labeled-path`, `--judge-prompt-init`,
    `--judge-prompt-version`, `--judge-edit-budget`.
  - Runs a global pass plus one per `task_kind`; reflects a candidate judge
    prompt, scores its verdicts against the human-labeled set, and accepts only
    when held-out agreement improves (`human_agreement` gate). Emits a
    judge-candidate package with per-`task_kind` best prompt + version.
  - Judge verdicts are memoized by `(prompt, item_id)` so an item judged in the
    global pass is not re-judged in its per-`task_kind` pass.

## v0.2.0b2

- Fix Claude Code structured-output compatibility by using the public
  `--json-schema` flag, with legacy `--schema` fallback when advertised by the
  installed Claude CLI.
- Fail early with actionable guidance when a Claude CLI has no schema-backed
  structured-output flag.
