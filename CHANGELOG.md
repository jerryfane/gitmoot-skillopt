# Changelog

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
