# CLI Reference

## Gitmoot Optimizer

```bash
gitmoot-skillopt optimize \
  --training-package training.json \
  --artifact-root ~/.gitmoot/evals/blobs \
  --out-root outputs/run-1 \
  --candidate-output outputs/run-1/candidate.json
```

### Arguments

| Argument | Description |
|---|---|
| `--training-package` | Gitmoot SkillOpt training package from `gitmoot skillopt export` |
| `--artifact-root` | Gitmoot blob root, usually `~/.gitmoot/evals/blobs` |
| `--out-root` | Optimizer output directory |
| `--candidate-output` | Candidate package JSON path to import back into Gitmoot |
| `--dry-run` | Emit deterministic fixture output without trainer/model calls |

### Judge-prompt optimization (#345 Phase 2)

Instead of tuning the skill, `optimize` can tune the **judge prompt** against a
held-out human-labeled set — the freeze-and-alternate counterpart to skill
optimization, gated on held-out human agreement. See
[Judge-prompt optimization](../guide/judge-prompt-optimization.md) for the
concept and guardrails.

```bash
gitmoot-skillopt optimize \
  --training-package training.json \
  --artifact-root ~/.gitmoot/evals/blobs \
  --out-root outputs/judge-1 \
  --candidate-output outputs/judge-1/judge_candidate.json \
  --judge-prompt-optimization \
  --judge-human-labeled-path held-out-labels.json \
  --evaluator-backend codex --optimizer-backend codex
```

| Argument | Description |
|---|---|
| `--judge-prompt-optimization` | Tune the judge prompt (not the skill); forces the `human_agreement` gate |
| `--judge-human-labeled-path` | JSON/JSONL held-out set of `{id?, human_verdict, artifact, task_kind?}` (required) |
| `--judge-prompt-init` | Initial judge prompt text; defaults to the built-in verdict judge prompt |
| `--judge-prompt-version` | Base judge-prompt version tag stamped on accepted variants (default `v0`) |
| `--judge-edit-budget` | Maximum edits per judge-prompt reflect pass (default `4`) |

Output is a judge-candidate package (`kind: gitmoot-skillopt-judge-candidate`)
whose `variants` map carries, per `task_kind`, the baseline vs. best held-out
agreement, whether the candidate was `accepted`, the `judge_prompt_version`, and
the accepted `best_prompt`.

### Contract Smoke

```bash
gitmoot-skillopt optimize \
  --training-package examples/gitmoot/mvp-fixture/training.json \
  --artifact-root examples/gitmoot/mvp-fixture/blobs \
  --out-root /tmp/gitmoot-skillopt-smoke \
  --candidate-output /tmp/gitmoot-skillopt-smoke/candidate.json \
  --dry-run
```

Import the generated candidate with:

```bash
gitmoot skillopt import \
  --file /tmp/gitmoot-skillopt-smoke/candidate.json \
  --artifact-dir /tmp/gitmoot-skillopt-smoke/artifacts
```

Or run the full temp-home Gitmoot import smoke:

```bash
.venv/bin/python scripts/gitmoot_contract_smoke.py --gitmoot-bin /path/to/gitmoot
```

## Training

```bash
python scripts/train.py --config <config.yaml> [overrides...]
```

### Arguments

| Argument | Description |
|---|---|
| `--config` | Path to YAML config file (required) |
| `key=value` | Override any config parameter |

### Examples

```bash
# Basic training
python scripts/train.py --config configs/searchqa/default.yaml

# With overrides
python scripts/train.py \
  --config configs/searchqa/default.yaml \
  --cfg-options optimizer.learning_rate=16 optimizer.lr_scheduler=linear

# With custom initial skill
python scripts/train.py \
  --config configs/searchqa/default.yaml \
  --cfg-options env.skill_init=skills/my_seed.md
```

## Evaluation

```bash
python scripts/eval_only.py --config <config.yaml> --skill <skill.md>
```

### Arguments

| Argument | Description |
|---|---|
| `--config` | Path to YAML config file (required) |
| `--skill` | Path to skill document to evaluate (required) |
| `--split` | Evaluation split: `test` (default), `valid`, `train` |

### Examples

```bash
# Evaluate best skill on test set
python scripts/eval_only.py \
  --config configs/searchqa/default.yaml \
  --skill outputs/searchqa/run_001/skills/best_skill.md

# Evaluate on validation set
python scripts/eval_only.py \
  --config configs/searchqa/default.yaml \
  --skill outputs/searchqa/run_001/skills/best_skill.md \
  --split valid
```

## WebUI

```bash
python -m skillopt_webui.app [--port PORT] [--share]
```

| Argument | Default | Description |
|---|---|---|
| `--port` | 7860 | Port number |
| `--share` | false | Create public Gradio link |
