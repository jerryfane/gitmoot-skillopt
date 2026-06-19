# Judge-Prompt Optimization

In LLM-judged tasks the judge prompt is itself part of the system, not a fixed
constant. A vague or miscalibrated judge prompt produces a noisy reward signal,
and SkillOpt only ever climbs the reward it is given. Judge-prompt optimization
treats the judge prompt as a first-class, trainable artifact and tunes it the
same disciplined way SkillOpt tunes skills — but against a different objective.

## The Judge Prompt Is an Optimizable Artifact

The judge prompt is:

- **Per-`task_kind`.** Each `task_kind` (e.g. `vue_landing_page`) gets its own
  judge prompt, because "good" means different things for a landing page, a
  social post, or a math answer.
- **Versioned.** Each tuned judge prompt is a numbered version, so every score
  is traceable to the exact prompt that produced it and a regression can be
  rolled back.
- **Optimizable.** It is improved by an optimization loop rather than hand-edited
  once, the same way a skill document is.

The objective is the key difference. Skills are tuned to maximize the judge's
reward. The **judge prompt is tuned to maximize HUMAN AGREEMENT** — how often the
judge's verdict matches a held-out set of human labels — not to maximize any
skill's score.

## The Three Guardrails

Optimizing the thing that grades you is dangerous without discipline. Three
guardrails keep it honest:

1. **Independent ground-truth objective.** The judge prompt is tuned against
   held-out human labels (a human-agreement objective), never against the skill
   scores it produces. The judge is graded by humans; the skill is graded by the
   judge. The two objectives stay separate so the loop cannot quietly inflate its
   own reward.
2. **Frozen and alternated.** The judge is tuned on its own loop, then
   **frozen** while skills are optimized against it. Skill optimization and judge
   optimization never run at the same time; they alternate. During a normal skill
   run the judge does not change, so the reward signal is a fixed target.
3. **Hierarchical scope.** Judge prompts resolve in layers —
   **global → `task_kind` → per-prompt**. A global default is overridden by a
   `task_kind`-specific prompt, which can be further specialized per individual
   prompt. Tuning happens at the narrowest useful scope and inherits otherwise.

## Cost and Cadence Discipline

Judge-prompt optimization is deliberately cheap and rare:

- **Offline.** It runs out-of-band, not inside a skill run.
- **No new rollouts.** It reuses already-captured human-labeled data. It does not
  spend new target/agent executions to re-tune the judge.
- **Capped and infrequent.** Re-tuning is bounded and run rarely, not every step.

The steady state is the important part: **normal skill runs use the frozen judge
at today's cost.** A skill run pays for its rollouts and the frozen judge's
scoring passes — nothing more. **Re-tuning the judge inside every skill run is
the anti-pattern**: it would multiply cost, make the reward a moving target, and
break the independence between the judge objective and the skill objective.

## Where Gitmoot Carries It

Gitmoot owns the judge-prompt artifact and threads the active prompt and its
version through the training-package contract. The per-`task_kind` prompt and
version travel on the evaluator profile's judge config:

- `evaluator_profile.judge.config.judge_prompt_templates` — the per-`task_kind`
  judge-prompt templates.
- `evaluator_profile.judge.config.judge_prompt_version` — the version of the
  judge prompt in effect for the run.

Carrying the version alongside the prompt is what makes every captured score
attributable to a specific frozen judge, which is what the frozen-and-alternated
guardrail relies on. See
[Gitmoot MVP Workflow](gitmoot-mvp-workflow.md) for the surrounding export →
optimize → import → review flow.
