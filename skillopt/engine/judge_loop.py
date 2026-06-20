"""Judge-prompt optimization loop orchestration (#345 Phase 2).

Pure, model-agnostic glue that turns the judge-tuning building blocks into one
end-to-end pass. Given a baseline judge prompt and a held-out human-labeled set,
it runs a global pass plus one pass per ``task_kind`` present in the data; each
pass produces baseline judge verdicts, asks an injected *proposer* for a
candidate judge prompt, scores the candidate's verdicts, and gates
accept-if-improves via :func:`run_judge_prompt_optimization`.

Model calls live behind two injected callables so the loop is unit-testable with
fakes (no codex / optimizer calls):

- ``judge_runner(prompt, items) -> {item_id: verdict}`` — runs a judge prompt
  over the held-out items and returns its accept/reject verdicts.
- ``proposer(prompt, items, verdicts, task_kind, edit_budget) -> str | None`` —
  proposes a candidate judge prompt (already patched) or ``None``.

The real codex-backed wiring is supplied by
``gitmoot_skillopt.judge_optimize.run_judge_optimize``.
"""

from __future__ import annotations

from typing import Any, Callable

from skillopt.engine.trainer import JudgePromptCandidate, run_judge_prompt_optimization

JudgeRunner = Callable[[str, list], dict[str, Any]]
JudgeProposer = Callable[[str, list, dict[str, Any], str, int], "str | None"]

JUDGE_CANDIDATE_KIND = "gitmoot-skillopt-judge-candidate"


def _item_task_kind(item: Any) -> str:
    raw = item.get("task_kind") if isinstance(item, dict) else getattr(item, "task_kind", "")
    return str(raw or "").strip()


def _item_id(item: Any) -> str:
    # Mirror the id derivation used by run_judge_prompt_over_labeled and
    # score_human_agreement so cached verdicts align with the labels.
    raw = item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
    return str(raw)


def optimize_judge_prompts(
    baseline_prompt: str,
    labeled_items: list,
    *,
    judge_runner: JudgeRunner,
    proposer: JudgeProposer,
    base_version: str = "v0",
    edit_budget: int = 4,
) -> dict[str, Any]:
    """Run one judge-prompt optimization pass globally and per ``task_kind``.

    Returns a structured judge-candidate package: a ``variants`` map keyed by
    ``task_kind`` (``"_global"`` for the all-items pass), each carrying the
    baseline/best agreement, the accepted judge prompt + version, and the gate
    history. A variant is ``accepted`` only when a candidate strictly improved
    held-out human agreement over the frozen baseline judge.

    Verdicts are memoized by ``(prompt, item_id)`` so an item judged in the
    global pass is not re-judged (a real model call) when it reappears in its
    per-task_kind pass — important because the judge runner is codex-backed.
    """
    baseline_prompt = (baseline_prompt or "").strip()
    tagged = [(item, _item_task_kind(item)) for item in labeled_items]
    kinds = sorted({kind for _, kind in tagged if kind})
    passes = [""] + kinds  # "" = global pass over every item

    verdict_cache: dict[tuple[str, str], Any] = {}

    def run_cached(prompt: str, items: list) -> dict[str, Any]:
        missing = [item for item in items if (prompt, _item_id(item)) not in verdict_cache]
        if missing:
            fresh = judge_runner(prompt, missing)
            for item in missing:
                item_id = _item_id(item)
                verdict_cache[(prompt, item_id)] = fresh.get(item_id)
        return {_item_id(item): verdict_cache[(prompt, _item_id(item))] for item in items}

    variants: dict[str, Any] = {}
    for task_kind in passes:
        # The global pass takes every item; a per-kind pass also includes untagged
        # items so it still sees shared context (matching the task_kind filter in
        # run_judge_prompt_optimization).
        items = [item for item, kind in tagged if task_kind == "" or kind in ("", task_kind)]
        if not items:
            continue

        baseline_verdicts = run_cached(baseline_prompt, items)
        candidates: list[JudgePromptCandidate] = []
        candidate_prompt = proposer(baseline_prompt, items, baseline_verdicts, task_kind, edit_budget)
        candidate_prompt = (candidate_prompt or "").strip()
        if candidate_prompt and candidate_prompt != baseline_prompt:
            candidate_verdicts = run_cached(candidate_prompt, items)
            candidates.append(
                JudgePromptCandidate(
                    prompt=candidate_prompt,
                    verdicts=candidate_verdicts,
                    origin=f"judge_reflect_{task_kind or 'global'}",
                    task_kind=task_kind,
                )
            )

        result = run_judge_prompt_optimization(
            baseline_prompt,
            candidates,
            items,
            baseline_verdicts=baseline_verdicts,
            base_version=base_version,
            task_kind=task_kind,
        )
        variants[task_kind or "_global"] = {
            "task_kind": task_kind,
            "n_items": len(items),
            "baseline_agreement": result.baseline_agreement,
            "best_agreement": result.best_agreement,
            "best_origin": result.best_origin,
            "judge_prompt_version": result.judge_prompt_version,
            "accepted": result.best_origin != "baseline_judge",
            "best_prompt": result.best_prompt,
            "history": result.history,
        }

    return {
        "kind": JUDGE_CANDIDATE_KIND,
        "contract_version": 1,
        "judge_prompt_version_base": base_version,
        "n_labeled": len(labeled_items),
        "variants": variants,
    }
