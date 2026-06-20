"""Judge-prompt optimization entrypoint (#345 Phase 2).

Wires the codex-backed judge runner + reflect proposer into the pure
:func:`skillopt.engine.judge_loop.optimize_judge_prompts` loop, runs the judge
preflight, and writes a judge-candidate package. This is the end-to-end path the
``optimize --judge-prompt-optimization`` CLI invokes — the freeze-and-alternate
counterpart to skill optimization, gated on held-out human agreement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gitmoot_skillopt.contracts import TrainingPackage
from gitmoot_skillopt.preflight import run_judge_preflight
from skillopt.datasets import load_human_labeled_set
from skillopt.engine.judge_loop import optimize_judge_prompts
from skillopt.envs.gitmoot.evaluator import (
    DEFAULT_JUDGE_VERDICT_PROMPT,
    run_judge_prompt_over_labeled,
)
from skillopt.gradient.reflect import propose_judge_prompt_edit
from skillopt.optimizer.skill import apply_patch


def run_judge_optimize(
    *,
    training_package: str,
    out_root: str,
    candidate_output: str,
    human_labeled_path: str,
    judge_prompt_init: str = "",
    judge_prompt_version: str = "v0",
    edit_budget: int = 4,
    optimizer_backend: str = "openai_chat",
    optimizer_model: str = "",
    evaluator_id: str = "",
    evaluator_backend: str = "",
    evaluator_model: str = "",
) -> dict[str, Any]:
    from .optimize import _require_file

    package = TrainingPackage.load(_require_file(training_package, "training package"))
    labeled_path = _require_file(human_labeled_path, "judge human-labeled set")
    labeled = load_human_labeled_set(str(labeled_path))
    if not labeled:
        raise ValueError(
            "judge-prompt optimization requires a non-empty held-out human-labeled set "
            "(--judge-human-labeled-path)"
        )

    preflight = run_judge_preflight(
        package,
        optimizer_backend=optimizer_backend,
        optimizer_model=optimizer_model,
        evaluator_id=evaluator_id,
        evaluator_backend=evaluator_backend,
        evaluator_model=evaluator_model,
    )
    evaluator_config = preflight.evaluator_config
    baseline_prompt = (judge_prompt_init or "").strip() or DEFAULT_JUDGE_VERDICT_PROMPT

    def judge_runner(prompt: str, items: list) -> dict[str, Any]:
        return run_judge_prompt_over_labeled(prompt, items, evaluator_config)

    def proposer(prompt: str, items: list, verdicts: dict[str, Any], task_kind: str, budget: int) -> str | None:
        proposal = propose_judge_prompt_edit(prompt, items, verdicts, task_kind=task_kind, edit_budget=budget)
        if not proposal or not isinstance(proposal.get("patch"), dict):
            return None
        try:
            return apply_patch(prompt, proposal["patch"])
        except Exception:  # noqa: BLE001
            return None

    result = optimize_judge_prompts(
        baseline_prompt,
        labeled,
        judge_runner=judge_runner,
        proposer=proposer,
        base_version=judge_prompt_version,
        edit_budget=edit_budget,
    )

    out_root_path = Path(out_root).expanduser()
    out_root_path.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2, ensure_ascii=False, default=str)
    (out_root_path / "judge_optimization.json").write_text(payload, encoding="utf-8")
    candidate_path = Path(candidate_output).expanduser()
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(payload, encoding="utf-8")
    return result
