"""Validation gate — accept / reject candidate skills.

Analogous to validation-based early stopping and model selection in neural
network training: compares the candidate's score against the current and
best scores, then returns an accept/reject decision.

The trainer owns side-effects (cache lookup, rollout, printing, state
mutation).  This module is the pure decision function.

Metric selection
----------------
Four gate metrics are supported:

* ``"hard"`` (default, backward-compatible):
  Compare candidate vs current/best using *hard* exact-match accuracy.
* ``"soft"``:
  Compare using *soft* per-item score (F1 / partial credit / etc.).
  Use this when a small held-out selection set has too few items for
  hard accuracy to be sensitive to incremental skill improvements.
* ``"mixed"``:
  Compare using a weighted average ``(1 - w) * hard + w * soft``.
  ``w`` is configurable via ``mixed_weight`` (default ``0.5``).
* ``"human_agreement"`` (#345 Phase 2, judge-prompt optimization):
  Compare using the fraction of a labeled held-out set on which the
  judge's accept/reject verdict matches the human verdict. This is the
  objective used when tuning the *judge prompt* (not the skill) — see
  :func:`score_human_agreement` and :func:`evaluate_human_agreement_gate`.
  It does not consume rollout hard/soft scores; the caller supplies the
  agreement fractions directly.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

GateAction = Literal["accept_new_best", "accept", "reject"]
GateMetric = Literal["hard", "soft", "mixed", "human_agreement"]

# Canonical human verdict labels accepted in a held-out labeled set.
_PROMOTE_LABELS = frozenset({"promote", "accept", "yes", "approve", "approved", "pass", "true", "1"})
_REJECT_LABELS = frozenset({"reject", "deny", "no", "fail", "false", "0"})


@dataclass(frozen=True)
class GateResult:
    """Immutable outcome of the validation gate."""

    action: GateAction
    current_skill: str
    current_score: float
    best_skill: str
    best_score: float
    best_step: int


@dataclass(frozen=True)
class GateBlock:
    """Structured reason a validation gate cannot make a decision."""

    blocker: str
    items: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"blocker": self.blocker, "items": self.items}


def find_gate_block(results: list[object]) -> GateBlock | None:
    """Return a block report when required gate results are not scored."""
    items = [_blocking_item(result) for result in results if _is_gate_unscored(result)]
    if not items:
        return None
    return GateBlock(blocker=_primary_blocker(items), items=items)


def require_scored_gate_results(results: list[object]) -> None:
    """Raise if a gate would be comparing incomplete evaluation results."""
    block = find_gate_block(results)
    if block is not None:
        raise ValueError(f"blocked:{block.blocker}")


def select_gate_score(
    hard: float,
    soft: float,
    metric: GateMetric = "hard",
    mixed_weight: float = 0.5,
) -> float:
    """Project (hard, soft) onto a single comparison metric.

    Parameters
    ----------
    hard, soft
        Aggregate hard / soft scores from a rollout batch (both 0..1).
    metric
        Which metric to compare on.
    mixed_weight
        For ``"mixed"``: weight given to ``soft``. Must be in ``[0, 1]``.
        Ignored for ``"hard"`` / ``"soft"``.
    """
    if metric == "hard":
        return float(hard)
    if metric == "soft":
        return float(soft)
    if metric == "mixed":
        w = max(0.0, min(1.0, float(mixed_weight)))
        return (1.0 - w) * float(hard) + w * float(soft)
    if metric == "human_agreement":
        # The human-agreement objective is computed by ``score_human_agreement``
        # against a labeled held-out set, not projected from rollout hard/soft.
        # When this projection is reached, ``hard`` already carries the
        # pre-computed agreement fraction (the judge-tuning path passes it as
        # ``cand_hard``), so we surface it directly.
        return float(hard)
    raise ValueError(
        f"unknown gate metric {metric!r}; expected 'hard', 'soft', 'mixed', or 'human_agreement'"
    )


def _is_gate_unscored(result: object) -> bool:
    status = str(_result_field(result, "score_status", "") or "").strip().lower()
    return status == "unscored" or _result_field(result, "hard", None) is None or _result_field(result, "soft", None) is None


def _blocking_item(result: object) -> dict[str, Any]:
    blocker = _result_blocker(result)
    item = {
        "id": str(_result_field(result, "id", "unknown")),
        "blocker": blocker,
        "score_status": str(_result_field(result, "score_status", "unscored") or "unscored"),
        "target_status": str(_result_field(result, "target_status", "") or ""),
        "evaluator_status": str(_result_field(result, "evaluator_status", "") or ""),
        "target_trace_path": str(_result_field(result, "target_trace_path", "") or ""),
        "evaluator_trace_path": str(_result_field(result, "evaluator_trace_path", "") or ""),
        "fail_reason": str(_result_field(result, "fail_reason", "") or ""),
    }
    return {key: value for key, value in item.items() if value != ""}


def _result_blocker(result: object) -> str:
    explicit = str(_result_field(result, "blocker", "") or "").strip()
    if explicit:
        return explicit
    target_status = str(_result_field(result, "target_status", "") or "").strip().lower()
    evaluator_status = str(_result_field(result, "evaluator_status", "") or "").strip().lower()
    if target_status == "failed":
        return "target_rollout_failed"
    if evaluator_status == "not_run":
        return "evaluator_not_run"
    if evaluator_status == "failed":
        return "evaluator_failed"
    if _result_field(result, "hard", None) is None or _result_field(result, "soft", None) is None:
        return "invalid_evaluator_score"
    return "unscored"


def _primary_blocker(items: list[dict[str, Any]]) -> str:
    priority = [
        "target_rollout_failed",
        "evaluator_not_run",
        "evaluator_failed",
        "invalid_evaluator_score",
    ]
    blockers = {str(item.get("blocker") or "") for item in items}
    for blocker in priority:
        if blocker in blockers:
            return blocker
    return next((blocker for blocker in blockers if blocker), "unscored")


def _result_field(result: object, key: str, default: Any) -> Any:
    if hasattr(result, key):
        return getattr(result, key)
    if isinstance(result, dict):
        return result.get(key, default)
    return default


def evaluate_gate(
    candidate_skill: str,
    cand_hard: float,
    current_skill: str,
    current_score: float,
    best_skill: str,
    best_score: float,
    best_step: int,
    global_step: int,
    *,
    cand_soft: float = 0.0,
    metric: GateMetric = "hard",
    mixed_weight: float = 0.5,
) -> GateResult:
    """Pure gate decision: compare candidate score to current/best.

    Parameters
    ----------
    candidate_skill
        The candidate skill content being evaluated.
    cand_hard, cand_soft
        Aggregate hard / soft scores of the candidate on the selection set.
    current_skill, current_score
        The currently-active skill and its *metric-space* score.
    best_skill, best_score, best_step
        The best-so-far skill, its *metric-space* score, and the step
        at which it was accepted.
    global_step
        Current global training step (recorded if a new best is accepted).
    cand_soft
        Soft score of the candidate; only consulted when ``metric != "hard"``.
        Defaults to ``0.0`` for backward compatibility with callers that
        previously passed only ``cand_hard``.
    metric
        Which metric to compare on. Defaults to ``"hard"`` to preserve
        the original gate behavior.
    mixed_weight
        Weight on ``soft`` when ``metric == "mixed"``.

    Returns
    -------
    GateResult
        Updated state; the caller decides what to do with it (print,
        mutate trainer state, log, etc.).
    """
    cand_score = select_gate_score(cand_hard, cand_soft, metric, mixed_weight)

    if cand_score > current_score:
        if cand_score > best_score:
            return GateResult(
                action="accept_new_best",
                current_skill=candidate_skill,
                current_score=cand_score,
                best_skill=candidate_skill,
                best_score=cand_score,
                best_step=global_step,
            )
        return GateResult(
            action="accept",
            current_skill=candidate_skill,
            current_score=cand_score,
            best_skill=best_skill,
            best_score=best_score,
            best_step=best_step,
        )
    return GateResult(
        action="reject",
        current_skill=current_skill,
        current_score=current_score,
        best_skill=best_skill,
        best_score=best_score,
        best_step=best_step,
    )


# ── Human-agreement objective (#345 Phase 2, judge-prompt optimization) ─────


def normalize_human_verdict(verdict: Any) -> bool:
    """Map a human verdict label to a boolean ``promote`` decision.

    Accepts the canonical ``"promote"`` / ``"reject"`` labels (and a few common
    synonyms / booleans). Raises ``ValueError`` for anything unrecognised so a
    mislabeled held-out set fails loudly rather than silently scoring wrong.
    """
    if isinstance(verdict, bool):
        return verdict
    token = str(verdict).strip().lower()
    if token in _PROMOTE_LABELS:
        return True
    if token in _REJECT_LABELS:
        return False
    raise ValueError(
        f"unknown human verdict {verdict!r}; expected one of "
        f"{sorted(_PROMOTE_LABELS)} or {sorted(_REJECT_LABELS)}"
    )


def score_human_agreement(
    labeled_items: Iterable[Mapping[str, Any]],
    judge_verdicts: Mapping[str, Any] | Iterable[Any],
    *,
    id_key: str = "id",
    human_key: str = "human_verdict",
) -> float:
    """Fraction of a labeled held-out set where judge verdict == human verdict.

    Parameters
    ----------
    labeled_items
        The held-out human-labeled set. Each item is a mapping carrying at least
        a human verdict under *human_key* (and, for the mapping form of
        *judge_verdicts*, an id under *id_key*).
    judge_verdicts
        Either a mapping ``{item_id: judge_verdict}`` or a positional iterable of
        judge verdicts aligned with *labeled_items*. Each verdict is normalised
        via :func:`normalize_human_verdict` (accept/promote → ``True``).
    id_key, human_key
        Field names used to read the item id and the human verdict.

    Returns
    -------
    float
        Agreement fraction in ``[0, 1]``. An empty labeled set scores ``0.0``.
    """
    items = list(labeled_items)
    if not items:
        return 0.0

    if isinstance(judge_verdicts, Mapping):
        def _judge_for(idx: int, item: Mapping[str, Any]) -> Any:
            return judge_verdicts.get(str(item.get(id_key)))
    else:
        verdict_list = list(judge_verdicts)

        def _judge_for(idx: int, item: Mapping[str, Any]) -> Any:
            return verdict_list[idx] if idx < len(verdict_list) else None

    agree = 0
    for idx, item in enumerate(items):
        human = normalize_human_verdict(item[human_key])
        raw_judge = _judge_for(idx, item)
        if raw_judge is None:
            # A missing judge verdict is treated as disagreement (the judge
            # failed to render a verdict on a labeled item).
            continue
        if normalize_human_verdict(raw_judge) == human:
            agree += 1
    return agree / len(items)


def evaluate_human_agreement_gate(
    candidate_prompt: str,
    cand_agreement: float,
    current_prompt: str,
    current_agreement: float,
    best_prompt: str,
    best_agreement: float,
    best_step: int,
    global_step: int,
) -> GateResult:
    """Gate a candidate *judge prompt* on held-out human agreement.

    Mirrors :func:`evaluate_gate`'s accept-if-improves logic, but the comparison
    quantity is the held-out human-agreement fraction rather than rollout
    hard/soft. Returns a :class:`GateResult` whose ``*_skill`` fields carry the
    judge-prompt text (the optimized artifact in this mode).
    """
    return evaluate_gate(
        candidate_skill=candidate_prompt,
        cand_hard=float(cand_agreement),
        current_skill=current_prompt,
        current_score=float(current_agreement),
        best_skill=best_prompt,
        best_score=float(best_agreement),
        best_step=best_step,
        global_step=global_step,
        metric="human_agreement",
    )
