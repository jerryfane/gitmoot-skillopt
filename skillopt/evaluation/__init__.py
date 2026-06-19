"""ReflACT Evaluation -- candidate skill validation and model selection.

Analogous to validation-based early stopping and model selection in neural
network training: evaluates candidate skills on held-out selection sets and
decides whether to accept or reject proposed updates.
"""
from skillopt.evaluation.gate import (  # noqa: F401
    GateAction,
    GateMetric,
    GateResult,
    evaluate_gate,
    evaluate_human_agreement_gate,
    normalize_human_verdict,
    score_human_agreement,
    select_gate_score,
)
