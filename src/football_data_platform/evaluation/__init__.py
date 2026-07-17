"""Probability evaluation and model governance."""

from football_data_platform.evaluation.governance import (
    PAIRED_BOOTSTRAP_RESAMPLES,
    PAIRED_BOOTSTRAP_SEED,
    PAIRED_BOOTSTRAP_VERSION,
    ChallengerEvidence,
    EvaluationComparison,
    EvaluationPairReference,
    EvaluationSubgroup,
    PairedEvaluation,
    PromotionDecision,
    PromotionPolicy,
    SubgroupDiagnostic,
    aggregate_challenger_evidence,
    assess_promotion,
)
from football_data_platform.evaluation.metrics import EvaluationRecord

__all__ = [
    "ChallengerEvidence",
    "EvaluationComparison",
    "EvaluationPairReference",
    "EvaluationRecord",
    "EvaluationSubgroup",
    "PAIRED_BOOTSTRAP_RESAMPLES",
    "PAIRED_BOOTSTRAP_SEED",
    "PAIRED_BOOTSTRAP_VERSION",
    "PairedEvaluation",
    "PromotionDecision",
    "PromotionPolicy",
    "SubgroupDiagnostic",
    "aggregate_challenger_evidence",
    "assess_promotion",
]
