import pytest
from greycloak.models import DivergenceJudgment, Severity
from greycloak.modules import aggregate_judgments


def _j(score, diverged=None, **kw):
    return DivergenceJudgment(
        diverged=(score >= 0.5) if diverged is None else diverged,
        divergence_score=score, **kw)


def test_empty_raises():
    with pytest.raises(ValueError):
        aggregate_judgments([], Severity.HIGH)


def test_single_vote_is_identity():
    j = _j(0.7)
    assert aggregate_judgments([j], Severity.HIGH) is j


def test_median_score_and_majority_vote():
    agg = aggregate_judgments([_j(0.2, False), _j(0.8, True), _j(0.9, True)], Severity.HIGH)
    assert agg.divergence_score == 0.8
    assert agg.diverged is True


def test_confidence_reflects_agreement():
    tight = aggregate_judgments([_j(0.80), _j(0.82), _j(0.79)], Severity.HIGH)
    split = aggregate_judgments([_j(0.0, False), _j(1.0, True), _j(0.5, True)], Severity.HIGH)
    assert tight.confidence > split.confidence


def test_fields_from_nearest_median():
    votes = [_j(0.1, False, rationale="low"), _j(0.6, True, rationale="mid"),
             _j(0.95, True, rationale="high")]
    agg = aggregate_judgments(votes, Severity.HIGH)  # median 0.6
    assert agg.rationale == "mid"


def test_even_k_score_and_diverged_are_consistent():
    # split 2-vote: median 0.65 >= 0.5 must NOT be reported as not-diverged
    agg = aggregate_judgments([_j(0.4, False), _j(0.9, True)], Severity.HIGH)
    assert agg.divergence_score == 0.65
    assert agg.diverged is True


def test_even_k_boundary_median_half():
    agg = aggregate_judgments([_j(0.3, False), _j(0.4, False), _j(0.6, True), _j(0.7, True)],
                              Severity.HIGH)
    assert agg.divergence_score == 0.5
    assert agg.diverged is True
