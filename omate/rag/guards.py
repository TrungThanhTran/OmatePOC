"""
Hallucination guards for clinical report generation.

Three-layer defense:
  Layer 1: Citation grounding — every claim must have a [SOURCE: ...] tag
  Layer 2: SelfCheckGPT — NLI consistency across multiple samples
  Layer 3: Hard refusal — return None if confidence < threshold

Reference: Manakul et al. (2023) SelfCheckGPT, EMNLP 2023
           https://arxiv.org/abs/2303.08896
"""

import re
from dataclasses import dataclass
from typing import Callable


@dataclass
class GuardResult:
    passed: bool
    confidence: float           # 0–1
    citation_score: float       # fraction of claims with valid citations
    consistency_score: float    # SelfCheck NLI consistency (0=bad, 1=good)
    failed_citations: list[str]
    flagged_sentences: list[str]
    reason: str                 # human-readable explanation


CITATION_PATTERN = re.compile(
    r'\[SOURCE:\s*resource_id=([^\],\]]+)(?:,\s*[^\]]+)?\]'
)

CLAIM_SPLIT_PATTERN = re.compile(r'(?<=[.!?])\s+')


# ---------------------------------------------------------------------------
# Layer 1: Citation grounding
# ---------------------------------------------------------------------------

def extract_citations(text: str) -> list[str]:
    """Extract all resource IDs from [SOURCE: resource_id=XXX] tags."""
    return [m.group(1).strip() for m in CITATION_PATTERN.finditer(text)]


def split_into_sentences(text: str) -> list[str]:
    sentences = CLAIM_SPLIT_PATTERN.split(text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def validate_citations(report: str, resource_exists_fn: Callable[[str], bool]
                        ) -> tuple[list[str], list[str]]:
    """
    Validate all [SOURCE: resource_id=XXX] tags in the report.

    Args:
        report: generated report text
        resource_exists_fn: callable that returns True if resource_id exists

    Returns:
        (valid_citations, failed_citations)
    """
    all_citations = extract_citations(report)
    failed = [c for c in all_citations if not resource_exists_fn(c)]
    valid = [c for c in all_citations if resource_exists_fn(c)]
    return valid, failed


def compute_citation_score(report: str, resource_exists_fn: Callable) -> float:
    """
    Fraction of factual sentences that have valid citations.
    Sentences without [SOURCE] tags are counted as uncited.
    """
    sentences = split_into_sentences(report)
    if not sentences:
        return 0.0

    cited_count = 0
    for sentence in sentences:
        cites = extract_citations(sentence)
        if cites and all(resource_exists_fn(c) for c in cites):
            cited_count += 1

    return round(cited_count / len(sentences), 3)


# ---------------------------------------------------------------------------
# Layer 2: SelfCheckGPT (lightweight NLI-based)
# ---------------------------------------------------------------------------

def simple_nli_consistency(sentence: str, passages: list[str]) -> float:
    """
    Simplified NLI consistency check.

    Real SelfCheckGPT uses a fine-tuned NLI model (mDeBERTa-v3-base).
    This POC uses token overlap as a proxy — good enough to demonstrate
    the concept. Swap with SelfCheckNLI from the selfcheckgpt library
    for production.

    Returns: hallucination probability (higher = more likely hallucinated)
    """
    if not passages:
        return 0.5

    sentence_tokens = set(sentence.lower().split())
    sentence_tokens -= {"the", "a", "an", "is", "are", "was", "were",
                         "in", "of", "to", "and", "or", "for", "with"}

    if not sentence_tokens:
        return 0.0

    # Check how many passages contain similar content
    agreement_count = 0
    for passage in passages:
        passage_tokens = set(passage.lower().split())
        overlap = len(sentence_tokens & passage_tokens) / len(sentence_tokens)
        if overlap >= 0.3:
            agreement_count += 1

    agreement_ratio = agreement_count / len(passages)
    # Convert agreement → hallucination probability (inverted)
    hallucination_prob = 1.0 - agreement_ratio
    return round(hallucination_prob, 3)


def selfcheck_report(primary: str, variations: list[str],
                      threshold: float = 0.30) -> tuple[float, list[str]]:
    """
    Run SelfCheckGPT consistency across multiple samples.

    Args:
        primary: primary report (temperature=0)
        variations: list of variation reports (temperature>0)
        threshold: sentences above this hallucination probability are flagged

    Returns:
        (overall_risk_score, flagged_sentences)
    """
    if not variations:
        return 0.0, []

    sentences = split_into_sentences(primary)
    scores = []
    flagged = []

    for sentence in sentences:
        score = simple_nli_consistency(sentence, variations)
        scores.append(score)
        if score > threshold:
            flagged.append(sentence)

    overall_risk = max(scores) if scores else 0.0
    return round(overall_risk, 3), flagged


# ---------------------------------------------------------------------------
# Layer 3: Combined confidence + hard refusal
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = float(__import__("os").getenv("CONFIDENCE_THRESHOLD", "0.80"))


def compute_confidence(citation_score: float, consistency_score: float) -> float:
    """Weighted combination of citation quality and NLI consistency."""
    return round(0.5 * citation_score + 0.5 * consistency_score, 3)


def run_hallucination_guards(
    primary_report: str,
    generate_fn: Callable[[float], str],
    resource_exists_fn: Callable[[str], bool],
    n_samples: int = 4,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> GuardResult:
    """
    Run all three hallucination guard layers.

    Args:
        primary_report: the primary generated report (temperature=0)
        generate_fn: callable(temperature) -> report_text for sampling
        resource_exists_fn: callable(resource_id) -> bool
        n_samples: number of variations for SelfCheckGPT
        confidence_threshold: minimum confidence to pass

    Returns:
        GuardResult with pass/fail and detailed scores
    """
    # Layer 1: Citation validation
    valid_cites, failed_cites = validate_citations(primary_report, resource_exists_fn)
    citation_score = compute_citation_score(primary_report, resource_exists_fn)

    # Layer 2: SelfCheckGPT consistency
    variations = [generate_fn(temperature=0.7) for _ in range(n_samples)]
    overall_risk, flagged_sentences = selfcheck_report(primary_report, variations)
    consistency_score = round(1.0 - overall_risk, 3)

    # Layer 3: Combined confidence
    confidence = compute_confidence(citation_score, consistency_score)
    passed = confidence >= confidence_threshold

    if not passed:
        reason = (
            f"Confidence {confidence:.2f} below threshold {confidence_threshold:.2f}. "
            f"Citation score: {citation_score:.2f}, "
            f"Consistency score: {consistency_score:.2f}. "
            + (f"Failed citations: {failed_cites}. " if failed_cites else "")
            + (f"Flagged sentences: {len(flagged_sentences)}." if flagged_sentences else "")
        )
    else:
        reason = f"Passed all guards. Confidence: {confidence:.2f}."

    return GuardResult(
        passed=passed,
        confidence=confidence,
        citation_score=citation_score,
        consistency_score=consistency_score,
        failed_citations=failed_cites,
        flagged_sentences=flagged_sentences,
        reason=reason,
    )
