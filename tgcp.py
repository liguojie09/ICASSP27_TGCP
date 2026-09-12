"""Text-Guarded Canonical Projection (TGCP): minimal readout-level method.

View order: 25/10, 25/5, 50/10, 50/5 (mm/s, mm/mV).
Stream order: raw, speed-normalized, trace-normalized.
All readouts for one call must refer to the same ECG recording and adapter.
No model loading, image preprocessing, fitting, or dataset I/O is performed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np


VIEWS = ((25, 10), (25, 5), (50, 10), (50, 5))
LABELS = ("CD", "HYP", "MI", "STTC")
CANDIDATES = ("NORM",) + tuple(
    ", ".join(label for bit, label in enumerate(LABELS) if mask & (1 << bit))
    for mask in range(1, 16)
)
BLEND = 0.875
FUSION_WEIGHTS = (0.5, 0.8, 0.65)
DECISION_FIELDS = (
    "prediction", "runner_up", "top1_score", "top2_score", "score_margin",
    "candidate_scores", "candidate_probabilities", "candidate_entropy",
    "candidate_scored_token_lengths",
)
Readout = Mapping[str, Any]


@dataclass(frozen=True)
class AlignmentReadout:
    """Interfaces to one view's already-fitted, frozen alignment readouts.

    Both callbacks accept a (batch, 198) array of *unstandardized* features.
    Each must apply its own fitted standardization internally. Ridge returns
    (batch, 16) centered-log predictions; the classifier returns (batch, K)
    probabilities in ``class_indices`` order. Missing classes are zero-filled.
    Callbacks must not fit or update parameters.
    """

    ridge_predict: Callable[[np.ndarray], np.ndarray]
    class_predict_proba: Callable[[np.ndarray], np.ndarray]
    class_indices: Sequence[int]


def _softmax(scores: np.ndarray) -> np.ndarray:
    """Normalize scores at unit temperature, along the final dimension."""
    value = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    return value / value.sum(axis=-1, keepdims=True)


def score_candidates(
    token_log_probabilities: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute complete-candidate mean log scores and softmax (paper Eq. 1).

    Args:
        token_log_probabilities: Sixteen teacher-forced candidate sequences
            in CANDIDATES order. Include EOS; exclude prompt and padding.
            Entries are conditional log probabilities, not model logits.

    Returns:
        Two arrays of shape (16,): mean log scores and normalized scores.
        Raw top-1 decisions must come from the original scorer's tie policy.
    """
    if len(token_log_probabilities) != len(CANDIDATES):
        raise ValueError("Exactly 16 complete candidate sequences are required.")
    values = [np.asarray(seq, dtype=np.float64) for seq in token_log_probabilities]
    if any(v.ndim != 1 or not v.size or not np.isfinite(v).all()
           or (v > 0).any() for v in values):
        raise ValueError("Each candidate needs finite, nonpositive token log probabilities.")
    scores = np.asarray([value.mean() for value in values])
    return scores, _softmax(scores)


def _probabilities(row: Readout) -> np.ndarray:
    """Read named candidates in the fixed inventory order without altering row."""
    value = np.asarray([row["candidate_probabilities"][c] for c in CANDIDATES],
                       dtype=np.float64)
    if not np.isfinite(value).all() or (value <= 0).any():
        raise ValueError("Candidate probabilities must be finite and positive.")
    if not np.isclose(value.sum(), 1.0, atol=2e-6):
        raise ValueError("Candidate probabilities must sum to one.")
    return value / value.sum()


def three_stream_features(probabilities: np.ndarray) -> np.ndarray:
    """Form the paper's 198-D feature from a normalized (3, 16) array.

    Order: probabilities (48), centered log probabilities (48), top-1
    one-hots (48), entropies (3), probability margins (3), and absolute
    differences raw/speed, raw/trace, speed/trace (48). Exact feature ties
    use the first candidate index, matching the frozen implementation.
    """
    value = np.asarray(probabilities, dtype=np.float64)
    if value.shape != (3, 16) or not np.isfinite(value).all() or (value <= 0).any():
        raise ValueError("Expected a positive, finite (3, 16) probability array.")
    if not np.allclose(value.sum(axis=1), 1.0, atol=2e-6):
        raise ValueError("Each stream must be normalized.")
    logp = np.log(value.clip(1e-12, 1.0))
    centered = logp - logp.mean(axis=1, keepdims=True)
    one_hot = np.eye(16, dtype=np.float64)[value.argmax(axis=1)]
    ordered = np.sort(value, axis=1)
    entropy = -(value * logp).sum(axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    differences = np.concatenate((abs(value[0] - value[1]),
                                  abs(value[0] - value[2]),
                                  abs(value[1] - value[2])))
    return np.concatenate((value.ravel(), centered.ravel(), one_hot.ravel(),
                           entropy, margin, differences))


def _entropy_record(logp: np.ndarray) -> dict[str, Any]:
    """Preserve the existing readout's entropy-field representation."""
    by_temperature = {}
    for temperature in (0.5, 1.0, 2.0):
        probability = _softmax(logp / temperature)
        entropy = float(-(probability * np.log(probability)).sum())
        by_temperature[f"{temperature:g}"] = {
            "entropy_nats": entropy,
            "normalized_entropy": entropy / np.log(len(CANDIDATES)),
        }
    return {"temperature_grid": [0.5, 1.0, 2.0], "by_temperature": by_temperature}


def align_readouts(
    streams: Sequence[Sequence[Readout]],
    readouts: Sequence[AlignmentReadout],
) -> list[dict[str, Any]]:
    """Align four views using three streams and three frozen readout pairs.

    Args:
        streams: Nested readout records indexed as [view][stream], size 4 x 3.
        readouts: Alignment interfaces for views 1, 2, 3, in that order.

    Returns:
        Four independent records. View 0 retains its raw decision. For each
        other view, Ridge blending (Eq. 2), log fusion (Eq. 3), and top-entry
        swapping determine the aligned distribution and recorded prediction.
        Aligned ``candidate_scores`` are log aligned probabilities, not the
        original sequence scores. Restoration always uses raw score margins.
    """
    if len(streams) != 4 or any(len(view) != 3 for view in streams) or len(readouts) != 3:
        raise ValueError("Expected four views, three streams, and three readout pairs.")
    output = [deepcopy(dict(streams[0][0]))]
    for v, fitted in enumerate(readouts, start=1):
        arms = np.stack([_probabilities(row) for row in streams[v]])
        feature = three_stream_features(arms)[None, :]
        logits = np.asarray(fitted.ridge_predict(feature), dtype=np.float64)
        local = np.asarray(fitted.class_predict_proba(feature), dtype=np.float64)
        classes = np.asarray(fitted.class_indices)
        if logits.shape != (1, 16) or not np.isfinite(logits).all():
            raise ValueError("Ridge must return a finite (1, 16) array.")
        if (classes.ndim != 1 or not np.issubdtype(classes.dtype, np.integer)
                or not len(classes) or len(set(classes.tolist())) != len(classes)
                or (classes < 0).any() or (classes >= 16).any()):
            raise ValueError("Classifier indices must be distinct integers in [0, 15].")
        if (local.shape != (1, len(classes)) or not np.isfinite(local).all()
                or (local < 0).any() or not np.isclose(local.sum(), 1.0)):
            raise ValueError("Classifier must return normalized class probabilities.")

        value = BLEND * _softmax(logits)[0] + (1.0 - BLEND) * arms.mean(axis=0)
        value /= value.sum()
        class_probability = np.zeros(16, dtype=np.float64)
        class_probability[classes] = local[0]
        weight = FUSION_WEIGHTS[v - 1]
        fused = (weight * np.log(class_probability.clip(1e-12, 1.0))
                 + (1.0 - weight) * np.log(value.clip(1e-12, 1.0)))
        selected, previous = int(fused.argmax()), int(value.argmax())
        value[previous], value[selected] = value[selected], value[previous]

        # Keep the selected index explicitly; rounded/tied probabilities can
        # have a different first argmax. Preserve the frozen stable rank fields:
        # order[1] can equal selected under an exact tie. It is a rank slot,
        # not necessarily a distinct alternative to the recorded prediction.
        logp = np.log(value)
        order = np.argsort(-value, kind="stable")
        second = int(order[1])
        row = deepcopy(dict(streams[v][0]))
        row.update({
            "prediction": CANDIDATES[selected], "runner_up": CANDIDATES[second],
            "top1_score": float(logp[selected]), "top2_score": float(logp[second]),
            "score_margin": float(logp[selected] - logp[second]),
            "candidate_scores": dict(zip(CANDIDATES, logp.tolist())),
            "candidate_probabilities": dict(zip(CANDIDATES, value.tolist())),
            "candidate_entropy": _entropy_record(logp),
        })
        output.append(row)
    return output


def _copy_decision(target: dict[str, Any], source: Readout) -> None:
    """Copy decision fields together, preserving target view metadata."""
    for key in DECISION_FIELDS:
        if key in source:
            target[key] = deepcopy(source[key])


def text_guard(
    raw: Sequence[Readout],
    aligned: Sequence[Readout],
    text_prediction: str,
) -> list[dict[str, Any]]:
    """Apply canonical preservation and the joint rule (Algorithm 1).

    Args:
        raw: Four raw readouts. ``prediction`` is the recorded candidate name;
            ``score_margin`` is its original top-1 minus top-2 sequence score.
        aligned: Four aligned readouts for the same recording and view order.
        text_prediction: One shared text-only candidate name per adapter and
            execution context, not a patient-specific ground-truth label.

    Returns:
        Four copied records; inputs are not mutated. If the canonical answer
        differs from the probe, project it to every view. Otherwise, when raw
        disagreement would become aligned unanimity, restore one disagreeing
        raw view with maximal raw margin (ties use the earlier view).
    """
    if len(raw) != 4 or len(aligned) != 4 or text_prediction not in CANDIDATES:
        raise ValueError("Expected four paired readouts and a valid text-only candidate.")
    if any(row["prediction"] not in CANDIDATES for row in (*raw, *aligned)):
        raise ValueError("Readout predictions must use the fixed candidate inventory.")
    output = [deepcopy(dict(row)) for row in aligned]
    _copy_decision(output[0], raw[0])
    anchor = raw[0]["prediction"]
    if anchor != text_prediction:
        for row in output[1:]:
            _copy_decision(row, raw[0])
    else:
        eligible = [v for v in range(1, 4) if raw[v]["prediction"] != anchor]
        unanimous = all(row["prediction"] == anchor for row in output)
        if eligible and unanimous:
            if not all(np.isfinite(float(raw[v]["score_margin"])) for v in eligible):
                raise ValueError("Restoration requires finite raw sequence-score margins.")
            chosen = max(eligible, key=lambda v: (float(raw[v]["score_margin"]), -v))
            _copy_decision(output[chosen], raw[chosen])
    return output


def tgcp(
    streams: Sequence[Sequence[Readout]],
    readouts: Sequence[AlignmentReadout],
    text_prediction: str,
) -> list[dict[str, Any]]:
    """Run alignment and text-guarded projection on one complete four-view set."""
    aligned = align_readouts(streams, readouts)
    return text_guard([view[0] for view in streams], aligned, text_prediction)
