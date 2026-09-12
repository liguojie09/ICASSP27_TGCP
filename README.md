# TGCP: Stabilizing ECG Image Interpretation Across Paper Speed and Gain

Minimal method code for **Text-Guarded Canonical Projection (TGCP)**.

TGCP stabilizes diagnostic readouts from four calibrated electrocardiogram (ECG) views of the same recording. It preserves the raw canonical readout and prevents new all-view agreement with a shared text-only prediction.

This release contains one Python module, this README, a NumPy dependency file, and two existing paper figures. **No model weights, fitted parameter files, datasets, patient-level outputs, or training/evaluation scripts are included.**

## Motivation

Changing paper speed and gain changes an ECG's presentation, not its source recording or reference diagnosis. In the paper's released ECG-R1 audit on 512 paired fold-10 patients, diagnostic sets change for **90.43%** of patients. Joint changes reduce macro-F1 from **59.33%** to **21.15%**.

Simply copying one view can eliminate disagreement. TGCP instead makes explicit which reference is retained and which new consensus event is prevented.

![Paper Figure 1: calibration sensitivity, adapted-model stabilization, and native ECG illustration crops.](assets/calibration_gap.svg)

*Figure 1 from the manuscript. Released-model gap measurements and adapted-model method results are separate comparisons. The embedded ECG crops are the paper's existing illustrations, not additional dataset files.*

## Method

![Paper Figure 2: three-stream canonical alignment followed by text-guarded joint inference.](assets/method_overview.svg)

*Figure 2 from the manuscript. Orange denotes raw/canonical fields; purple, pink, and green denote alignment, restoration, and retention. This figure shows the full pipeline; this minimal code starts at its candidate-scoring/readout interface.*

1. **Score complete diagnostic sets.** Average candidate-token log probabilities, including the end-of-sequence token, and apply a unit-temperature softmax.
2. **Align noncanonical views.** Construct the 198-dimensional three-stream feature. Frozen Ridge and logistic readouts align the distribution and selected diagnosis to raw canonical targets. Blend, fuse, and swap the selected entry.
3. **Apply the joint rule.** Retain the raw canonical decision fields. Project the anchor when it differs from the text-only prediction; otherwise restore one raw disagreement if alignment would create new text-matching unanimity.

### Joint rule

For raw decisions $a^v$, aligned decisions $b^v$, and shared text-only answer $t$, view $v=0$ is canonical.

| Condition after canonical preservation | Action |
| --- | --- |
| $a^0 \ne t$ | Copy the raw canonical decision fields to all views. |
| $a^0=t$, raw views disagree, aligned views are unanimous | Restore an anchor-disagreeing noncanonical raw readout with the largest raw sequence-score margin. |
| Otherwise | Keep aligned outputs. |

Margin ties use the earlier view. Projection and restoration copy the recorded prediction and associated decision fields together; the prediction is not recomputed from rounded probabilities.

The rule retains the raw canonical readout and prevents a new event in which all four outputs match $t$. This is an output-event constraint, not a clinical safety or correctness label.

## Code layout

```text
TGCP_method_only/
├── tgcp.py
├── README.md
├── requirements.txt
└── assets/
    ├── calibration_gap.svg
    └── method_overview.svg
```

### What the code implements

| Function | Paper correspondence |
| --- | --- |
| `score_candidates` | Complete-candidate scoring and normalized distributions, Eq. (1). |
| `three_stream_features` | The fixed 198-dimensional feature in Section 2.2. |
| `align_readouts` | Ridge blending, log fusion, and top-entry swapping, Eqs. (2)–(3). |
| `text_guard` | Canonical preservation, projection, and restoration, Algorithm 1 and Eqs. (4)–(5). |
| `tgcp` | Composition of alignment and the joint rule. |

The module is a **readout-level implementation**, not a standalone image-to-diagnosis application. Image normalization, prompt/tokenizer construction, and ECG-R1 execution remain upstream. Supply their raw, speed-normalized, and trace-normalized readouts in the order below. For paper-equivalent inputs, preprocessing uses the paper's fixed raster layout and known calibration, not arbitrary resizing.

## Installation

Python 3.10 or newer:

```bash
pip install -r requirements.txt
```

Only NumPy is imported by the method. Supplied fitted callbacks may have their own dependencies.

## Input and output contract

Each call processes **one recording with all four paired views**, not four unrelated images.

- View order: `25/10`, `25/5`, `50/10`, `50/5` (mm/s, mm/mV).
- Stream order within each view: `raw`, `speed-normalized`, `trace-normalized`.
- `streams[view][stream]` is a readout mapping; there are 12 image readouts per recording.
- `CANDIDATES` fixes the 16 diagnostic-set names and their order: NORM and all nonempty subsets of CD, HYP, MI, and STTC.
- `text_prediction` is a candidate name produced once per adapter/execution context without an image, shared across patients. Do not substitute a patient label.
- Three `AlignmentReadout` objects correspond to noncanonical views 1, 2, and 3.

Decision records use these original fields:

```text
prediction
runner_up
top1_score
top2_score
score_margin
candidate_scores
candidate_probabilities
candidate_entropy
candidate_scored_token_lengths
```

Candidate scores/probabilities are mappings keyed by names in `CANDIDATES`. Raw `score_margin` is the recorded top-1 minus top-2 **sequence score**, not a probability margin. Aligned score fields instead contain log aligned probabilities; projection/restoration reuses the corresponding raw fields. Full raw decision fields must be supplied for full-field preservation. Missing source fields are not synthesized, so partial records should not be used to claim full-field preservation. Metadata remains attached to its destination view. Input objects are not mutated.

**Exact ties:** raw predictions are retained as recorded. In aligned records, `runner_up` preserves the frozen implementation's second stable probability-rank slot; it can equal the explicitly selected prediction in a tie. Do not interpret it as a guaranteed distinct alternative. The terminal rule uses recorded predictions and raw margins, not this aligned rank slot.

Alignment receives unstandardized feature rows of shape `(batch, 198)`. Each frozen callback includes its own already-fitted standardization. Ridge returns `(batch, 16)`; the classifier returns `(batch, K)` with its actual candidate indices. Missing classes are zero-filled before log fusion.

### Integration

The following function connects **already-fitted pipelines and already-produced readouts** to the method. It does not fit or load them.

```python
from tgcp import AlignmentReadout, tgcp

def stabilize_recording(streams, ridge_pipelines, classifier_pipelines, text_prediction):
    """Stabilize one four-view recording with externally supplied frozen readouts."""
    readouts = [
        AlignmentReadout(
            ridge_predict=ridge.predict,
            class_predict_proba=classifier.predict_proba,
            class_indices=classifier[-1].classes_,
        )
        for ridge, classifier in zip(ridge_pipelines, classifier_pipelines)
    ]
    return tgcp(streams, readouts, text_prediction)
```

Here each pipeline applies its own frozen standard scaler. The classifier's `classes_` must contain integer indices into `CANDIDATES`, not string diagnosis names. The three pairs are ordered by noncanonical view. If the caller already has aligned outputs, use `text_guard(raw, aligned, text_prediction)` directly.

The fixed blend is 0.875 and the three fusion coefficients are (0.50, 0.80, 0.65), as in the paper. These are method hyperparameters, not distributed learned weights.

## Reported results

Existing paper results on 512 fold-10 patients, using three adapted ECG-R1 seeds with shared alignment readouts. Entries are **mean ± sample standard deviation**; units are **%**.

| Readout | Canonical F1 (%) | Worst-view F1 (%) | Disagreement (%) | Distribution MAE (%) |
| --- | ---: | ---: | ---: | ---: |
| Adapted raw | 45.26 ± 0.95 | 34.87 ± 0.48 | 25.13 ± 0.45 | 1.84 ± 0.03 |
| TGCP | 45.26 ± 0.95 | 44.79 ± 0.19 | 6.32 ± 1.26 | 0.38 ± 0.22 |
| Canonical copy (analytic reference) | 45.26 ± 0.95 | 45.26 ± 0.95 | 0.00 | 0.00 |

F1 is macro-averaged over five diagnostic classes: CD, HYP, MI, NORM, and STTC, not the 16 candidate sets. Disagreement is the percentage of patients with at least one noncanonical diagnostic set different from the canonical set. Distribution MAE averages the absolute differences between each noncanonical distribution and that method's canonical distribution over patients, three noncanonical views, and 16 candidates.

The paired worst-view gain of TGCP over adapted raw is **9.92 ± 0.64 percentage points**. Worst-view F1 is the minimum across views within each seed, then summarized across seeds. The sample standard deviation uses denominator $3-1$; distribution MAE (%) is the original MAE multiplied by 100.

Canonical copying is an algebraic reference, not another model run. It has zero disagreement but does not enforce TGCP's no-new-text-matching-unanimity constraint. TGCP's numerical comparison is against the same adapted weights, not against released ECG-R1's canonical F1.

The table is transcribed programmatically from the manuscript's existing aggregate results. This code release does not run new experiments or claim to reproduce these numbers without the external model and fitted readouts.

## Reuse notes

The two SVGs are unchanged manuscript figures with embedded illustration crops. No separate source ECG images are bundled. They are documentation assets, not inputs to the module. Source datasets retain their own terms; this package does not assign them a new license.

The implementation has no file access, downloads, command-line entry point, model checkpoint, optimizer, or experimental runner. It can be published as a small method-code repository and linked from the manuscript once a real repository URL is available.
