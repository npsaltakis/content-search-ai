# Evaluation Plan

This document outlines a lightweight evaluation plan for retrieval quality in `content-search-ai`, with emphasis on validating the current adaptive similarity thresholds used in the search modules.

---

## 1. Purpose

The current retrieval system uses heuristic adaptive thresholds such as:
- `mean + 0.3 * std`

These thresholds are practical and currently work as part of the retrieval pipeline, but they have not yet been justified experimentally.

The goal of this evaluation plan is to:
- validate whether the current thresholds are reasonable
- compare alternative threshold variants
- produce thesis-ready evidence for threshold selection
- improve the methodological rigor of the retrieval system

---

## 2. Current Threshold Usage

At the time of writing, threshold logic appears in:

- `core/image_search.py`
  - text -> image
  - image -> image
- `core/pdf_search.py`
  - text -> PDF
  - PDF -> PDF
- `core/audio_search.py`
  - text -> audio semantic search

Current main heuristic:
- `MIN_SIM = mean + 0.3 * std`

Audio semantic search currently uses:
- `MIN_SIM = mean` when `std == 0`
- otherwise `MIN_SIM = mean + 0.3 * std`

---

## 3. Key Evaluation Question

The main question is:

Does the current adaptive threshold produce a good balance between:
- relevant results
- stable filtering
- avoidance of weak matches

---

## 4. Proposed Evaluation Strategy

The safest first approach is a small manual evaluation with representative queries.

This means:
- choose a small set of test queries for each modality
- record the returned results
- manually judge whether top results are relevant
- compare different threshold variants without changing the whole architecture

This is suitable for a thesis project because it is:
- lightweight
- understandable
- reproducible
- realistic at the current project scale

---

## 5. Threshold Variants to Compare

Suggested threshold candidates:
- `mean`
- `mean + 0.2 * std`
- `mean + 0.3 * std`
- `mean + 0.5 * std`

Optional future candidates:
- percentile-based thresholding
- modality-specific thresholds
- learned threshold selection

For the current stage, the first four variants are enough.

---

## 6. Modalities to Evaluate

### 6.1 Text -> Image

Suggested sample queries:
- simple object queries
- action-based queries
- Greek-language queries
- multilingual edge cases

What to observe:
- relevance of top results
- whether weak results are filtered out
- whether useful results disappear under stricter thresholds

### 6.2 Image -> Image

Suggested sample cases:
- near-duplicate images
- visually similar but not identical images
- difficult cross-scene similarity cases

What to observe:
- whether strong visual matches remain
- whether noisy results are removed appropriately

### 6.3 Text -> PDF

Suggested sample queries:
- academic topic queries
- technical terminology
- domain-specific concepts present in indexed PDFs

What to observe:
- whether relevant pages appear in top results
- whether paragraph-level evidence supports the ranking
- whether thresholding filters weak pages too aggressively

### 6.4 PDF -> PDF

Suggested sample cases:
- same or nearly identical document
- same topic, different wording
- unrelated document

What to observe:
- whether document-level matches remain strong
- whether page-level evidence looks coherent

### 6.5 Text -> Audio

Suggested sample queries:
- transcript-based topic queries
- semantic paraphrases of transcript content
- emotion queries already supported by the system

What to observe:
- whether transcript-based retrieval returns meaningful matches
- whether semantic thresholding is too permissive or too strict

---

## 7. Evaluation Method

### 7.1 Manual relevance judgments

For each query:
1. run the search
2. record the top-k results
3. judge each result as:
   - relevant
   - partially relevant
   - not relevant

This can be stored in a simple table or spreadsheet.

### 7.2 Compare threshold variants

For each modality and each test query:
1. run the same query using each threshold variant
2. compare:
   - number of returned results
   - quality of top results
   - sensitivity to weak matches

### 7.3 Keep the retrieval logic fixed

During the initial evaluation phase:
- do not redesign the ranking algorithm
- do not change embeddings or core models
- only compare threshold behavior

This keeps the experiment controlled and easier to explain in the thesis.

---

## 8. Suggested Metrics

### 8.1 Immediate practical metrics

These are easiest to apply first:
- number of relevant results in top-k
- ratio of relevant to irrelevant results
- qualitative usefulness of returned set

### 8.2 Thesis-friendly metrics

Once manual judgments exist, the following can be estimated:
- Precision@K
- Recall@K
- Mean Average Precision (mAP)

Even a small manually labeled evaluation set would strengthen the thesis significantly.

---

## 9. Recommended First Evaluation Dataset

A small but useful first benchmark could be:
- 10 text -> image queries
- 5 image -> image examples
- 10 text -> PDF queries
- 5 PDF -> PDF examples
- 10 text -> audio queries

This is intentionally small so it can be completed manually without turning into a large annotation project.

---

## 10. Expected Outcomes

This evaluation should help answer:
- whether `mean + 0.3 * std` is a reasonable default
- whether some modalities prefer stricter or looser filtering
- whether one global thresholding philosophy is good enough
- whether the current heuristic can be defended academically

---

## 11. Safe Next Step

The recommended immediate next step is:

1. create a small representative query set
2. record manual relevance judgments
3. compare the threshold variants listed above
4. document the findings before changing any retrieval code

This keeps the process safe, controlled, and suitable for thesis documentation.
