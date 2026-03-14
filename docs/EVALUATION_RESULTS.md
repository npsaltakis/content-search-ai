# Evaluation Results

This document records the first completed experimental threshold-validation pass for the retrieval system.

---

## Scope

This first pass was executed on the **Text -> PDF** retrieval pipeline.

Reason for choosing this modality first:
- the PDF archive has enough indexed material to test meaningfully
- many PDF filenames clearly expose topic labels
- this allows a small, controlled relevance proxy without changing retrieval code

At the time of this run:
- indexed image items: 152
- indexed PDF pages: 1351
- indexed PDF documents: 67
- indexed audio items in DB: 3

Because the indexed audio set is currently very small and the image archive would require broader manual visual annotation, the first experimental validation focused on PDFs.

---

## Tested Threshold Variants

The following threshold variants were compared:
- `mean`
- `mean + 0.2 * std`
- `mean + 0.3 * std`
- `mean + 0.5 * std`

These correspond to alpha values:
- `0.0`
- `0.2`
- `0.3`
- `0.5`

For consistency, this first-pass experiment evaluated the top `K=5` results for each query.
The system itself supports configurable `top_k`; `K=5` was used here only as the evaluation cutoff for comparison.

---

## Query Set Used

The first-pass PDF query set contained 10 topic-oriented queries:
- `decision trees`
- `genetic algorithms`
- `neural networks`
- `logistic regression`
- `minmax pruning`
- `ευριστική αναζήτηση`
- `τοπική αναζήτηση`
- `αναζήτηση με αντιπαλότητα`
- `προβλήματα ικανοποίησης περιορισμών`
- `τυφλή αναζήτηση`

---

## Relevance Proxy Used

This first pass used a conservative filename/topic relevance proxy:
- a result counted as relevant when the returned PDF filename clearly matched the intended topic of the query
- this is weaker than full page-level manual annotation, but strong enough for a first threshold-sensitivity pass

Important limitation:
- this experiment evaluates threshold sensitivity, not full end-to-end semantic quality of the model across all modalities

---

## Summary Results

See also:
- `evaluation/pdf_threshold_summary.csv`
- `evaluation/pdf_threshold_top5.csv`

Summary table for evaluation cutoff `K=5`:

| Alpha | Threshold formula | Avg returned pages per query | Relevant hits in top-5 | Queries with at least 1 relevant top-5 result |
|---|---|---:|---:|---:|
| 0.0 | `mean` | 746.1 | 8 | 4 |
| 0.2 | `mean + 0.2 * std` | 632.2 | 8 | 4 |
| 0.3 | `mean + 0.3 * std` | 569.6 | 8 | 4 |
| 0.5 | `mean + 0.5 * std` | 440.8 | 8 | 4 |

---

## Main Findings

1. The threshold variants changed the **size of the accepted result set** substantially.
2. In this sampled PDF experiment, using evaluation cutoff `K=5`, the threshold variants did **not improve the top-5 relevance profile**.
3. Increasing the threshold from `0.0` to `0.5` reduced average returned pages per query from `746.1` to `440.8`.
4. The current default `mean + 0.3 * std` reduced result-set size compared with looser settings, while keeping the same observed top-5 relevance profile as the tested alternatives.

---

## Interpretation

For this first-pass PDF evaluation, `mean + 0.3 * std` appears to be a reasonable practical default because it:
- filters more aggressively than `mean` and `mean + 0.2 * std`
- does not reduce the observed top-5 relevance at evaluation cutoff `K=5` compared with the tested alternatives in this sample
- avoids the need for a stricter threshold without evidence of top-5 improvement

This does **not** prove that `0.3` is globally optimal.

It does support the claim that the current heuristic is a defensible baseline for the present project scale.

---

## Honest Limitations of This First Pass

- only the Text -> PDF modality was experimentally checked in this pass
- relevance was judged with a filename/topic proxy rather than full page-level annotation
- image and audio threshold validation still require broader manual judgment sets
- relevance at evaluation cutoff `K=5` remained largely stable across tested thresholds, so threshold tuning alone is not enough to solve deeper semantic mismatches for some queries

---

## Practical Conclusion for 3.1

The project now has:
- a documented evaluation plan
- a practical evaluation workspace
- a metrics script
- a first completed threshold-sensitivity experiment on the PDF retrieval pipeline

This is enough to treat **3.1 as completed for a first thesis-ready validation pass**, with the clear note that broader multimodal evaluation remains a future extension.
