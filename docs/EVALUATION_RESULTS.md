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


---

## Second Pass: Text -> Image Threshold Validation

A second first-pass threshold-sensitivity evaluation was executed on the **Text -> Image** retrieval pipeline.

Reason for choosing this modality next:
- the image archive is large enough to support a small manual inspection pass
- a few visually clear concepts were available in the indexed set
- this allows an initial multimodal extension of the threshold analysis without changing retrieval code

Supporting files for this pass:
- `evaluation/image_threshold_summary.csv`
- `evaluation/image_threshold_top5.csv`
- `evaluation/image_eval_sheets/`

### Query Set Used

The image pass used 5 visually grounded text queries:
- `football player`
- `portrait man`
- `boat`
- `firefighters`
- `band orchestra`

### Relevance Proxy Used

This pass used a conservative visual-target proxy:
- for each query, one clearly relevant image or one small relevant filename set was identified in advance
- top-5 outputs were then checked against this proxy and visually inspected through contact sheets

Important limitation:
- unlike the PDF pass, image relevance cannot be judged reliably from filenames alone
- this image pass should therefore be treated as a small qualitative threshold-sensitivity study, not a full benchmark

### Summary Results

| Alpha | Threshold formula | Avg returned images per query | Relevant hits in top-5 | Queries with at least 1 relevant top-5 result |
|---|---|---:|---:|---:|
| 0.0 | `mean` | 69.4 | 5 | 5 |
| 0.2 | `mean + 0.2 * std` | 60.2 | 5 | 5 |
| 0.3 | `mean + 0.3 * std` | 53.0 | 5 | 5 |
| 0.5 | `mean + 0.5 * std` | 43.6 | 5 | 5 |

### Main Findings

1. As in the PDF pass, stricter thresholds reduced the accepted result-set size substantially.
2. In this sampled image experiment, the tested threshold variants did not change the observed top-5 relevance proxy.
3. The current default `mean + 0.3 * std` again behaved like a reasonable middle ground: it filtered more weak results than looser variants while preserving the same top-5 proxy score in this sample.
4. Visual inspection showed that some broad text queries still introduce semantic drift inside the top-5, especially for categories such as `portrait man` or `firefighters`.

### Interpretation

For this first Text -> Image pass, `mean + 0.3 * std` remains a defensible practical default because it:
- reduces the number of returned images compared with looser settings
- preserves the same observed top-5 proxy result in the sampled queries
- avoids moving to a stricter threshold without evidence of improved top-5 behavior

At the same time, the image pass makes one limitation clearer than the PDF pass:
- threshold tuning alone is not enough to solve semantic ambiguity in broad visual queries

### Honest Limitations of This Image Pass

- only 5 manually selected text queries were used
- relevance judgments were based on small visual target sets rather than full annotation of all top results
- the image archive contains many personal or event-style photos, making broad-category labeling noisier than in the PDF archive
- this pass validates threshold behavior more than full semantic retrieval quality

### Practical Conclusion

The project now has first-pass threshold validation evidence for:
- **Text -> PDF**
- **Text -> Image**

This strengthens the thesis by showing that the current adaptive threshold has now been checked on more than one modality, even though broader multimodal evaluation is still future work.
