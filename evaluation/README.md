# Evaluation Workspace

This folder contains the practical files needed to execute the retrieval-threshold evaluation described in `docs/EVALUATION_PLAN.md`.

## Files

- `query_set.csv`
  - expanded starter query list for manual multimodal evaluation
- `judgments_template.csv`
  - empty template for manual relevance judgments
- `pdf_threshold_summary.csv`
  - summary of the completed first-pass PDF threshold experiment
- `pdf_threshold_top5.csv`
  - top-5 PDF observations used in the first evaluation pass
- `EVALUATION_STATUS.md`
  - current modality-by-modality evaluation status

## Suggested workflow

1. Review `docs/EVALUATION_PLAN.md`.
2. Review `evaluation/EVALUATION_STATUS.md` to see what is already complete.
3. Adjust `evaluation/query_set.csv` to match the currently indexed archive content.
4. Run the app and execute the listed queries manually.
5. Record the returned results in a copy of `evaluation/judgments_template.csv`.
6. Fill the `relevance_label` field with one of:
   - `relevant`
   - `partial`
   - `not_relevant`
7. Run the metrics script:

```bash
python scripts/calculate_metrics.py path/to/judgments.csv
```

## Notes

- The PDF modality already has a first completed threshold-validation pass.
- Image evaluation needs manual visual inspection because filenames are not enough for reliable judgments.
- Audio evaluation is currently constrained by the small indexed audio set in the database.
- The safest workflow is still manual evaluation without changing retrieval code.
