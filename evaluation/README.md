# Evaluation Workspace

This folder contains the practical files needed to execute the retrieval-threshold evaluation described in `docs/EVALUATION_PLAN.md`.

## Files

- `query_set.csv`
  - starter query list for manual evaluation
- `judgments_template.csv`
  - empty template for manual relevance judgments

## Suggested workflow

1. Review `docs/EVALUATION_PLAN.md`.
2. Expand or adjust `evaluation/query_set.csv` to match the archive content you want to evaluate.
3. Run the app and execute the listed queries manually.
4. Record the returned results in a copy of `evaluation/judgments_template.csv`.
5. Fill the `relevance_label` field with one of:
   - `relevant`
   - `partial`
   - `not_relevant`
6. Run the metrics script:

```bash
python scripts/calculate_metrics.py path/to/judgments.csv
```

## Notes

- The current audio benchmark is limited because only a small number of audio items are indexed in the database.
- The safest first pass is manual evaluation without changing retrieval code.
