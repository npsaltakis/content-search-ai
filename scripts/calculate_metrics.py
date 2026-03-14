import csv
import sys
from collections import defaultdict
from pathlib import Path


VALID_LABELS = {"relevant", "partial", "not_relevant"}
SCORING = {"relevant": 1.0, "partial": 0.5, "not_relevant": 0.0}


def load_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        rows = list(reader)

    if not rows:
        raise ValueError("Judgments CSV is empty.")

    required = {
        "run_id",
        "threshold_variant",
        "query_id",
        "modality",
        "rank",
        "result_id",
        "result_label",
        "relevance_label",
    }
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return rows


def summarise(rows):
    by_variant = defaultdict(list)

    for row in rows:
        label = (row.get("relevance_label") or "").strip().lower()
        if label not in VALID_LABELS:
            raise ValueError(
                f"Invalid relevance_label '{row.get('relevance_label')}' "
                f"for query_id={row.get('query_id')} rank={row.get('rank')}"
            )
        by_variant[row["threshold_variant"]].append(row)

    summary = {}
    for variant, variant_rows in by_variant.items():
        scores = [SCORING[row["relevance_label"].strip().lower()] for row in variant_rows]
        relevant = sum(1 for score in scores if score == 1.0)
        partial = sum(1 for score in scores if score == 0.5)
        not_relevant = sum(1 for score in scores if score == 0.0)

        query_groups = defaultdict(list)
        for row in variant_rows:
            query_groups[row["query_id"]].append(SCORING[row["relevance_label"].strip().lower()])

        precision_like = sum(scores) / len(scores) if scores else 0.0
        avg_query_score = (
            sum(sum(group) / len(group) for group in query_groups.values()) / len(query_groups)
            if query_groups
            else 0.0
        )

        summary[variant] = {
            "rows": len(variant_rows),
            "queries": len(query_groups),
            "relevant": relevant,
            "partial": partial,
            "not_relevant": not_relevant,
            "precision_like": precision_like,
            "avg_query_score": avg_query_score,
        }

    return summary


def print_summary(summary):
    print("Threshold evaluation summary")
    print("=" * 30)

    for variant in sorted(summary):
        data = summary[variant]
        print(f"\nVariant: {variant}")
        print(f"Rows evaluated      : {data['rows']}")
        print(f"Queries evaluated   : {data['queries']}")
        print(f"Relevant            : {data['relevant']}")
        print(f"Partial             : {data['partial']}")
        print(f"Not relevant        : {data['not_relevant']}")
        print(f"Precision-like score: {data['precision_like']:.3f}")
        print(f"Avg query score     : {data['avg_query_score']:.3f}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/calculate_metrics.py <judgments.csv>")
        raise SystemExit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        raise SystemExit(1)

    rows = load_rows(csv_path)
    summary = summarise(rows)
    print_summary(summary)


if __name__ == "__main__":
    main()
