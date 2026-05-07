import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map labels using label_mappings.json, balance by mean count, "
            "and split into balanced/rest CSVs."
        )
    )
    parser.add_argument(
        "--input",
        default="data/raw/NLP_Dataset_2026_Expanded.xlsx",
        help="Path to the extended dataset (.xlsx or .csv).",
    )
    parser.add_argument(
        "--label-col",
        default="PartCondition",
        help="Column containing the label to map.",
    )
    parser.add_argument(
        "--mappings",
        default="src/aircraft_nlp/config/label_mappings.json",
        help="Path to label_mappings.json.",
    )
    parser.add_argument(
        "--output-balanced",
        default="data/raw/balanced.csv",
        help="Path to write the balanced dataset.",
    )
    parser.add_argument(
        "--output-rest",
        default="data/raw/rest.csv",
        help="Path to write the remaining dataset.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for downsampling.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    mappings_path = Path(args.mappings)
    output_balanced = Path(args.output_balanced)
    output_rest = Path(args.output_rest)

    df = load_dataset(input_path)

    if args.label_col not in df.columns:
        raise KeyError(f"Label column not found: {args.label_col}")

    with mappings_path.open("r", encoding="utf-8") as f:
        mappings = json.load(f)

    label_series = df[args.label_col]
    label_normalized = (
        label_series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    df["label_mapped"] = label_normalized.map(mappings).fillna("other")

    counts = df["label_mapped"].value_counts()
    mean_count = counts.mean()
    max_threshold = int(mean_count)

    if max_threshold < 1:
        raise ValueError("Mean count is too small to sample.")

    balanced_parts = []
    rest_parts = []

    for label, group in df.groupby("label_mapped", dropna=False):
        if len(group) > max_threshold:
            sampled = group.sample(n=max_threshold, random_state=args.random_state)
            remainder = group.drop(sampled.index)
            balanced_parts.append(sampled)
            rest_parts.append(remainder)
        else:
            balanced_parts.append(group)

    balanced_df = pd.concat(balanced_parts, ignore_index=True)
    rest_df = pd.concat(rest_parts, ignore_index=True) if rest_parts else df.iloc[0:0]

    output_balanced.parent.mkdir(parents=True, exist_ok=True)
    output_rest.parent.mkdir(parents=True, exist_ok=True)

    balanced_df.to_csv(output_balanced, index=False)
    rest_df.to_csv(output_rest, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Mapped labels: {len(counts)}")
    print(f"Mean count: {mean_count:.2f}")
    print(f"Max threshold: {max_threshold}")
    print(f"Balanced rows: {len(balanced_df)}")
    print(f"Rest rows: {len(rest_df)}")
    print(f"Balanced saved to: {output_balanced}")
    print(f"Rest saved to: {output_rest}")


if __name__ == "__main__":
    main()
