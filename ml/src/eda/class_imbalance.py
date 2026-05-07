from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


EDA_PLOTS_DIR = Path(__file__).resolve().parent / "temp" / "plots"


@dataclass
class ClassImbalanceSummary:
    label_column: str
    total_samples: int
    num_classes: int
    majority_class: str
    majority_count: int
    minority_class: str
    minority_count: int
    imbalance_ratio: float
    majority_share: float
    is_imbalanced: bool
    imbalance_level: str


class ClassImbalanceEDA:
    """Analyze class distribution and imbalance severity for a label column."""

    def __init__(
        self,
        df: pd.DataFrame,
        label_column: str,
        moderate_ratio_threshold: float = 3.0,
        severe_ratio_threshold: float = 10.0,
    ):
        self._df = df
        self._label_column = label_column
        self._moderate_ratio_threshold = moderate_ratio_threshold
        self._severe_ratio_threshold = severe_ratio_threshold

    def _validate_inputs(self) -> None:
        if self._df is None:
            raise ValueError("Input dataframe is None")

        if self._label_column not in self._df.columns:
            raise ValueError(f"Column '{self._label_column}' not found")

        if self._df.empty:
            raise ValueError("Input dataframe is empty")

    def class_distribution(self, include_percent: bool = True) -> pd.DataFrame:
        """Return class counts and optional percentage share sorted by frequency."""
        self._validate_inputs()

        counts = (
            self._df[self._label_column]
            .fillna("<MISSING>")
            .astype(str)
            .value_counts(dropna=False)
            .rename_axis("class_label")
            .reset_index(name="count")
        )

        if include_percent:
            total = counts["count"].sum()
            counts["percent"] = (counts["count"] / total) * 100
            counts["percent"] = counts["percent"].round(4)

        return counts

    def summary(self) -> ClassImbalanceSummary:
        """Compute high-level imbalance metrics for quick decision-making."""
        distribution = self.class_distribution(include_percent=True)

        majority_row = distribution.iloc[0]
        minority_row = distribution.iloc[-1]

        majority_count = int(majority_row["count"])
        minority_count = int(minority_row["count"])
        imbalance_ratio = float("inf") if minority_count == 0 else majority_count / minority_count
        majority_share = majority_count / int(distribution["count"].sum())

        if imbalance_ratio >= self._severe_ratio_threshold:
            level = "severe"
            is_imbalanced = True
        elif imbalance_ratio >= self._moderate_ratio_threshold:
            level = "moderate"
            is_imbalanced = True
        else:
            level = "low"
            is_imbalanced = False

        return ClassImbalanceSummary(
            label_column=self._label_column,
            total_samples=int(distribution["count"].sum()),
            num_classes=int(len(distribution)),
            majority_class=str(majority_row["class_label"]),
            majority_count=majority_count,
            minority_class=str(minority_row["class_label"]),
            minority_count=minority_count,
            imbalance_ratio=round(imbalance_ratio, 4) if imbalance_ratio != float("inf") else imbalance_ratio,
            majority_share=round(majority_share, 4),
            is_imbalanced=is_imbalanced,
            imbalance_level=level,
        )

    def recommendations(self) -> list[str]:
        """Return practical next-step recommendations based on imbalance severity."""
        summary = self.summary()

        recs: list[str] = []
        if not summary.is_imbalanced:
            recs.append("Imbalance is low; proceed with baseline model and monitor per-class metrics.")
            recs.append("Track macro-F1 and per-class recall during training.")
            return recs

        recs.append("Evaluate per-class precision/recall/F1, not only overall accuracy.")
        recs.append("Use stratified train/validation split to preserve label ratios.")

        if summary.imbalance_level == "moderate":
            recs.append("Start with class weighting in the loss function.")
            recs.append("Consider light random oversampling of minority classes.")
        else:
            recs.append("Use class weighting as default and compare with targeted resampling.")
            recs.append("Test robust methods for severe skew (e.g., focal loss or constrained sampling).")
            recs.append("Set a minimum support threshold for classes with very few samples.")

        return recs

    def future_strategy_options(self) -> dict[str, dict[str, str]]:
        """Document strategy hooks to implement later for imbalance mitigation."""
        return {
            "class_weighting": {
                "status": "ready_to_implement",
                "note": "Compute inverse-frequency class weights and pass them to model loss.",
            },
            "random_oversampling": {
                "status": "ready_to_implement",
                "note": "Duplicate minority-class rows to a target ratio for classical models.",
            },
            "random_undersampling": {
                "status": "ready_to_implement",
                "note": "Downsample majority class for fast experiments and ablations.",
            },
            "smote_or_variant": {
                "status": "future",
                "note": "Add synthetic sampling only when feature space supports it and leakage is controlled.",
            },
            "focal_loss": {
                "status": "future",
                "note": "Useful for neural models when severe imbalance remains after weighting.",
            },
        }

    @staticmethod
    def _resolve_least_k(least_k: int | None, total_classes: int) -> int:
        if total_classes <= 0:
            return 0
        if least_k is None or least_k <= 0 or least_k >= total_classes:
            return total_classes
        return least_k

    def export_distribution_plots(
        self,
        output_dir: str | Path = EDA_PLOTS_DIR,
        least_k: int | None = 10,
    ) -> dict[str, Any]:
        """Export class distribution plots (bar, pie, and least-k bar)."""
        distribution = self.class_distribution(include_percent=True)
        total_classes = int(len(distribution))
        least_k_used = self._resolve_least_k(least_k, total_classes)

        if plt is None:
            return {
                "paths": {},
                "config": {
                    "least_k_requested": least_k,
                    "least_k_used": least_k_used,
                    "total_classes": total_classes,
                },
            }

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        least_distribution = distribution.sort_values("count", ascending=True).head(least_k_used)

        labels = distribution["class_label"].astype(str)
        counts = distribution["count"].astype(int)

        bar_plot_path = out_dir / "class_distribution_bar.png"
        pie_plot_path = out_dir / "class_distribution_pie.png"
        least_k_plot_path = out_dir / "class_distribution_least_k_bar.png"

        plt.figure(figsize=(10, 6))
        plt.bar(labels, counts)
        plt.title(f"Class Distribution: {self._label_column}")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(bar_plot_path, dpi=200)
        plt.close()

        plt.figure(figsize=(8, 8))
        plt.pie(
            counts,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
        )
        plt.title(f"Class Share: {self._label_column}")
        plt.tight_layout()
        plt.savefig(pie_plot_path, dpi=200)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.barh(
            least_distribution["class_label"].astype(str),
            least_distribution["count"].astype(int),
        )
        plt.title(f"Least-{least_k_used} Class Counts: {self._label_column}")
        plt.xlabel("Count")
        plt.ylabel("Class")
        plt.tight_layout()
        plt.savefig(least_k_plot_path, dpi=200)
        plt.close()

        return {
            "paths": {
                "bar_plot": bar_plot_path,
                "pie_plot": pie_plot_path,
                "least_k_plot": least_k_plot_path,
            },
            "config": {
                "least_k_requested": least_k,
                "least_k_used": least_k_used,
                "total_classes": total_classes,
            },
        }

    def export_reports(self, output_dir: str | Path, least_k: int | None = 10) -> dict[str, Any]:
        """Export class distribution CSV and summary/recommendation JSON files."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        distribution = self.class_distribution(include_percent=True)
        summary = self.summary()

        distribution_path = out_dir / "class_distribution.csv"
        summary_path = out_dir / "class_imbalance_summary.json"
        plot_result = self.export_distribution_plots(output_dir=out_dir / "plots", least_k=least_k)
        plot_paths = plot_result["paths"]
        plot_config = plot_result["config"]

        distribution.to_csv(distribution_path, index=False)

        payload: dict[str, Any] = {
            "summary": asdict(summary),
            "recommendations": self.recommendations(),
            "future_strategy_options": self.future_strategy_options(),
            "plot_paths": {
                k: str(v)
                for k, v in plot_paths.items()
            },
            "plot_config": plot_config,
        }
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        report_paths: dict[str, Any] = {
            "distribution_csv": distribution_path,
            "summary_json": summary_path,
            "plot_config": plot_config,
        }
        report_paths.update(plot_paths)
        return report_paths


def analyze_class_imbalance(
    df: pd.DataFrame,
    label_column: str,
    output_dir: str | Path | None = None,
    least_k: int | None = 10,
) -> dict[str, Any]:
    """Convenience function for one-shot class imbalance EDA."""
    analyzer = ClassImbalanceEDA(df=df, label_column=label_column)
    summary = analyzer.summary()

    result: dict[str, Any] = {
        "summary": asdict(summary),
        "distribution": analyzer.class_distribution(include_percent=True),
        "recommendations": analyzer.recommendations(),
        "future_strategy_options": analyzer.future_strategy_options(),
    }

    if output_dir is not None:
        report_paths = analyzer.export_reports(output_dir=output_dir, least_k=least_k)
        result["report_paths"] = report_paths
        if "plot_config" in report_paths:
            result["plot_config"] = report_paths["plot_config"]

    return result
