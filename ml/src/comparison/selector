# ==================================
# Model Selection Module
# ==================================
"""
Select the best model from comparison based on specified metric.
"""

import pandas as pd
from typing import Tuple, Dict, Any, Optional


def select_best_model(
    comparison_df: pd.DataFrame,
    metric: str = "f1_score",
) -> Tuple[str, Dict[str, float]]:
    """
    Select the best model based on a specific metric.
    
    Parameters:
        comparison_df: DataFrame from compare_models()
        metric: Which metric to use for selection 
                (default: "f1_score", options: "accuracy", "precision", "recall")
    
    Returns:
        Tuple of (best_model_name, best_metrics_dict)
    
    Example:
        >>> comparison = compare_models(svm, bert, ...)
        >>> best_name, best_metrics = select_best_model(comparison, metric="f1_score")
        >>> print(f"Best Model: {best_name}")
        >>> print(f"F1-Score: {best_metrics['f1_score']:.4f}")
    """
    
    # Validate metric exists
    if metric not in comparison_df.columns:
        available = list(comparison_df.columns)
        raise ValueError(
            f"Metric '{metric}' not found. Available metrics: {available}"
        )
    
    # Find the index of the best model for the given metric
    best_idx = comparison_df[metric].idxmax()
    
    # Get best model name and metrics
    best_model_name = best_idx
    best_metrics = comparison_df.loc[best_idx].to_dict()
    
    return best_model_name, best_metrics


def select_best_model_weighted(
    comparison_df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[str, float, Dict[str, float]]:
    """
    Select the best model using weighted average of all metrics.
    
    Parameters:
        comparison_df: DataFrame from compare_models()
        weights: Dict of {metric: weight} 
                 (default: equal weights for all metrics)
    
    Returns:
        Tuple of (best_model_name, weighted_score, best_metrics_dict)
    
    Example:
        >>> weights = {"f1_score": 0.5, "accuracy": 0.3, "precision": 0.2}
        >>> best_name, score, metrics = select_best_model_weighted(
        ...     comparison, weights=weights
        ... )
    """
    
    if weights is None:
        # Equal weights for all metrics
        weights = {col: 1.0 / len(comparison_df.columns) for col in comparison_df.columns}
    
    # Validate weights sum to 1.0
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 0.01:
        raise ValueError(
            f"Weights must sum to 1.0, got {total_weight}. "
            f"Weights: {weights}"
        )
    
    # Calculate weighted score for each model
    weighted_scores = {}
    for model_name in comparison_df.index:
        score = 0.0
        for metric, weight in weights.items():
            if metric in comparison_df.columns:
                score += comparison_df.loc[model_name, metric] * weight
        weighted_scores[model_name] = score
    
    # Find best model
    best_model_name = max(weighted_scores, key=weighted_scores.get)
    best_weighted_score = weighted_scores[best_model_name]
    best_metrics = comparison_df.loc[best_model_name].to_dict()
    
    return best_model_name, best_weighted_score, best_metrics


def print_selection(best_model_name: str, best_metrics: Dict[str, float]) -> None:
    """
    Pretty print the selection results.
    
    Parameters:
        best_model_name: Name of best model
        best_metrics: Metrics dict for best model
    """
    print("\n" + "="*70)
    print("MODEL SELECTION RESULTS")
    print("="*70)
    print(f"Best Model: {best_model_name.upper()}")
    print("\nMetrics:")
    for metric, value in best_metrics.items():
        print(f"  {metric:12s}: {value:.4f}")
    print("="*70 + "\n")


def print_weighted_selection(
    best_model_name: str,
    weighted_score: float,
    best_metrics: Dict[str, float],
    weights: Dict[str, float],
) -> None:
    """
    Pretty print the weighted selection results.
    
    Parameters:
        best_model_name: Name of best model
        weighted_score: Weighted average score
        best_metrics: Metrics dict for best model
        weights: Weights used
    """
    print("\n" + "="*70)
    print("MODEL SELECTION RESULTS (WEIGHTED)")
    print("="*70)
    print(f"Best Model: {best_model_name.upper()}")
    print(f"Weighted Score: {weighted_score:.4f}")
    print("\nWeights Used:")
    for metric, weight in weights.items():
        print(f"  {metric:12s}: {weight:.2%}")
    print("\nMetrics:")
    for metric, value in best_metrics.items():
        print(f"  {metric:12s}: {value:.4f}")
    print("="*70 + "\n")


def save_selection_result(
    best_model_name: str,
    best_metrics: Dict[str, float],
    output_path: str = "ml/data/processed/model_selection_result.txt",
) -> None:
    """
    Save the selection result to a file.
    
    Parameters:
        best_model_name: Name of best model
        best_metrics: Metrics dict for best model
        output_path: Where to save the file
    """
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MODEL SELECTION RESULT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Best Model: {best_model_name}\n\n")
        f.write("Metrics:\n")
        for metric, value in best_metrics.items():
            f.write(f"  {metric:12s}: {value:.4f}\n")
        f.write("\n" + "="*70 + "\n")
    
    print(f"Selection result saved to: {output_path}")


if __name__ == "__main__":
    # Test example
    test_df = pd.DataFrame({
        "f1_score": [0.7891, 0.8567],
        "accuracy": [0.8234, 0.8923],
        "precision": [0.7823, 0.8512],
        "recall": [0.7956, 0.8621],
    }, index=["SVM", "BERT"])
    
    # Simple selection
    best_name, best_metrics = select_best_model(test_df, metric="f1_score")
    print_selection(best_name, best_metrics)
    
    # Weighted selection
    weights = {"f1_score": 0.5, "accuracy": 0.3, "precision": 0.2}
    best_name, score, best_metrics = select_best_model_weighted(test_df, weights=weights)
    print_weighted_selection(best_name, score, best_metrics, weights)
