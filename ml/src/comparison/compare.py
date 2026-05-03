# ==================================
# Model Comparison Module
# ==================================
"""
Compare performance metrics between two models.
Returns a DataFrame with side-by-side comparison.
"""

import pandas as pd
from typing import Dict, Any, Tuple


def compare_models(
    model_a,
    model_b,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    metrics_a: Dict[str, float] = None,
    metrics_b: Dict[str, float] = None,
) -> pd.DataFrame:
    """
    Compare two models based on their evaluation metrics.
    
    Parameters:
        model_a: First model instance
        model_b: Second model instance
        model_a_name: Name for first model (e.g., "SVM")
        model_b_name: Name for second model (e.g., "BERT")
        metrics_a: Metrics dict for model_a {accuracy, f1_score, precision, recall}
        metrics_b: Metrics dict for model_b {accuracy, f1_score, precision, recall}
    
    Returns:
        pd.DataFrame: Comparison table with models as rows and metrics as columns
    
    Example:
        >>> svm = SVMModel()
        >>> bert = BertModel()
        >>> 
        >>> svm_metrics = svm.evaluate(X_test, y_test)
        >>> bert_metrics = bert.evaluate(texts_test, y_test)
        >>> 
        >>> comparison = compare_models(
        ...     svm, bert,
        ...     "SVM", "BERT",
        ...     svm_metrics, bert_metrics
        ... )
        >>> print(comparison)
    """
    
    # Validate inputs
    if metrics_a is None:
        raise ValueError("metrics_a (SVM metrics) cannot be None")
    if metrics_b is None:
        raise ValueError("metrics_b (BERT metrics) cannot be None")
    
    # Ensure metrics are dicts
    if not isinstance(metrics_a, dict):
        raise TypeError(f"metrics_a must be dict, got {type(metrics_a)}")
    if not isinstance(metrics_b, dict):
        raise TypeError(f"metrics_b must be dict, got {type(metrics_b)}")
    
    # Create comparison DataFrame
    comparison_data = {
        model_a_name: metrics_a,
        model_b_name: metrics_b,
    }
    
    comparison_df = pd.DataFrame(comparison_data).T
    
    # Round to 4 decimal places for readability
    comparison_df = comparison_df.round(4)
    
    # Sort columns by importance (F1 first, then accuracy, etc.)
    column_order = ['f1_score', 'accuracy', 'precision', 'recall']
    existing_cols = [col for col in column_order if col in comparison_df.columns]
    comparison_df = comparison_df[existing_cols]
    
    return comparison_df


def print_comparison(comparison_df: pd.DataFrame) -> None:
    """
    Pretty print the comparison table.
    
    Parameters:
        comparison_df: DataFrame from compare_models()
    """
    print("\n" + "="*70)
    print("MODEL COMPARISON RESULTS")
    print("="*70)
    print(comparison_df.to_string())
    print("="*70 + "\n")


def get_metric_differences(comparison_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculate the difference between model metrics.
    
    Parameters:
        comparison_df: DataFrame from compare_models()
    
    Returns:
        Dict with metric differences (e.g., {"f1_score": {"difference": 0.05, "winner": "Model B"}})
    """
    differences = {}
    
    for metric in comparison_df.columns:
        values = comparison_df[metric].values
        diff = values[1] - values[0]  # Model B - Model A
        
        winner = comparison_df.index[1] if diff > 0 else comparison_df.index[0]
        
        differences[metric] = {
            "difference": abs(diff),
            "winner": winner,
            "model_a_value": values[0],
            "model_b_value": values[1],
        }
    
    return differences


def print_differences(differences: Dict[str, Dict[str, float]]) -> None:
    """
    Pretty print the metric differences.
    
    Parameters:
        differences: Dict from get_metric_differences()
    """
    print("\n" + "="*70)
    print("METRIC DIFFERENCES")
    print("="*70)
    
    for metric, diff_info in differences.items():
        winner = diff_info["winner"]
        diff = diff_info["difference"]
        model_a_val = diff_info["model_a_value"]
        model_b_val = diff_info["model_b_value"]
        
        print(f"\n{metric.upper()}:")
        print(f"  Model A: {model_a_val:.4f}")
        print(f"  Model B: {model_b_val:.4f}")
        print(f"  Difference: {diff:.4f}")
        print(f"  Winner: {winner} ")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # Test example
    sample_metrics_a = {
        "accuracy": 0.8234,
        "f1_score": 0.7891,
        "precision": 0.7823,
        "recall": 0.7956,
    }
    
    sample_metrics_b = {
        "accuracy": 0.8923,
        "f1_score": 0.8567,
        "precision": 0.8512,
        "recall": 0.8621,
    }
    
    comparison = compare_models(
        None, None,  # Models not used in this test
        "SVM", "BERT",
        sample_metrics_a, sample_metrics_b
    )
    
    print_comparison(comparison)
    
    differences = get_metric_differences(comparison)
    print_differences(differences)
