# ==================================
# Pipeline Module - Sprint 1-5
# ==================================
"""
Complete ML Pipeline with multiple stages:

SPRINT 1-2: Data Preparation
- Ingest data
- Clean text
- Analyze class distribution

SPRINT 3: Modeling & Evaluation (Split & Merge)
- Branch A: SVM + TF-IDF
- Branch B: BERT
- Merge: Compare and select best

SPRINT 4-5: (Placeholder for future)
- Explainability
- Deployment
"""

from pathlib import Path
import pandas as pd
import torch

from src.ingestion.ingest import ingest_data
from src.preprocessing.text_cleaning import TextCleaner
from src.eda.class_imbalance import analyze_class_imbalance
from src.features.engineer import FeatureEngineer
from src.model.traditional.svm import SVMModel
from src.model.bert.wrapper import BertModel
from src.comparison.compare import compare_models, print_comparison, get_metric_differences, print_differences
from src.comparison.selector import select_best_model, print_selection, save_selection_result
from src.utils.paths import RAW_FILE, dataset_report_dir
from src.utils.paths import PROCESSED_DIR

# ==================================
# SPRINT 1-2: Data Preparation
# ==================================

def run_pipeline(file_path: str = None):
    """
    SPRINT 1-2: Data Preparation Pipeline
    
    Steps:
    1. Ingest raw data
    2. Clean text
    3. Analyze class distribution
    4. Feature engineering (optional)
    
    Parameters:
        file_path: Path to raw data file
    
    Returns:
        Tuple of (df_cleaned, X_tfidf, y)
    """
    
    if file_path is None:
        file_path = str(RAW_FILE)
    else:
        file_path = str(file_path)

    print("\n" + "="*80)
    print("SPRINT 1-2: DATA PREPARATION")
    print("="*80)

#====================================
# Pipeline Module
# ==================================


from src.eda import analyze_class_imbalance
from src.ingestion.ingest import ingest_data
from src.preprocessing.text_cleaning import TextCleaner
from src.model.features import FeatureEngineer
from src.utils.paths import RAW_FILE, dataset_report_dir

def run_pipeline(file_path: str = RAW_FILE):
    file_path = str(file_path)

    # =========================
    # STEP 1: INGESTION
    # =========================
    print('\n' + '-'*80)
    print('STEP 1: DATA INGESTION')
    print('-'*80)
    
    try:
        df = ingest_data(file_path)
        print(f' Loaded {len(df)} samples')
    except Exception as e:
        print(f' Error loading data: {e}')
        raise
    print('-'*20)
    print('DATA INGESTION')
    df = ingest_data(file_path)

    # =========================
    # STEP 2: CLEANING
    # =========================
    print('\n' + '-'*80)
    print('STEP 2: DATA CLEANING')
    print('-'*80)
    
    try:
        cleaner = TextCleaner(df)
        df = cleaner.pipe()
        print(f' Cleaned text')
        print(df[['discrepancy', 'discrepancy_clean']].head(3))
    except Exception as e:
        print(f' Error cleaning data: {e}')
        raise
    print('-'*20)
    print('DATA CLEANING')
    cleaner = TextCleaner(df)
    df = cleaner.pipe()
    print(df[['discrepancy', 'discrepancy_clean']])

    # =========================
    # STEP 3: EDA (CLASS IMBALANCE)
    # =========================
    print('\n' + '-'*80)
    print('STEP 3: EXPLORATORY DATA ANALYSIS')
    print('-'*80)
    print('-'*20)
    print('EDA: CLASS IMBALANCE')

    label_column = "partcondition"
    least_k = 10
    report_dir = dataset_report_dir(file_path)
    
    try:
        eda_result = analyze_class_imbalance(
            df=df,
            label_column=label_column,
            output_dir=report_dir,
            least_k=least_k,
        )

        summary = eda_result["summary"]
        print(f"\n Imbalance Level: {summary['imbalance_level'].upper()}")
        print(f" Ratio: {summary['imbalance_ratio']:.2f}")
        print(f" Majority: {summary['majority_class']} ({summary['majority_count']} samples)")
        print(f" Minority: {summary['minority_class']} ({summary['minority_count']} samples)")

        print('\n Recommendations:')
        for rec in eda_result["recommendations"]:
            print(f"   - {rec}")

        plot_config = eda_result.get("plot_config")
        if isinstance(plot_config, dict):
            print(f"\n Plot Config:")
            print(f" Least-k requested: {plot_config['least_k_requested']}")
            print(f" Least-k used: {plot_config['least_k_used']} of {plot_config['total_classes']} classes")

        if "report_paths" in eda_result:
            print('\n EDA Artifacts:')
            for name, path in eda_result["report_paths"].items():
                if name != "plot_config":
                    print(f"   - {name}: {path}")
    
    except Exception as e:
        print(f'Warning during EDA: {e}')

    # =========================
    # STEP 4: FEATURE ENGINEERING (OPTIONAL)
    # =========================
    print('\n' + '-'*80)
    print('STEP 4: FEATURE ENGINEERING')
    print('-'*80)

    try:
        fe = FeatureEngineer(df, "discrepancy_clean", label_column)
        X, y = fe.process(method="tfidf")
        print(f' Generated TF-IDF features: {X.shape}')
    except Exception as e:
        print(f' Error in feature engineering: {e}')
        X, y = None, None

    print('\n' + '-'*80)
    print(" SPRINT 1-2 COMPLETED")
    print('-'*80 + '\n')
    
    return df, X, y

# ==================================
# SPRINT 3: Modeling & Evaluation
# ==================================

def run_sprint3_pipeline(
    file_path: str = None,
    text_column: str = "discrepancy_clean",
    label_column: str = "partcondition",
    compare_models_flag: bool = True,
    save_models: bool = True,
) -> tuple:
    """
    SPRINT 3: Modeling & Evaluation Pipeline (Split & Merge)
    
    This pipeline:
    1. Prepares data (Sprint 1-2)
    2. BRANCH A: Trains SVM with TF-IDF
    3. BRANCH B: Trains BERT
    4. MERGE: Compares models and selects best
    
    Parameters:
        file_path: Path to raw data file (default: RAW_FILE)
        text_column: Column with cleaned text (default: "discrepancy_clean")
        label_column: Column with labels (default: "partcondition")
        compare_models_flag: Whether to compare models (default: True)
        save_models: Whether to save trained models (default: True)
    
    Returns:
        Tuple of (best_model_name, best_metrics, comparison_df)
    
    Example:
        >>> best_name, best_metrics, comp = run_sprint3_pipeline()
        >>> print(f"Best Model: {best_name}")
        >>> print(f"F1-Score: {best_metrics['f1_score']:.4f}")
    """
    
    if file_path is None:
        file_path = str(RAW_FILE)
    else:
        file_path = str(file_path)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 1: DATA PREPARATION (Sprint 1-2 Output)
    # ═══════════════════════════════════════════════════════════════════════════
    
    print("\n" + "─"*80)
    print("PHASE 1: DATA PREPARATION (Sprint 1-2)")
    print("─"*80)
    
    # Step 1: Ingest data
    print("\n 1.Ingesting data...")
    try:
        df = ingest_data(file_path)
        print(f" Loaded {len(df)} samples")
    except Exception as e:
        print(f" Error loading data: {e}")
        raise
    
    # Step 2: Clean data
    print("\n 2.Cleaning text...")
    try:
        cleaner = TextCleaner(df)
        df = cleaner.pipe()
        df =df.sample(n=3000,random_state=42)

        print(f" Sampled dataset size: {len(df)}")
        print(f" Cleaned text")
        print(f" Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f" Error cleaning data: {e}")
        raise
    
    # Step 3: EDA
    print("\n 3.Analyzing class distribution...")
    try:
        eda_result = analyze_class_imbalance(
            df=df,
            label_column=label_column,
            output_dir=dataset_report_dir(file_path),
            least_k=10,
        )
        print(f"  Imbalance Level: {eda_result['summary']['imbalance_level'].upper()}")
        print(f"  Total Samples: {eda_result['summary']['total_samples']}")
        print(f"  Number of Classes: {eda_result['summary']['num_classes']}")
    except Exception as e:
        print(f" Warning during EDA: {e}")
    
    # Store original data for later use
    num_labels = len(df[label_column].unique())
    texts_full = df[text_column].tolist()

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    labels_full = le.fit_transform(df[label_column])
    
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print("\nLabel mapping created ({len(label_mapping)} classes)")
    print(label_mapping)

    if 'eda_result' in locals():
        print("\n EDA Summary:")
        summary = eda_result["summary"]
        print(summary)
        
        print("\n Recommendations:")
        for rec in eda_result["recommendations"]:
            print("-", rec)

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 2: BRANCH A - Traditional ML (SVM + TF-IDF)
    # ══════════════════════=====================================================

    
    print("\n" + "─"*80)
    print("PHASE 2: BRANCH A - Traditional ML (SVM + TF-IDF)")
    print("─"*80)
    
    svm_model = None
    svm_metrics = None
    
    try:
        # Step 4: Feature Engineering
        print("\n 4.Feature Engineering (TF-IDF)...")
        fe = FeatureEngineer(df, text_column, label_column)
        X_tfidf, y = fe.process(method="tfidf", max_features=1000)
        print(f" Generated TF-IDF features: {X_tfidf.shape}")
        
        # Step 5: Train SVM
        print("\n 5.Training SVM Model...")
        svm_model = SVMModel()
        svm_model.train(X_tfidf, y)
        print(f" SVM model trained")
        
        # Step 6: Save SVM model
        if save_models:
            svm_model.save()
        
        # Step 7: Evaluate SVM
        print("\n 6.Evaluating SVM Model...")
        svm_metrics = svm_model.evaluate(X_tfidf, y)
        print(f"   Accuracy : {svm_metrics['accuracy']:.4f}")
        print(f"   F1-Score : {svm_metrics['f1_score']:.4f}")
        print(f"   Precision: {svm_metrics['precision']:.4f}")
        print(f"   Recall   : {svm_metrics['recall']:.4f}")
        print(f"   SVM evaluation complete")
        
    except Exception as e:
        print(f" Error in Branch A (SVM): {e}")
        raise
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 3: BRANCH B - Deep Learning (BERT)
    # ═══════════════════════════════════════════════════════════════════════════
    
    print("\n" + "─"*80)
    print("PHASE 3: BRANCH B - Deep Learning (BERT)")
    print("─"*80)
    
    bert_model = None
    bert_metrics = None
    
    try:
        # Step 8: Initialize BERT
        print("\n 7.Initializing BERT Model...")
        bert_model = BertModel(num_labels=num_labels)
        print(f" BERT model initialized")
        print(f" Device: {bert_model.device}")
        
        # Step 9: Train BERT
        print("\n 8.Training BERT Model...")
        print(f" Samples: {len(texts_full)}")
        print(f" Epochs: 3")
        loss_history = bert_model.train(
            texts_full[:2000], 
            labels_full[:2000], 
            batch_size=8, 
            epochs=1, 
            lr=5e-5
        )
        print(f" BERT model trained")
        print(f" Final Loss: {loss_history[-1]:.4f}")
        
        # Step 10: Save BERT model
        if save_models:
            bert_model.save()
            
        
        # Step 11: Evaluate BERT
        print("\n 9.Evaluating BERT Model...")
        bert_metrics = bert_model.evaluate(texts_full, labels_full)
        print(f"   Accuracy : {bert_metrics['accuracy']:.4f}")
        print(f"   F1-Score : {bert_metrics['f1_score']:.4f}")
        print(f"   Precision: {bert_metrics['precision']:.4f}")
        print(f"   Recall   : {bert_metrics['recall']:.4f}")
        print(f"   BERT evaluation complete")
        
    except Exception as e:
        print(f" Error in Branch B (BERT): {e}")
        raise
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 4: MERGE POINT - Compare & Select Best Model
    # ═══════════════════════════════════════════════════════════════════════════
    
    print("\n" + "─"*80)
    print("PHASE 4: MERGE POINT - Model Comparison & Selection")
    print("─"*80)
    
    try:
        # Step 12: Compare models
        print("\n 10.Comparing Models...")
        comparison_df = compare_models(
            svm_model, bert_model,
            "SVM", "BERT",
            svm_metrics, bert_metrics
        )
        print_comparison(comparison_df)
        
        # Step 13: Get differences
        print("\n 11.Calculating Metric Differences...")
        differences = get_metric_differences(comparison_df)
        print_differences(differences)
        
        # Step 14: Select best model
        print("\n 12.Selecting Best Model...")
        best_model_name, best_metrics = select_best_model(
            comparison_df, 
            metric="f1_score"
        )
        print_selection(best_model_name, best_metrics)
        
        # Step 15: Save selection result
        save_selection_result(
            best_model_name, 
            best_metrics,
            output_path=PROCESSED_DIR / "model_selection_result.txt"
        )
        
    except Exception as e:
        print(f" Error in Merge Phase: {e}")
        raise
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FINAL OUTPUT
    # ═══════════════════════════════════════════════════════════════════════════
    
    print("\n" + "="*80)
    print(" SPRINT 3 PIPELINE COMPLETE")
    print("="*80)
    print(f"\n BEST MODEL: {best_model_name}")
    print(f" F1-Score: {best_metrics['f1_score']:.4f}")
    print(f" Accuracy: {best_metrics['accuracy']:.4f}")
    print(f" Saved to: {PROCESSED_DIR / 'models/'}")
    print("="*80 + "\n")
    
    return best_model_name, best_metrics, comparison_df

# ==================================
# MAIN ENTRY POINT
# ==================================

if __name__ == "__main__":
    """
    Main execution point.
    Choose which pipeline to run:
    - Run Sprint 1-2: run_pipeline()
    - Run Sprint 3: run_sprint3_pipeline()
    """
    
    # Option 1: Run only Sprint 1-2 (Data Preparation)
    # df, X, y = run_pipeline()
    
    # Option 2: Run complete Sprint 3 (Split & Merge) - RECOMMENDED
    best_model_name, best_metrics, comparison_df = run_sprint3_pipeline()

    
    # Step 3: Text Preprocessing (tokenization for BERT)
        # TODO: add text preprocessing module wrapper function.
    
    # Step 4: Feature Engineering (TF-IDF, BoW, label encoding)
        # TODO: add feature engineering module wrapper function.


# =========================
# RUN PIPELINE
# =========================
if __name__ == "__main__":
    run_pipeline()
