# ==================================
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
    print('-'*20)
    print('DATA INGESTION')
    df = ingest_data(file_path)

    # =========================
    # STEP 2: CLEANING
    # =========================
    print('-'*20)
    print('DATA CLEANING')
    cleaner = TextCleaner(df)
    df = cleaner.pipe()
    print(df[['discrepancy', 'discrepancy_clean']])

    # =========================
    # STEP 3: EDA (CLASS IMBALANCE)
    # =========================
    print('-'*20)
    print('EDA: CLASS IMBALANCE')

    label_column = "partcondition"
    least_k = 10
    report_dir = dataset_report_dir(file_path)
    eda_result = analyze_class_imbalance(
        df=df,
        label_column=label_column,
        output_dir=report_dir,
        least_k=least_k,
    )

    summary = eda_result["summary"]
    print(
        f"Imbalance level: {summary['imbalance_level']} | "
        f"ratio: {summary['imbalance_ratio']} | "
        f"majority: {summary['majority_class']} ({summary['majority_count']}) | "
        f"minority: {summary['minority_class']} ({summary['minority_count']})"
    )

    print('Recommendations:')
    for rec in eda_result["recommendations"]:
        print(f"- {rec}")

    plot_config = eda_result.get("plot_config")
    if isinstance(plot_config, dict):
        print(
            f"Least-k plot requested: {plot_config['least_k_requested']} | "
            f"used: {plot_config['least_k_used']} of {plot_config['total_classes']} classes"
        )

    if "report_paths" in eda_result:
        print('EDA artifacts:')
        for name, path in eda_result["report_paths"].items():
            if name == "plot_config":
                continue
            print(f"- {name}: {path}")

    # =========================
    # STEP 4: FEATURE ENGINEERING
    # =========================
    print('-'*20)
    print('FEATURE ENGINEERING')

    fe = FeatureEngineer(df, "discrepancy_clean", label_column)

    X, y = fe.process(method="tfidf")

    print('-'*20)
    print("Pipeline completed")
    print('-'*20)
    
    return X, y


    # Step 3: Text Preprocessing (tokenization for BERT)
        # TODO: add text preprocessing module wrapper function.
    
    # Step 4: Feature Engineering (TF-IDF, BoW, label encoding)
        # TODO: add feature engineering module wrapper function.


# =========================
# RUN PIPELINE
# =========================
if __name__ == "__main__":
    run_pipeline()
