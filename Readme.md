# Capestone Project
## P000231DS - Applied NLP for Aerospace Text Analysis


## Project Structure

The project is organised into two main components:

* `/ml` → Machine Learning pipeline (data + backend logic)
* `/web` → Placeholder for future frontend development (e.g., Flask app UI)

P000231DS/
└── sprint1-3/
    ├── ml/
    │   ├── data/
    │   │   └── raw/
    │   │       └── NLP_Dataset_2026.xlsx
    │   │
    │   └── src/
    │       ├── app/
    │       │   ├── __init__.py
    │       │   ├── main.py
    │       │   └── pipeline.py
    │       │
    │       ├── comparison/
    │       │   ├── compare.py
    │       │   └── selector.py
    │       │
    │       ├── eda/
    │       │   ├── __init__.py
    │       │   └── class_imbalance.py
    │       │
    │       ├── features/
    │       │   ├── __init__.py
    │       │   └── engineer.py
    │       │
    │       ├── ingestion/
    │       │   ├── __init__.py
    │       │   └── ingest.py
    │       │
    │       ├── preprocessing/
    │       │   ├── __init__.py
    │       │   ├── text_cleaning.py
    │       │   └── tokenizer.py
    │       │
    │       ├── utils/
    │       │   ├── __init__.py
    │       │   └── paths.py
    │       │
    │       ├── model/
    │       │   ├── __init__.py
    │       │   ├── base.py
    │       │   ├── features.py
    │       │
    │       │   ├── bert/
    │       │   │   ├── __init__.py
    │       │   │   ├── bert.py
    │       │   │   ├── data_prep.py
    │       │   │   ├── tokenizer.py
    │       │   │   └── wrapper.py
    │       │   │
    │       │   ├── evaluation/
    │       │   │   ├── __init__.py
    │       │   │   ├── evaluate.py
    │       │   │   └── metrics.py
    │       │   │
    │       │   ├── inference/
    │       │   │   ├── __init__.py
    │       │   │   └── inference.py
    │       │   │
    │       │   ├── traditional/
    │       │   │   ├── __init__.py
    │       │   │   └── svm.py
    │       │   │
    │       │   └── training/
    │       │       ├── __init__.py
    │       │       ├── bert_trainer.py
    │       │       ├── callbacks.py
    │       │       └── optimize.py
    │
    ├── .gitattributes
    ├── .gitignore
    ├── Issues.md
    └── Readme.md
    └── requirements.txt
