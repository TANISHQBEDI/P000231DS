.
├── data
│   ├── interim
│   ├── processed
│   └── raw
│       ├── NLP_Dataset_2026_Expanded.xlsx
│       └── NLP_Dataset_2026.xlsx
├── experiments
│   ├── abbrev_candidates.csv
│   └── analysis.py
├── models
│   ├── label_mapping.json
│   ├── model.pt
│   └── tokenizer
│       ├── tokenizer_config.json
│       └── tokenizer.json
├── pipeline
│   └── smoke.py
├── pyproject.toml
├── README.md
├── scripts
│   ├── install-dev.ps1
│   └── install-dev.sh
└── src
    └── aircraft_nlp
        ├── __init__.py
        ├── config
        │   ├── abbreviations.json
        │   └── label_mappings.json
        ├── data
        │   ├── __init__.py
        │   ├── preprocessing.py
        │   ├── source
        │   │   ├── __init__.py
        │   │   ├── base.py
        │   │   ├── local_file_source.py
        │   │   └── s3_source.py
        │   ├── splitting.py
        │   └── validate.py
        └── models
            ├── bert.py
            ├── data_prep.py
            ├── evaluate.py
            └── train.py