# Capestone Project
## P000231DS - Applied NLP for Aerospace Text Analysis


## Project Structure

The project is organised into two main components:

* `/ml` → Machine Learning pipeline (data + backend logic)
* `/web` → Placeholder for future frontend development (e.g., Flask app UI)

project_root/
│
├── ml/
│   ├── data/
│   │   ├── raw/         # Original, unmodified data from sources
│   │   ├── clean/       # Cleaned data (after quality checks)
│   │   └── processed/   # Fully preprocessed data ready for modelling
│   │
│   └── src/
│       ├── ingestion.py       # Data loading and ingestion scripts
│       ├── cleaning.py        # Data cleaning and validation logic
│       ├── preprocessing.py   # NLP preprocessing (tokenisation, etc.)
│       ├── features.py        # Feature engineering (TF-IDF, BoW, etc.)
│       └── pipeline.py        # End-to-end pipeline orchestration
│
├── web/                 # Reserved for frontend (to be implemented later)
│
├── README.md
└── requirements.txt
