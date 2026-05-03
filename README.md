aircraft-maintenance-nlp/
│
├── configs/                      # Centralized configs (VERY IMPORTANT)
│   ├── model.yaml
│   ├── training.yaml
│   ├── inference.yaml
│   └── data.yaml
│
├── data/
│   ├── raw/                     # Immutable raw data
│   ├── processed/               # Cleaned + normalized
│   ├── interim/                 # Temporary pipeline outputs
│   └── external/                # Dictionaries, mappings (abbreviations)
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   ├── preprocessing.py     # normalization, cleaning
│   │   ├── augmentation.py
│   │   └── validation.py
│   │
│   ├── features/
│   │   ├── tokenizer.py
│   │   └── feature_builder.py   # TF-IDF, hybrid features
│   │
│   ├── models/
│   │   ├── bert_classifier.py
│   │   ├── losses.py            # focal loss, weighted loss
│   │   └── model_utils.py
│   │
│   ├── training/
│   │   ├── train.py             # main training entrypoint
│   │   ├── trainer.py           # training loop
│   │   ├── evaluate.py
│   │   └── metrics.py
│   │
│   ├── inference/
│   │   ├── predictor.py         # model inference wrapper
│   │   ├── pipeline.py          # full pipeline (preprocess + predict)
│   │   └── postprocessing.py
│   │
│   ├── explainability/
│   │   ├── shap_explainer.py
│   │   ├── lime_explainer.py
│   │   └── attention_vis.py
│   │
│   ├── active_learning/
│   │   ├── sampler.py           # uncertainty sampling
│   │   ├── strategies.py
│   │   └── selector.py
│   │
│   ├── utils/
│   │   ├── logger.py
│   │   ├── config_loader.py
│   │   └── seed.py
│
├── pipelines/                   # Orchestration layer
│   ├── training_pipeline.py
│   ├── inference_pipeline.py
│   └── retraining_pipeline.py
│
├── api/                         # Serving layer
│   ├── main.py                  # FastAPI entrypoint
│   ├── routes/
│   │   ├── predict.py
│   │   ├── explain.py
│   │   └── feedback.py
│   └── schemas.py               # request/response models
│
├── hitl/
│   ├── ui/                      # (optional) frontend
│   └── backend/
│       ├── feedback_store.py
│       └── review_logic.py
│
├── experiments/                 # Notebooks, exploration
│   ├── eda.ipynb
│   └── modeling.ipynb
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_pipeline.py
│
├── scripts/
│   ├── run_training.sh
│   ├── run_inference.sh
│   └── backfill_data.py
│
├── models/                      # Saved models (or use MLflow)
│   └── bert_v1/
│
├── mlruns/                      # MLflow tracking
│
├── requirements.txt
├── Dockerfile
└── README.md