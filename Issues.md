# Project Issues Register

## Scope
This register covers findings from review of:
- ml/src/app
- ml/src/ingestion
- ml/src/model
- ml/src/preprocessing
- ml/src/utils
- ml/src/bert

## Severity Scale
- Critical: likely to cause incorrect behavior, data loss, or major reproducibility failures
- High: strong risk of incorrect results, unstable pipeline behavior, or hard-to-debug failures
- Medium: maintainability, reliability, or evaluation quality concerns
- Low: style, hygiene, or non-blocking implementation gaps

## Open Issues

| ID | Severity | Area | File | Issue | Why It Matters | Recommended Action |
|---|---|---|---|---|---|---|
| I-001 | Critical | Ingestion Paths | ml/src/ingestion/ingest.py:25 | `ML_DIR` resolves to `ml/src`, which makes `RAW_DATA_DIR` point to `ml/src/data/raw` instead of `ml/data/raw`. | Data is read/written in inconsistent locations, hurting traceability and reproducibility. | Use `src.utils.paths` as single source of truth or compute `ML_DIR` correctly from project root. |
| I-002 | Critical | Pipeline Orchestration | ml/src/app/pipeline.py:12 | `run_pipeline(file_path=...)` ignores the passed `file_path` and always uses `RAW_FILE`. | Caller intent is ignored; experiments are not reproducible from function inputs. | Pass `file_path` into `ingest_data` and keep default only at function boundary. |
| I-003 | High | Preprocessing Hand-off | ml/src/app/pipeline.py:36 | Feature engineering uses `discrepancy` even after creating `discrepancy_clean`. | Cleaning effort is bypassed, degrading feature quality and explainability consistency. | Feed `discrepancy_clean` into `FeatureEngineer` (or make source column explicit in config). |
| I-004 | High | Data Validation | ml/src/ingestion/ingest.py:88 | `df.empty` is checked before `df is None`. | Can raise `AttributeError` instead of a controlled validation error. | Reorder checks: first `df is None`, then `df.empty`. |
| I-005 | High | Data Validation Order | ml/src/ingestion/ingest.py:170 | Validation runs before column standardization while required columns are case-sensitive (`OperatorControlNumber`, `Discrepancy`). | Ingestion can fail on valid files with minor naming variation. | Standardize columns before required-column validation, or normalize required column names. |
| I-006 | High | Text Normalization Quality | ml/src/preprocessing/text_cleaning.py:103 | `Series.replace(dict)` applies exact-cell replacement for phrase map, not in-text token replacement. | Aerospace abbreviations inside longer strings remain unexpanded, reducing model signal quality. | Replace phrase-map step with regex/token-level replacements on the text body. |
| I-007 | Medium | Entrypoint Design | ml/src/app/main.py:7 | `Bootstrap.run()` is defined without `self` and called via class assignment (`app = Bootstrap`). | Works accidentally; fragile and non-idiomatic execution path. | Use a normal `main()` function or mark `run` as `@staticmethod` and call explicitly. |
| I-008 | Medium | Packaging/Imports | ml/src/preprocessing/tokenizer.py:81 | `from src.ingestion import ingest_data` fails because package `__init__.py` does not export it. | Local smoke test path likely crashes. | Import from `src.ingestion.ingest` or export symbol in `src/ingestion/__init__.py`. |
| I-009 | Medium | Error Handling | ml/src/ingestion/ingest.py:68 | Broad `except Exception` wraps all errors into generic `Exception`. | Loses exception specificity and complicates diagnostics/monitoring. | Raise typed exceptions and preserve traceback context (e.g., `raise ... from e`). |
| I-010 | Low | Code Hygiene | ml/src/app/pipeline.py:5 | Unused imports (`pandas`) and dead TODO/comments after `return`. | Adds noise and reduces maintainability. | Remove dead code and unused imports. |
| I-011 | Low | Code Hygiene | ml/src/model/features.py:15 | Unused imports (`ingest_data`, `DataCleaner`). | Increases maintenance burden and confusion. | Remove unused imports and keep module responsibilities focused. |
| I-012 | Low | Observability | ml/src/app/pipeline.py:16 | Print-based logging throughout data and feature pipeline. | Weak production observability and poor audit trail. | Use structured logging (`logging` + run identifiers + dataset/model metadata). |
| I-013 | High | Device Selection Bug | ml/src/bert/evaluate.py:43 | `device` default is string `'none'`; truthy value prevents auto-device selection, causing `model.to('none')` failure. | Evaluation can fail immediately on default call path. | Use `device: str | None = None`; resolve with `'cuda' if available else 'cpu'`. |
| I-014 | Medium | BERT Backbone Mismatch | ml/src/bert/model.py:17 | Default backbone is `bert-base-uncased`, while project context targets ModernBERT. | Inconsistent tokenization/model behavior and benchmarking confusion. | Align defaults to ModernBERT and keep tokenizer-model pairing explicit in config. |
| I-015 | Medium | Empty Critical Modules | ml/src/bert/inference.py:1 | Inference module is empty. | No standardized prediction path for deployment or HITL review. | Implement deterministic inference API with confidence output and label decoding. |
| I-016 | Medium | Empty Critical Modules | ml/src/bert/optimize.py:1 | Optimization module is empty. | No defined path for hyperparameter tuning or calibration workflow. | Add optimization entry points (search space, objective, tracked artifacts). |
| I-017 | Medium | Training Robustness | ml/src/bert/train.py:76 | Training loop lacks gradient clipping, scheduler, and mixed precision hooks. | Greater instability risk and slower/less efficient training in large models. | Add optional grad clipping, LR scheduler, AMP (`torch.cuda.amp`) controls. |
| I-018 | Medium | Reproducibility Controls | ml/src/bert/train.py:44 | No seed-setting or deterministic controls in training/evaluation modules. | Run-to-run variance is harder to explain in safety-critical workflows. | Set and log global seeds (`torch`, `numpy`, `random`) and deterministic flags where feasible. |
| I-019 | Medium | Artifact Management | ml/src/bert/train.py:104 | `save_model` writes to relative `model.pt` with no metadata/versioning. | Checkpoints are hard to trace to dataset/model config and run conditions. | Save to structured artifact directory with timestamp/run_id and config sidecar. |
| I-020 | Medium | Label Contract Drift | ml/src/bert/data_prep.py:90 | Missing labels are replaced with `'unknown'` and then learned as a class. | Introduces artificial class; may hide data quality problems and skew metrics. | Make this behavior configurable (`drop_missing_label` vs `impute_unknown`) and log counts. |
| I-021 | Medium | Split Fragility on Rare Classes | ml/src/bert/data_prep.py:163 | Stratified split can fail when rare classes have too few samples for requested split ratios. | Runtime failures on imbalanced datasets are likely in maintenance logs. | Add pre-check for minimum class counts and fallback strategy with explicit warnings. |
| I-022 | Low | Unused Imports | ml/src/bert/evaluate.py:12 | `pandas`, `numpy`, `os` are imported but unused. | Unnecessary clutter. | Remove unused imports. |
| I-023 | Low | Unused Imports | ml/src/bert/train.py:12 | `pandas`, `numpy`, `os` are imported but unused. | Unnecessary clutter. | Remove unused imports. |
| I-024 | Low | Type Specificity | ml/src/bert/tokenizer.py:29 | Return type uses generic `dict` for metadata. | Weak interface contract for downstream tooling. | Use explicit typed dict/dataclass for metadata schema. |

## Explainability and HITL Gaps (Cross-Cutting)

| ID | Severity | Gap | Where Observed | Why It Matters | Recommended Action |
|---|---|---|---|---|---|
| X-001 | High | No SHAP integration path yet | ml/src/bert/inference.py, ml/src/bert/optimize.py, ml/src/app/pipeline.py | Project objective requires token-level rationale per prediction. | Add a dedicated explainability service that returns class, confidence, top tokens, and artifact IDs. |
| X-002 | High | No confidence threshold gating | ml/src/bert/inference.py (empty), pipeline orchestration | HITL workflow depends on escalating low-confidence/high-risk predictions. | Implement policy-based escalation (`confidence < threshold`, class-risk rules). |
| X-003 | Medium | No prediction artifact schema | Across app/bert modules | Hard to audit who predicted what, with which model/data/version. | Define and persist a prediction record schema (input hash, model version, logits/probs, explanation summary). |

## Recommended Fix Order
1. Resolve data path consistency and pipeline argument usage (I-001, I-002).
2. Fix evaluation device bug and preprocessing hand-off (I-013, I-003).
3. Harden validation and text normalization behavior (I-004, I-005, I-006).
4. Establish inference + explainability + HITL routing foundations (I-015, X-001, X-002, X-003).
5. Improve training reproducibility/artifact controls (I-017, I-018, I-019, I-021).
6. Clean hygiene items (I-010, I-011, I-022, I-023, I-024).

## Tracking Notes
- Statuses are currently implicit as OPEN; this file can be extended with owner/ETA/status columns.
- For each fix, record linked PR, test evidence, and any metric impact.
