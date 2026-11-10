# Contextual xG: project context and development handoff

Updated 2026-11-10 after notebook cleanup, saved-model verification, and creation of three analysis notebooks.

Place this file at the root of the cloned `contextual-xg` repository. Ask the next coding assistant to read it before changing the project.

## Start here

This is a Python expected goals project using StatsBomb open data. The original experiment is complete: data preparation, eight feature groups, baseline models, tuning, stacking, calibration selection, saved-model verification, and held-out test evaluation have all finished.

The selected model is `stack_identity`. It combines four tuned base models through a logistic regression meta-model. Identity calibration leaves the stack's probabilities unchanged.

The user develops the project in VS Code using the official Google Colab extension. The main notebook has been cleaned up, and separate notebooks now cover match performance, contextual shot exploration, and team attacking profiles. The user has requested publication of these changes to GitHub. Inspect the current files and Git diff before further edits, and preserve the completed experiment.

The earlier runtime allocation failure was followed by a successful user-run Colab resume session on 2026-11-10. Saved outputs verify all 6,555 validation predictions through the complete model chain, with maximum difference 2.22e-16. This establishes that the recorded resume run worked; do not assume that its live kernel or Python objects still exist.

Working preferences:

- For runtime work, give one small executable notebook cell at a time, explain its purpose, and wait for the user's output.
- Reuse checkpoints and fitted models when the task does not require rebuilding them.
- Preserve existing user changes. The local notebook had uncommitted changes when this handoff was prepared.
- Keep the original test results as a recorded baseline. Put new experiments and their artifacts under distinct names or directories.
- This Python project was developed separately from the earlier local R project. Its notebook and saved Python artifacts are the implementation to continue.

## Repository and files to read

Repository: <https://github.com/chihab46/contextual-xg>

Read these files first:

1. `PROJECT_CONTEXT.md`, this document.
2. `data_source.json` and `modelling_config.json`.
3. `results/test_result.json` for the completed evaluation.
4. `results/frozen_manifest.json` for exact features, parameters, versions, artifact paths, and hashes.
5. `notebooks/contextual_xg_ipynb.ipynb` for the implementation.

The notebook's actual filename is `contextual_xg_ipynb.ipynb`. Inspect the directory before assuming it has been renamed to `contextual_xg.ipynb`.

The repository layout prepared for this update is:

```text
contextual-xg/
  LICENSE
  PROJECT_CONTEXT.md
  data_source.json
  modelling_config.json
  match_analysis/
    chelsea_tottenham_2016.ipynb
  shot_analysis/
    contextual_shot_explorer.ipynb
  team_analysis/
    team_attacking_profiles.ipynb
  notebooks/
    contextual_xg_ipynb.ipynb
  results/
    frozen_manifest.json
    test_calibration.png
    test_calibration_bins.parquet
    test_metrics.parquet
    test_model_comparison.png
    test_paired_comparison.parquet
    test_result.json
```

Commit `98a99318144c4d35b5b846e7e62661078c4f84ab` added the seven result files and two configuration files to `main`. All nine uploaded Git blob hashes were checked against the local originals. The existing notebook and licence were preserved.

`README.md`, `requirements.txt`, and a root `.gitignore` were not present in the inspected clone. Check again before creating them. Notebook organisation and shared helper consolidation are complete. Repository documentation, dependency files, and extraction into reusable Python modules remain possible future tasks.

### Notebook execution history

The main notebook now has 99 code cells and 45 Markdown cells. Duplicate recovery and feature-engineering sections, temporary diagnostics, and unused model collections were removed. Saving, scoring, and repeated tuning logic use shared helpers. XGBoost verification sets CUDA before predicting, and file recovery precedes hashing.

Its default `RUN_MODE = "resume"` loads stage-8 features, saved CV assignments, and fitted artifacts, then verifies validation predictions and displays the recorded test results. The user's saved Colab run completed this path without errors. This is not a new execution of the full training experiment.

`RUN_MODE = "reproduce"` requires a new `EXPERIMENT_ID` and writes under `experiments/<EXPERIMENT_ID>/` locally and on Drive. Nonempty experiment directories and existing frozen manifests or test predictions are protected. The complete reproduction branch has not been rerun since cleanup.

The original development snapshot had 128 code cells, including an empty final cell, and no Markdown cells. Three historical error outputs in that snapshot were removed during cleanup:

- Cell 42: `DRIVE_PROCESSED_DIR` was undefined during an earlier recovery attempt.
- Cell 113: CPU XGBoost inference failed an overly strict comparison with saved GPU predictions. GPU inference subsequently reproduced them exactly.
- Cell 117: the local `data_source.json` copy was missing. It and `matches.parquet` were restored from Drive in the following recovery step.

Those cell numbers refer to the old snapshot. They do not establish that the final experiment failed; later cells completed verification and evaluation successfully. The current notebook documents the setup and execution modes explicitly. Handle any dependency-installation restart before loading state, and keep reproduction separate from resuming saved artifacts.

### Analysis notebooks

All three analysis notebooks run independently on a CPU Python kernel and use the original frozen `stack_identity` test predictions. Their artifact root defaults to `/content/drive/MyDrive/xg-python`, with an override through `CONTEXTUAL_XG_ARTIFACT_ROOT`. They verify frozen hashes and never refit models or replace original artifacts. Public match metadata and events use the pinned data commit and a shared temporary cache.

- `match_analysis/chelsea_tottenham_2016.ipynb` covers Chelsea's 2-2 draw with Tottenham on 2016-05-02, match ID 3754092. Its 29 non-penalty shots yield 0.689116 npxG for Chelsea and 2.460320 for Tottenham. It includes a team summary, shot maps, cumulative xG, half-by-half comparison, chance rankings, and freeze frames.
- `shot_analysis/contextual_shot_explorer.ipynb` supports all 6,868 test shots across 273 matches. It provides match/team/player/outcome filters, the 67 raw features, saved base-model and benchmark forecasts, linked passes, context zoom, and paired observed-shot comparisons. Its starting comparison is Kane's 34:29 goal against Son's 39:13 blocked shot in the Chelsea-Tottenham match. It does not provide counterfactual or causal attribution controls.
- `team_analysis/team_attacking_profiles.ipynb` covers 98 teams with four to seven appearances each. It includes shot volume versus quality, attacking and opponent npxG per match, overlapping shot-pattern shares, shot maps, match logs, and an optional league/team selector. Authoritative home/away IDs come from the five pinned fixture catalogues. Both sides of each covered fixture count toward exposure, including zero-shot appearances. Pooled league references use ratios of totals.

The analysis notebooks were executed with actual frozen artifacts on a local CPU kernel; their saved figures were visually inspected. Synthetic edge-case checks and programmatic widget-callback tests passed. VS Code's widget display was not inspected. Static examples and direct plotting functions remain usable without widget rendering.

These analyses describe the test window from 2016-04-11 through 2016-05-17. They are not full-season rankings, new model-selection exercises, or fresh confirmatory tests. The complete shot-level prediction files remain outside the repository; the notebooks contain selected rendered outputs derived from them.

Older handoffs named `PYTHON_COLAB_SESSION_HANDOFF.md` and `PYTHON_COLAB_SESSION_HANDOFF_2026-11-08.md` exist in the earlier R workspace. They describe earlier stopping points. This document and the exported final result supersede their completion status.

## Data source and population

The source is [StatsBomb open data](https://github.com/hudl/open-data), pinned to:

```json
{
  "repository": "hudl/open-data",
  "commit": "3bfbffe1de5750ebd47d770be0bb924a10cde54f"
}
```

All competitions use season ID `27`, the 2015/2016 season.

| Competition | Competition ID | Matches |
| --- | ---: | ---: |
| Premier League | 2 | 380 |
| Ligue 1 | 7 | 377 |
| Bundesliga | 9 | 306 |
| Serie A | 12 | 380 |
| La Liga | 11 | 380 |
| Total | | 1,823 |

The historical commit restored the complete 306-match Bundesliga season. The branch initially inspected had only 34 Leverkusen matches. Preserve the pinned commit when reproducing the experiment. Keep the 377 available Ligue 1 matches; no missing fixtures were manufactured.

There are 45,719 extracted shots and 45,233 eligible non-penalty shots. Eligibility is `shot_type != 'Penalty'`. The binary target is `is_goal`, derived from shot outcome `Goal`. `benchmark_statsbomb_xg` is a comparison prediction and never a predictor.

### Fixed chronological split

| Split label | Match dates | Matches | Eligible shots |
| --- | --- | ---: | ---: |
| `train` | Through 2016-02-28 | 1,279 | 31,810 |
| `validation` | 2016-02-29 through 2016-04-10 | 271 | 6,555 |
| `test` | 2016-04-11 through 2016-05-17 | 273 | 6,868 |

Split by match and retain the saved assignments. The training label is `train`, not `training`. Match metadata uses `match_date` with a datetime dtype. Align tables by `shot_id`; prior merges reset pandas indices.

The original test set has now been evaluated. New decisions made after inspecting its results must not be described as independently validated on an untouched test set. Develop changes using training and validation data, preserve the original baseline, and document any new evaluation protocol. Use new independent data for a fresh confirmatory performance claim.

## Feature engineering

`FEATURE_GROUPS` is the ordered predictor whitelist. There are 67 raw predictors before encoding. Training types were 44 numeric, 13 Boolean, and 10 categorical.

| Group | Registry key | New predictors | Cumulative predictors |
| --- | --- | ---: | ---: |
| 1 | `01_shot_geometry` | 6 | 6 |
| 2 | `02_shot_and_event_context` | 10 | 16 |
| 3 | `03_goalkeeper_positioning` | 6 | 22 |
| 4 | `04_defensive_pressure` | 6 | 28 |
| 5 | `05_shot_obstruction` | 4 | 32 |
| 6 | `06_attacking_support_and_numerical_balance` | 7 | 39 |
| 7 | `07_assist_and_final_pass_context` | 16 | 55 |
| 8 | `08_possession_and_preceding_action_context` | 12 | 67 |

The stage-8 checkpoint JSON stores the group registry. The frozen manifest stores each selected model's ordered raw feature list. Read those definitions instead of selecting every numeric column from the full modelling table.

Exclude the target, outcomes, benchmark xG, IDs, split/date routing fields, and serialized freeze frames from automatic predictor selection.

Feature semantics to preserve:

- Coordinates use a 120 by 80 pitch. Distances are pitch units, and angles are radians. Goal centre is `[120, 40]`; posts are `[120, 36]` and `[120, 44]`.
- Three training shot locations outside nominal pitch bounds were retained with unavailable shot-dependent geometry. They were not clamped or dropped.
- Goalkeeper and player measurements use shot-event freeze frames. Unknown frames and observed frames with no qualifying player have different meanings. Outfield-player counts exclude goalkeepers. Supporting teammates exclude the shooter.
- Counts can be zero when absence was observed. Unavailable measurements remain missing for the model's fitted preprocessing. Do not replace every missing measurement with zero.
- Assisting passes are linked through each shot's explicit `key_pass_id`. Check match, team, period, and earlier event index. Do not substitute the previous event or use outcome-dependent `goal_assist` as a feature.
- Shots without a linked pass have false pass flags, `No linked pass` categories, and missing continuous pass measurements.
- Possession features use only the same match, period, possession, and team. Events must have an earlier event index and a timestamp no later than the shot. Aggregating the entire possession would include future events.
- Completed carry distance sums qualifying team carries, checking their start times and durations. The end-time tolerance is 0.001 seconds. Missing carry inputs make the aggregate missing. Carries ending later than the cutoff are excluded.

The appendixed feature registry gives the exact original names and order.

## Models and model selection

All 32 cumulative baselines finished: eight feature stages for each of logistic regression, GAM, GPU XGBoost, and Extra Trees.

Validation log loss selected each family's feature stage. Hyperparameter searches then used five expanding chronological folds within training. The primary search criterion was pooled per-shot out-of-fold log loss. Brier score, ROC AUC, and average precision were secondary metrics.

All matches on the same date stay together. Training dates precede held-out fold dates. Fold 0 contains 6,215 warmup shots; folds 1 through 5 supply 25,595 base-model out-of-fold rows. Learned preprocessing fits inside each training fold.

The four tuned models were refitted on all 31,810 training shots. There was no training-plus-validation refit for the published test evaluation.

| Family | Stage | Raw predictors | Selected settings |
| --- | ---: | ---: | --- |
| Logistic regression | 7 | 55 | `C=0.1`, L2, `lbfgs`, `max_iter=2000` |
| GAM | 7 | 55 | `lam=10`, 8 cubic spline basis functions, `max_iter=200` |
| XGBoost | 8 | 67 | Depth 4, child weight 10, learning rate 0.05, 231 rounds |
| Extra Trees | 5 | 32 | 500 trees, minimum leaf size 40, `max_features=1.0`, no bootstrap |

XGBoost also uses binary logistic loss, histogram trees, subsample 0.8, column subsample 0.8, L2 regularisation 1.0, and `max_bin=256`. Its final round count was selected using the last 20% of training match dates for internal early stopping, then it was fitted on all training shots. The external validation and test labels did not choose its rounds.

The GAM with CV-selected `lam=10` is the stack component. A baseline GAM with `lam=0.6` had slightly better validation log loss, but it was not substituted after the CV decision.

### Preprocessing

- Logistic regression uses numeric and Boolean median imputation with missing indicators, standardisation, categorical Unknown imputation, and one-hot encoding with unknown categories ignored.
- GAM uses the fitted dense preprocessor. Nonbinary numeric measurements and counts receive smooth terms; Boolean inputs, binary numeric inputs, missing indicators, and one-hot outputs receive linear terms. Use `predict_mu` for probabilities.
- Extra Trees uses imputation, missing indicators, and categorical encoding, with numeric scaling replaced by passthrough.
- XGBoost preserves numeric NaNs and uses dense float32 inputs. Categories use Unknown imputation and one-hot encoding. Retain the distinction between numeric zero and missing values.

### Stack construction

The exact base-model input order is:

```python
STACK_MODEL_ORDER = ["logistic_regression", "gam", "xgboost", "extra_trees"]
```

Base-model out-of-fold candidate choices used only strictly earlier held-out blocks. Fold 1 used defaults. Later blocks used earlier-block scores within the original candidate pools. Adaptive grid extensions were excluded from these historical choices. The selected full-training models matched the final choices within those original pools.

The meta-model trained on 25,595 such rows. It takes only the four named base probabilities, clips them at `1e-6` and `1 - 1e-6`, applies logit, standardises them with its fitted scaler, and applies logistic regression with fixed `C=1`, `solver='lbfgs'`, `max_iter=2000`, and seed 42. It is saved as `raw_meta_pipeline.joblib`.

Cast the four base probabilities to float64 before feeding the meta-pipeline. Preserve feature order and shot order. Do not refit its scaler at inference.

Further chronological meta-model cross-fitting produced 19,610 ensemble out-of-fold predictions. Calibration methods were compared on 14,443 shared scoring rows, using only earlier ensemble blocks to fit learned mappings.

| Calibration method | CV log loss |
| --- | ---: |
| Identity | 0.244007 |
| Sigmoid | 0.244445 |
| Isotonic | 0.247183 |

Identity was selected. The saved calibration bundle has `method='identity'`, `model=None`, and `probability_clip_epsilon=1e-6`. Identity returns the raw stack probability unchanged. Do not reuse a generic `calibrator` variable left over from the comparison loop.

## Saved-model verification and final results

Calibration finalization is complete. Reloaded identity calibration reproduced all 6,555 saved stack validation predictions exactly.

The complete saved prediction chain was then checked on validation:

- Logistic regression reproduced its forecasts exactly.
- GAM reproduced its forecasts exactly.
- Extra Trees differed by at most `2.22e-16`.
- Reloaded XGBoost initially defaulted to CPU and differed by at most `5.96046447754e-7`.
- Setting the reloaded booster to `device='cuda:0'` reproduced all 6,555 XGBoost forecasts exactly.
- The complete GPU base-model, meta-model, and identity-calibration chain differed from saved stack forecasts by at most `2.220446049250313e-16`.

The frozen inference configuration therefore records CUDA XGBoost, dense float32 XGBoost inputs, float64 meta-inputs, positive class 1, and four CPU threads. Tiny CPU/GPU differences are a known reproduction detail. Do not silently replace the strict historical comparison with a relaxed tolerance.

The selected stack's validation metrics were log loss 0.261458, Brier 0.075376, ROC AUC 0.811728, and average precision 0.388825.

Before test scoring, 21 source/model/provenance files were hashed and their local and Drive copies compared. The complete configuration was frozen. Predictions for the five project models were saved before loading test outcomes for scoring.

### Held-out test results

The test set has 6,868 shots, 273 matches, and 713 goals. StatsBomb xG coverage was complete, with no missing or invalid benchmark predictions. All six methods scored the same rows. Log loss and Brier score were independently recomputed from their formulas to check the scoring helper.

| Model | Log loss | Brier score | ROC AUC | Average precision |
| --- | ---: | ---: | ---: | ---: |
| Stack with identity calibration | 0.260198 | 0.074897 | 0.821797 | 0.430160 |
| StatsBomb xG benchmark | 0.261201 | 0.074532 | 0.819261 | 0.438834 |
| XGBoost | 0.262421 | 0.075326 | 0.817150 | 0.424612 |
| GAM | 0.262441 | 0.075554 | 0.817843 | 0.420754 |
| Logistic regression | 0.262934 | 0.075682 | 0.816731 | 0.418180 |
| Extra Trees | 0.264289 | 0.076377 | 0.815961 | 0.404083 |

Lower log loss and Brier are better. Higher ROC AUC and average precision are better. Read JSON/Parquet for unrounded values.

Paired bootstrap sampling resampled whole matches 5,000 times with seed 42. Each replicate pooled shot losses across its sampled matches, retaining the same sampled matches for both models. The 95% intervals are percentile intervals.

| Metric | Stack minus StatsBomb | 95% interval |
| --- | ---: | --- |
| Log loss | -0.001003 | [-0.005062, 0.002508] |
| Brier score | 0.000364 | [-0.000603, 0.001306] |

Both intervals include zero. The stack has better log-loss and ROC-AUC point estimates; StatsBomb has better Brier and average-precision point estimates. These results do not establish a clear advantage between them.

Descriptive calibration plots used within-model quantile bins. The stack total was 692.447 xG and the benchmark total was 663.261 xG, compared with 713 observed goals. These aggregate diagnostics do not prove perfect calibration or justify tuning on the test set.

The final result status is `complete`. The model-selection status in `modelling_config.json` is `test_evaluated`. Its stacking section still records `calibration_complete`, which describes the finished calibration stage.

## Runtime paths and saved artifacts

The repository contains exported results. Most datasets, intermediate predictions, and fitted models remain in the user's Google Drive project.

```python
from pathlib import Path

commit = "3bfbffe1de5750ebd47d770be0bb924a10cde54f"
PROJECT_DIR = Path("/content/xg-python")
DRIVE_PROJECT_DIR = Path("/content/drive/MyDrive/xg-python")
PROCESSED_DIR = PROJECT_DIR / "data/processed" / commit
DRIVE_PROCESSED_DIR = DRIVE_PROJECT_DIR / "data/processed" / commit
model_relative = Path("models") / commit
stacking_relative = model_relative / "stacking"
calibration_relative = stacking_relative / "calibration"
final_relative = model_relative / "final_evaluation"
```

Paths inside `results/frozen_manifest.json` and `results/test_result.json` refer to this original project-root layout. Exporting those JSON files into the repository's `results/` directory did not rewrite their provenance paths. A missing `models/<commit>/...` path in a Git clone does not by itself mean the Drive artifact is missing. Keep the frozen JSON unchanged and use an explicit artifact-root mapping when needed.

Key saved files, relative to either Colab or Drive project root:

```text
data_source.json
matches.parquet
modelling_config.json
data/processed/<commit>/features_stage_06.parquet
data/processed/<commit>/features_stage_07.parquet
data/processed/<commit>/features_stage_08.parquet
data/processed/<commit>/features_stage_08.json
data/processed/<commit>/linked_passes.parquet
data/processed/<commit>/possession_context.parquet
data/processed/<commit>/training_cv_assignments.parquet
models/<commit>/tuning_summary.json
models/<commit>/tuning/logistic_regression/best_pipeline.joblib
models/<commit>/tuning/gam/best_bundle.joblib
models/<commit>/tuning/xgboost/best_model.ubj
models/<commit>/tuning/xgboost/best_preprocessor.joblib
models/<commit>/tuning/extra_trees/best_pipeline.joblib
models/<commit>/stacking/oof_inputs.parquet
models/<commit>/stacking/oof_selection_history.parquet
models/<commit>/stacking/validation_inputs.parquet
models/<commit>/stacking/raw_meta_pipeline.joblib
models/<commit>/stacking/raw_validation_predictions.parquet
models/<commit>/stacking/raw_result.json
models/<commit>/stacking/ensemble_oof_predictions.parquet
models/<commit>/stacking/calibration/selected_calibrator.joblib
models/<commit>/stacking/calibration/selected_validation_predictions.parquet
models/<commit>/stacking/calibration/selected_result.json
models/<commit>/final_evaluation/frozen_manifest.json
models/<commit>/final_evaluation/test_model_predictions.parquet
models/<commit>/final_evaluation/test_evaluation_predictions.parquet
models/<commit>/final_evaluation/test_metrics.parquet
models/<commit>/final_evaluation/test_paired_comparison.parquet
models/<commit>/final_evaluation/test_model_comparison.png
models/<commit>/final_evaluation/test_calibration_bins.parquet
models/<commit>/final_evaluation/test_calibration.png
models/<commit>/final_evaluation/test_result.json
```

Each tuned family also has `best_result.json` and `best_validation_predictions.parquet`. Full fit metadata for all four selected families is embedded in the frozen manifest. Earlier baseline and candidate-search artifacts also remain on Drive.

Drive holds 19 raw-event batch tar archives under `data/raw/<commit>/event_batches/`. They contain the 1,823 gzip-compressed match-event files. Recover raw events only when the requested work needs them.

The two shot-level test prediction files remain on Drive and were intentionally omitted from the initial GitHub upload. Their hashes still appear in the final result's original artifact inventory.

## Environment and runtime recovery

The completed experiment used Python 3.13.16, a Tesla T4 with 15,360 MiB, seed 42, and a four-thread CPU budget. The frozen package versions are:

| Package | Version |
| --- | --- |
| NumPy | 2.4.6 |
| pandas | 2.2.3 |
| SciPy | 1.16.3 |
| scikit-learn | 1.6.1 |
| XGBoost | 3.4.1 |
| pyGAM | 0.12.0 |
| PyArrow | 23.0.1 |
| joblib | 1.6.0 |
| threadpoolctl | 3.7.0 |

Matplotlib, requests, and tqdm were used, but their versions were not included in the frozen version table. Do not invent version pins for them. Record any environment changes in a new experiment's metadata.

For a fresh runtime:

1. Confirm a live kernel with a newly executed timestamp cell. Saved notebook outputs are not live-state evidence.
2. Restore the recorded dependencies and mount Drive. Handle a required kernel restart before restoring Python state.
3. Define the project paths and read `data_source.json`, `modelling_config.json`, `matches.parquet`, and stage-8 metadata.
4. Load `features_stage_08.parquet` as `model_data`. Restore `FEATURE_GROUPS` from `features_stage_08.json` and validate the 45,233 IDs, eight groups, and 67 features.
5. Load only the fitted models and prediction tables needed for the requested task. Recover ordered features from their metadata. Recreate helper functions from the notebook as needed.
6. If cross-validation indices are required, align saved assignments to the current training table by `shot_id`. Reconstruct folds from their saved IDs and check chronology. Do not trust an earlier pandas index.

Do not restore raw events, re-engineer all features, or retrain all models just to inspect results or continue modelling from the completed checkpoint.

Checkpoint files restore data and metadata, not Python functions or every prior variable. Useful historical names include `model_data`, `FEATURE_GROUPS`, `stage8_metadata`, `training_data`, `validation_data`, `modelling_config`, `tuning_summary`, `STACK_MODEL_ORDER`, `stack_oof_inputs`, `raw_stack_pipeline`, and `saved_calibration_bundle`.

The final verification cells used `reloaded_base_models`, `reloaded_base_predictions`, `validation_by_id`, `reloaded_meta_pipeline`, `frozen_manifest`, `test_evaluation_data`, `test_metrics`, `test_paired_comparison`, and `test_calibration_bins`. Check existence and alignment before reusing any of them.

`save_project_artifact` writes via a temporary file locally, then copies via a temporary file to Drive. Preserve this backup behaviour for new artifacts. Avoid generic loop leftovers such as `candidate_pipeline`, `probabilities`, or `calibrator` when a specific fitted model is required.

## VS Code and Colab extension workflow

The user cloned the repository and intends to edit its notebook locally while executing Python on a hosted Colab runtime. The inspected local clone was `/home/chihab/Desktop/contextual-xg`; this is a computer path, not a Colab path.

- The notebook and Git working tree are local. Notebook cells execute on the selected Colab kernel.
- Files written under `/content` are on the remote runtime. They do not automatically appear in the local Git checkout.
- Google Drive holds persistent project artifacts under `/content/drive/MyDrive/xg-python` after mounting.
- Local supporting files can be sent through the extension's Upload to Colab action. Results intended for a Git commit must be brought back into the local checkout.
- Install runtime Python dependencies in the selected remote kernel. Installing a package locally does not install it into Colab.
- Commit notebook and source changes through local Git or VS Code Source Control. Keep large raw data and fitted model files outside normal Git history.

The official extension supports kernel selection through Select Kernel > Colab and Drive mounting through `Colab: Mount Google Drive to Server...`.

Reference: <https://github.com/googlecolab/colab-vscode/wiki/User-Guide>

### Historical connection failure on 2026-11-09

Last inspected versions were VS Code 1.144.0, Google Colab extension 0.9.7, and Jupyter extension 2025.9.1. These are observations from this session, not a requirement to downgrade future installations.

VS Code repeatedly reported unavailable T4, A100, G4, and L4 accelerators. The extension-host log contained:

```text
LongRunningOperationError:
CreateAssignment: generic::resource_exhausted:
assignWithRetry: after 40 retries:
context deadline exceeded (reason: NO_RUNTIMES)
```

Browser Colab also reported no GPU backend, then its offered CPU fallback returned `Sorry, no backends available. Please try again later.` The screenshot still displayed T4 / High-RAM. An explicit standard-RAM CPU request was suggested afterward, but its outcome was not confirmed. The user chose to retry later.

These historical observations establish an allocation failure in both clients. They do not establish a specific account quota, depleted balance, or a global outage. A subsequent user-run Colab resume session succeeded on 2026-11-10. Check current connection state before proposing more fixes, and preserve any active session the user needs.

A CPU runtime can support result inspection, plots, and many development tasks. Existing GPU-specific XGBoost cells require an appropriate GPU or an explicitly documented CPU adaptation. CPU inference has the small numerical differences recorded above.

## Changing the project safely

Begin with the user's requested modification and inspect the existing implementation. Preserve uncommitted edits and use local Git diffs to review changes.

For code organisation or refactoring, retain ordered features, fitted preprocessing, split assignments, shot alignment, probability dtypes, and the frozen model parameters. Use the recorded validation predictions to check that an intended behaviour-preserving change reproduces the baseline.

For modelling experiments, use distinct experiment IDs and artifacts. Fit learned preprocessing and model choices using the permitted training/validation protocol. Keep the original frozen manifest and exported test result intact. Describe post-test model development honestly and plan fresh independent evaluation when making a new confirmatory claim.

Markdown documentation, resume/reproduction modes, and separate analysis notebooks are implemented. A README, dependency files, a project-specific gitignore, reusable Python modules, and dedicated training/inference/evaluation entry points remain possible improvements. Implement these when they serve a requested change, rather than restarting the entire project automatically.

The repository has an MIT licence for original code. StatsBomb data and third-party dependencies retain their own terms. Preserve StatsBomb attribution and consult its data terms when publishing or redistributing data-derived material.

## Completion boundary

Completed and confirmed: all eight feature groups and checkpoints, 32 cumulative baselines, training-fold tuning, four saved tuned refits, chronological stacking, calibration selection and finalization, end-to-end saved-model validation checks, a frozen manifest, held-out test predictions and metrics, paired match-bootstrap intervals, two evaluation figures, Drive backups, and GitHub publication of the selected exports.

Also completed: main-notebook cleanup, a successful user-run Colab resume check, and three executed analysis notebooks with inspected figures and tested aggregation or selector behavior.

Still open: validation of a complete run through the cleaned reproduction branch, optional repository documentation and packaging, and any future model development. Current live runtime availability must be checked when needed. None of these requires repeating the completed experiment by default.

## Appendix: ordered raw feature registry

The following lists reproduce the stage-8 selected model's raw feature order, partitioned by the original group sizes. They are included for navigation. Saved checkpoint metadata and the frozen manifest remain the files to verify when loading an actual model.

```python
FEATURE_GROUPS = {
    "01_shot_geometry": [
        "distance_to_goal",
        "angle_to_goal",
        "absolute_lateral_distance",
        "shot_side",
        "inside_penalty_area",
        "inside_six_yard_box"
    ],
    "02_shot_and_event_context": [
        "shot_type",
        "body_part",
        "technique",
        "play_pattern",
        "under_pressure",
        "first_time",
        "one_on_one",
        "open_goal",
        "aerial_won",
        "follows_dribble"
    ],
    "03_goalkeeper_positioning": [
        "goalkeeper_visible",
        "goalkeeper_depth",
        "goalkeeper_lateral_offset",
        "goalkeeper_distance_to_goal",
        "goalkeeper_distance_to_shot",
        "goalkeeper_line_offset"
    ],
    "04_defensive_pressure": [
        "nearest_visible_defender_distance",
        "second_nearest_visible_defender_distance",
        "visible_defenders_within_1",
        "visible_defenders_within_2",
        "visible_defenders_within_3",
        "visible_defenders_within_5"
    ],
    "05_shot_obstruction": [
        "visible_defenders_in_shot_cone",
        "nearest_visible_blocker_distance",
        "visible_defenders_near_shot_line",
        "nearest_visible_defender_line_offset"
    ],
    "06_attacking_support_and_numerical_balance": [
        "nearest_visible_teammate_distance",
        "visible_teammates_within_5",
        "visible_teammates_within_10",
        "visible_teammates_within_15",
        "local_attacking_share_within_5",
        "local_attacking_share_within_10",
        "local_attacking_share_within_15"
    ],
    "07_assist_and_final_pass_context": [
        "has_assisting_pass",
        "pass_length",
        "pass_height",
        "pass_body_part",
        "pass_type",
        "pass_under_pressure",
        "pass_cross",
        "pass_cut_back",
        "pass_through_ball",
        "pass_switch",
        "seconds_from_pass_to_shot",
        "pass_forward_distance",
        "pass_end_to_shot_distance",
        "pass_to_goal_center_angle",
        "pass_enters_penalty_area",
        "pass_origin_zone"
    ],
    "08_possession_and_preceding_action_context": [
        "possession_team_events_before_shot",
        "possession_passes_before_shot",
        "possession_carries_before_shot",
        "possession_dribbles_before_shot",
        "observed_possession_seconds",
        "previous_team_event_type",
        "seconds_since_previous_team_event_start",
        "prior_shot_within_5_seconds",
        "seconds_since_previous_shot",
        "possession_start_distance_to_goal",
        "possession_forward_progress",
        "completed_carry_distance_before_shot"
    ]
}
```
