# End-to-End Data Management Pipeline for a Recommendation System

**Student name:** Deepak HariHara Sudan Narayanan  
**Roll number:** 2025AG05345

## Project description

A reproducible data-management and recommendation project for
the DM4ML Assignment-2. 
It creates a **synthetic catalog of exactly 5,000 English-language movies**, synthetic users and ratings, validates and quarantines bad records, builds reusable features, trains two recommenders, compares top-K ranking metrics, and records the run artifacts.

> **Synthetic data notice:** The 5,000 movie rows are generated metadata with
> templated English titles and descriptions; they are not 5,000 real films or
> a licensed public movie database. Every row has `language=English` and
> `language_code=en`. The deterministic dataset is generated locally without
> a download.

## Requirements

- Python 3.10 or later (tested with Python 3.14)
- macOS/Linux shell or Windows PowerShell
- Internet only for installing Python packages; pipeline execution itself is offline
- Dependencies: NumPy and pandas (see [requirements.txt](requirements.txt))

## Run from a fresh checkout

Run commands from the repository root (`DM4ML_recommendation-pipeline/`).

### macOS / Linux

```bash
cd /path/to/DM4ML_recommendation-pipeline
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m src.pipeline
```

### Windows PowerShell

```powershell
cd C:\path\to\DM4ML_recommendation-pipeline
py -3 --version
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m src.pipeline
```

The terminal should end with `STATUS: SUCCESS` and print the catalog count and
two model rows. The pipeline is deterministic for the configured random seed
and overwrites generated CSV/JSON/report artifacts when run again.

### Run the orchestrated workflow

The workflow entry point verifies task dependencies before running the same
pipeline:

```bash
python dags/workflow.py
```

On macOS/Linux, `dags/run_daily.sh` can be scheduled with cron after the
virtual environment has been created. Example (replace the paths):

```cron
0 2 * * * /absolute/path/to/DM4ML_recommendation-pipeline/dags/run_daily.sh >> /absolute/path/to/DM4ML_recommendation-pipeline/reports/scheduled_run.log 2>&1
```

Set `PYTHON=/path/to/python` in the wrapper's environment if the virtual
environment is stored elsewhere. On Windows, schedule
`.venv\Scripts\python.exe dags\workflow.py` with Task Scheduler.

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Project structure

```text
configs/default.json              Dataset size, seed, split, and metric settings
data/raw/                         Generated users, movies, and interactions
data/processed/                   Clean data, temporal split, matrix and aggregates
data/features/                    Reusable user, movie, and interaction features
data/errors/                      Quarantined invalid users, movies, interactions
src/ingestion/                    Synthetic English movie and interaction generator
src/validation/                   Schema, data-quality, language and FK checks
src/preparation/                  Cleaning and chronological train/test split
src/transformation/               Matrix, user activity and movie popularity
src/features/                     Training-only reusable feature tables
src/training/                     Popularity and personalized content recommenders
src/evaluation/                   Precision, Recall, NDCG, Coverage, Diversity
src/storage/                      Future PostgreSQL schema and adapter placeholder
src/reporting/                    Final report Markdown/PDF generator
dags/workflow.py                  Dependency-aware orchestration entry point
models/                           Run metadata and top-K recommendations
reports/                          Validation, evaluation, architecture and report
tests/                            Unit tests for pipeline behavior
```

## Pipeline stages and assignment requirements

1. **Ingestion** — creates 500 user profiles, 5,000 English movie metadata
   records and 16,000 synthetic valid ratings by default, plus five deliberate
   invalid interaction examples. Ratings are generated from genre preferences;
   five catalog items per genre are reserved for a controlled future-event
   holdout, while each user rates the other four during training. This makes
   the personalized ranking task reproducible and measurable. User, movie,
   and event IDs are stable and deterministic for a fixed seed.
2. **Validation** — checks required fields, missing values, duplicate keys,
   numeric types, rating bounds 1–5, parseable timestamps, English-language
   policy, and interaction foreign keys. Bad rows are separated into
   `data/errors/invalid_users.csv`, `invalid_movies.csv`, and
   `invalid_interactions.csv`; details are recorded in the validation report
   and log.
3. **Preparation and transformation** — imputes missing ages, normalizes
   fields, preserves UTC timestamps, and holds out at least the configured
   latest interaction per user. All events tied at the cutoff timestamp are
   held out, avoiding same-time leakage. Derived files include train/test
   data, user activity, movie popularity, and a user-item rating matrix.
4. **Feature management** — computes user activity/mean rating/preferred
   genre/time since activity, movie popularity/mean rating/time since
   activity, and interaction-level joined profile and movie fields. Definitions
   are in [reports/feature_catalog.md](reports/feature_catalog.md). All
   aggregates are computed from training data only.
5. **Model training** — trains a popularity baseline and a personalized
   rating-weighted genre content recommender. Popularity breaks ties and
   handles users with no training history. Parameters, metrics, and artifact
   paths are saved in `models/experiment_run.json` (a lightweight local
   experiment tracker; no tracking server required).
6. **Model evaluation** — compares Precision@5, Recall@5, NDCG@5, catalog
   Coverage, and genre Diversity. Ratings >= 4 are relevant. All test users
   are included; users without a relevant held-out event contribute zero
   Recall and NDCG. Models rank movies rather than predict rating values, so
   RMSE/MAE are not applicable.
7. **Orchestration** — `dags/workflow.py` runs the ordered dependency graph;
   the Python stages are deterministic and can be scheduled with cron or
   Task Scheduler.

## View generated outputs

After a successful run, open these files from the repository root:

| Output | Location |
|---|---|
| Raw 5,000-movie catalog | `data/raw/movies.csv` |
| Raw user profiles and ratings | `data/raw/users.csv`, `data/raw/interactions.csv` |
| Quality report/log | `reports/validation_report.json`, `reports/validation.log` |
| Error datasets | `data/errors/` |
| Train/test split and derived tables | `data/processed/` |
| Reusable features | `data/features/` |
| Model comparison table | `reports/model_evaluation.md`, `reports/model_evaluation.csv` |
| Recommended movie IDs by user/model | `models/recommendations.json` |
| Experiment parameters and metrics | `models/experiment_run.json` |
| Successful workflow transcript | `reports/workflow_run.txt` |
| Architecture and feature catalog | `reports/architecture.md`, `reports/feature_catalog.md` |
| Final report | `reports/final_report.md`, `reports/final_report.pdf` |

For the assignment's successful-run screenshot, run the pipeline in a visible
terminal and capture the output containing `STATUS: SUCCESS`.

## Configuration

Edit `configs/default.json` to adjust the seed, users, movie count,
interactions per user, featured movies per genre, temporal holdout, K, and relevance threshold. The
default generates exactly 5,000 movie rows and each user's valid event count
must exceed the configured holdout count. Tied timestamps can make the actual
holdout larger.


## Repository

This local clone's remote is
[`2025ag05345git/DM4ML_recommendation-pipeline`](https://github.com/2025ag05345git/DM4ML_recommendation-pipeline).
Pushes require write permission to that repository.
