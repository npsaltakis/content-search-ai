# Runbook

This runbook explains how to set up, run, verify, and troubleshoot `content-search-ai` without changing the application code or UI.

---

## 1. Purpose

This document is intended to support:
- thesis demonstrations
- local reproduction of the project
- environment setup on a new machine
- quick recovery if startup or indexing fails

---

## 2. Prerequisites

Recommended environment:
- Python environment created from `environment.yml`
- local write access to the project folder
- enough disk space for the `models/` folder and indexed data

Expected project folders:
- `data/images`
- `data/pdfs`
- `data/audio`
- `models/`

---

## 3. Environment Setup

### Option A: Conda

```bash
conda env create -f environment.yml
conda activate content-search-ai
```

### Option B: pip

```bash
pip install -r requirements.txt
```

Notes:
- the Conda environment is the preferred setup for this project
- the project can run on Windows, but the environment must be created first
- the application startup now uses the active Python environment rather than a hardcoded Streamlit path

---

## 4. Model Files

The repository does not store full model assets in Git because of their size.

At startup, the application checks the local `models/` directory:
- if the required model files are already present, they are reused
- if they are missing, the application attempts to download them from external storage

Practical implication:
- the first run may take longer
- internet access may be required on the first run
- later runs reuse the local model files

---

## 5. Data Preparation

Before starting the application, place archive content in the expected folders:

- `data/images`
- `data/pdfs`
- `data/audio`

Runtime query folders:
- `data/query`
- `data/query_images`

Recommended practice before a demo:
- verify that each folder contains the intended files
- avoid adding large new batches of files right before the presentation
- make sure the archive content has already been indexed at least once

---

## 6. Standard Startup Procedure

Run:

```bash
python main.py
```

Expected startup sequence:
1. the application checks required model files
2. missing models are downloaded if necessary
3. an initial sync runs for images, PDFs, and audio
4. watchdog services start for ongoing filesystem monitoring
5. Streamlit launches on `http://127.0.0.1:8501`
6. the browser opens automatically on the local machine

---

## 7. Recommended Demo Procedure

Before the actual presentation:
1. activate the correct environment
2. verify that the `models/` folder already exists and contains the downloaded assets
3. verify that the database file `content_search_ai.db` exists
4. run `python main.py`
5. wait for initial sync completion
6. confirm the Streamlit UI opens successfully
7. execute 1-2 known-safe sample queries before the live presentation begins

This reduces the chance of first-run surprises during the defense.

---

## 8. What to Verify Before a Presentation

Quick checklist:
- environment activates successfully
- `python main.py` starts without import errors
- `models/` exists locally
- `content_search_ai.db` exists
- indexed content is visible in the UI
- at least one text-to-image query works
- at least one text-to-PDF query works
- at least one text-to-audio or emotion-audio query works
- browser opens or can be opened manually at `http://127.0.0.1:8501`

---

## 9. Troubleshooting

### Problem: Streamlit does not open automatically
Check whether the app is still running in the terminal. If needed, open the browser manually at:

`http://127.0.0.1:8501`

### Problem: Startup fails because of missing models
Likely causes:
- first run without internet access
- interrupted model download
- incomplete `models/` folder

Suggested action:
- rerun the application in the correct environment
- verify network access if this is the first setup
- confirm that model files were written to `models/`

### Problem: New files do not appear in search
Likely causes:
- indexing has not completed yet
- the files were placed in the wrong folder
- the file type is unsupported or failed during ingestion

Suggested action:
- confirm the file was copied into the correct `data/` folder
- restart the app if needed
- check whether the database was updated after sync

### Problem: Audio results are limited
This project currently uses transcript-based audio retrieval plus emotion metadata. If the indexed audio set is small, retrieval variety will also be limited.

### Problem: Database schema mismatch after future changes
The project now includes lightweight schema versioning in `core/db/database_helper.py`. If future schema changes are introduced, confirm the migration logic before reusing an older database file.

---

## 10. Safe Recovery Steps

If the application behaves unexpectedly shortly before a demo:
1. stop the running app cleanly
2. reactivate the intended environment
3. verify that `models/` still exists
4. restart with `python main.py`
5. test one known-safe query per modality

Avoid making structural code changes immediately before a presentation.

---

## 11. Reproducibility Notes for the Thesis

The project is reproducible at thesis scale if the following are available:
- the source code
- the Python environment
- the required model files
- archive content placed under `data/`

Important limitation:
- model hosting currently depends on external storage rather than Git-tracked assets

This is acceptable for the current thesis prototype, but it should be documented clearly.

---

## 12. Related Documentation

For more details, see:
- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/EVALUATION_PLAN.md`
- `docs/EVALUATION_RESULTS.md`
- `evaluation/README.md`
- `docs/DEFENSE_PREP_EL.md`
