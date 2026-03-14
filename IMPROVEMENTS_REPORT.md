# Improvements Report

## Purpose

This file records proposed improvements for the thesis project so they can be used as a reference during the next development steps.

---

## 1. Critical Improvements

### 1.1 Improve startup portability and reproducibility
- The project correctly depends on a prepared Python/Conda environment, which is expected for an AI application.
- However, the application startup in `main.py` uses a hardcoded Streamlit executable path.
- This can fail on different operating systems, different Conda locations, different environment names, or different user setups.
- The issue is not the requirement for an environment itself, but the environment-specific startup method.
- A more portable startup approach would improve reproducibility and make the project easier to run on other machines.

Status:
- Completed
- `main.py` now launches Streamlit through the active Python environment using `python -m streamlit run app.py`
- automatic browser opening was preserved with an explicit local URL open step

Possible improvements:
- launch Streamlit through `python -m streamlit run app.py`
- use the active environment instead of a hardcoded absolute executable path

Expected benefit:
- the application startup becomes more portable
- the project becomes easier to run on different machines and operating systems
- the launcher becomes less dependent on one specific local setup

### 1.2 External model hosting as a deployment limitation
- The project depends on large model assets that are not stored inside the Git repository.
- The required models are currently downloaded from Google Drive through `core/model.py` when they are missing locally.
- This creates dependency on internet access, download quotas, and link availability.
- This is a practical distribution limitation, not necessarily a flaw in the retrieval logic itself.
- For the current thesis/demo version, this is an acceptable trade-off because the model files are too large to keep inside the repository.

Status:
- Partially addressed through documentation
- README now explains that model assets are kept outside Git and are downloaded only when missing locally

Possible improvements:
- document this behavior clearly in the README and setup steps
- improve user-facing messages when a model download fails
- keep the current "download only if missing" behavior explicit in the documentation
- consider alternative hosting/distribution options in future versions

### 1.3 Missing automated tests
- There is currently no visible automated test layer for:
  - search functionality
  - indexing
  - database consistency
  - application startup
- Adding tests would strengthen reliability and academic credibility.

---

## 2. High-Value Technical Improvements

### 2.1 Refactor the Streamlit application
- `app.py` is large and handles many responsibilities at once.
- It would be better to split it into smaller modules, for example:
  - dashboard
  - search views
  - shared UI helpers
  - service orchestration

Status:
- In progress
- First refactor step completed by extracting shared UI code into the new `ui/` package
- The following were moved out of `app.py`:
  - global styles
  - application header/logo rendering
  - dashboard rendering
  - large static informational content blocks
- Search tabs still remain inside `app.py` and can be extracted gradually in future steps

### 2.2 Improve architectural documentation
- The repository would benefit from a clearer architecture description.
- Useful additions:
  - system flow diagram
  - indexing pipeline diagram
  - retrieval pipeline description
  - explainability layer description

### 2.3 Introduce database versioning or migrations
- The SQLite schema is created directly from code in `core/db/database_helper.py`.
- This works for the current version, but schema evolution will become harder later.
- A migration/versioning strategy would make maintenance safer.

### 2.4 Reduce repeated heavy model loading
- Separate watchdog services load heavy models independently.
- This increases CPU/RAM usage and may reduce scalability.
- Future optimization could reduce duplicated loading costs.

---

## 3. Search Quality and Research Improvements

### 3.1 Validate retrieval thresholds experimentally
- Several retrieval modules use heuristic adaptive thresholds such as `mean + 0.3 * std`.
- This is practical, but should ideally be supported by experiments and evaluation.

### 3.2 Add evaluation metrics
- The thesis would be much stronger with measurable retrieval evaluation.
- Possible metrics:
  - Precision@K
  - Recall@K
  - Mean Average Precision (mAP)
  - manual relevance judgments

### 3.3 Clarify the design of audio retrieval
- Audio search is mainly transcript-based semantic retrieval plus emotion filtering.
- This should be clearly documented as a design choice and not presented as raw audio embedding retrieval.

### 3.4 Improve PDF explainability
- PDF explainability is already useful.
- It could be improved with:
  - highlighted text passages
  - richer snippets
  - better page-level evidence presentation

---

## 4. Consistency and Maintainability Improvements

### 4.1 Align README with the actual database schema
- The README mentions `audio_files`, while the code uses:
  - `audio_embeddings`
  - `audio_emotions`
- Documentation should match the real implementation.

### 4.2 Normalize path handling
- The project uses multiple relative-path assumptions between runtime, database, and filesystem watchers.
- This may cause problems when running the app from a different working directory.

### 4.3 Clean up naming consistency
- Some names reflect prototype evolution rather than final structure.
- More consistent naming would improve clarity for examiners and collaborators.

---

## 5. UI and Presentation Improvements

### 5.1 Strengthen the thesis/demo presentation layer
- The interface is functional, but it could better support demonstration and explanation.
- Useful additions:
  - "How it works" section
  - "Evaluation results" section
  - dataset statistics per modality

### 5.2 Separate retrieval from explainability more clearly
- The UI should make it obvious that explainability supports interpretation only.
- It should not appear to affect ranking.

### 5.3 Add more visual presentation of the pipeline
- A visible workflow summary in the app or documentation would improve usability and presentation quality.

---

## 6. Suggested Roadmap

### Phase 1
- Fix startup portability
- Align README and implementation details
- Improve documentation and reproducibility

### Phase 2
- Refactor `app.py`
- Separate UI, services, and persistence concerns

### Phase 3
- Add evaluation framework and retrieval metrics
- Create benchmark-oriented testing scenarios

### Phase 4
- Improve scalability and optimization
- Enhance explainability and presentation quality

---

## 7. General Thesis Assessment

The project already has a strong foundation because it includes:
- multimodal retrieval
- database-backed indexing
- explainability support
- real-time archive monitoring

The biggest opportunity is not only adding new features, but making the system:
- more reliable
- more measurable
- better documented
- easier to reproduce

This will improve both the technical quality of the software and the academic strength of the thesis.
