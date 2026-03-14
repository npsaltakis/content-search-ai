# Content Search AI

AI-powered multimodal search in digital multimedia archives.

This repository contains a university thesis project focused on content-based retrieval across:
- images
- PDF documents
- audio files

The system uses embeddings, transcript processing, emotion analysis, SQLite storage, and a Streamlit interface to support semantic search across multiple media types.

Version: `v1.8`

---

## Overview

The goal of the project is to support semantic retrieval in multimedia archives without relying only on filenames or simple keyword matching.

The application indexes archive content, stores searchable representations in SQLite, and exposes retrieval through a web interface built with Streamlit.

Supported capabilities:
- Text -> Image search
- Image -> Image similarity search
- Text -> PDF semantic retrieval
- PDF -> PDF similarity search
- Text -> Audio semantic retrieval
- Emotion -> Audio filtering
- Real-time filesystem monitoring and indexing
- Explainable retrieval results

---

## Core Design Principles

- Embedding-based retrieval as the main search mechanism
- Clear separation between retrieval and explainability
- Unified local database for indexed content
- Real-time archive synchronization through filesystem watchers
- Practical multimodal pipeline suitable for thesis experimentation and demonstration

---

## How It Works

### 1. Archive indexing
The system monitors the archive folders under `data/` and processes incoming files:
- images are converted into visual embeddings
- PDFs are split into pages and encoded as text embeddings
- audio files are transcribed, embedded, and classified by emotion

### 2. Database storage
The extracted representations are stored in a local SQLite database:
- image metadata and embeddings
- PDF page text and embeddings
- audio embeddings and emotion metadata

### 3. Retrieval
At query time, the system compares the query representation with stored vectors and returns the most relevant results.

### 4. Explainability
The user interface displays supporting evidence such as:
- similarity scores
- confidence indicators
- matched PDF paragraphs
- detected audio emotion and emotion probabilities

Important: explainability is intended to improve transparency and does not change the ranking logic.

---

## Search Modes

### Image Retrieval
- Text -> Image
- Image -> Image
- Uses CLIP image embeddings and M-CLIP text embeddings

### PDF Retrieval
- Text -> PDF pages
- PDF -> PDF similarity
- Uses page-level semantic embeddings
- Returns page-level results with paragraph-based evidence

### Audio Retrieval
- Text -> Audio semantic retrieval
- Emotion -> Audio filtering
- Audio is first transcribed with Whisper
- Search is performed over transcript embeddings
- Emotion labels and probabilities are used for filtering and explainability

Note: the audio pipeline is transcript-based retrieval with emotion analysis, not direct raw-audio embedding retrieval.

---

## Architecture Summary

Main application flow:
1. Start the application from `main.py`
2. Download required model files if they are missing
3. Run an initial synchronization of archive folders
4. Start watchdog services for images, PDFs, and audio
5. Launch the Streamlit interface from `app.py`

Main components:
- `main.py`: startup orchestration, sync, watchdog processes, Streamlit launch
- `app.py`: Streamlit user interface
- `core/image_search.py`: image retrieval logic
- `core/pdf_search.py`: PDF semantic retrieval and document similarity
- `core/audio_search.py`: audio search over transcript embeddings plus emotion metadata
- `core/watchdog/`: real-time indexing services
- `core/db/database_helper.py`: SQLite schema and data access layer

---

## Project Structure

```text
content-search-ai/
|-- app.py
|-- main.py
|-- README.md
|-- requirements.txt
|-- environment.yml
|-- content_search_ai.db
|-- models/
|-- assets/
|-- core/
|   |-- __init__.py
|   |-- image_search.py
|   |-- pdf_search.py
|   |-- audio_search.py
|   |-- emotion_model_v5.py
|   |-- explainability.py
|   |-- model.py
|   |-- db/
|   |   `-- database_helper.py
|   `-- watchdog/
|       |-- sync_manager.py
|       |-- watch_images_other.py
|       |-- watch_pdfs.py
|       `-- watch_audio_other.py
`-- data/
    |-- images/
    |-- pdfs/
    |-- audio/
    |-- query/
    `-- query_images/
```

---

## Database Schema

Database file:
- `content_search_ai.db`

Main tables currently used by the application:
- `images`: image metadata and embeddings
- `pdf_pages`: PDF page text and embeddings
- `audio_embeddings`: audio transcript embeddings
- `audio_emotions`: detected emotion labels and emotion probabilities
- `search_logs`: stored search history
- `watchdog_status`: indexing/watchdog state tracking

Note: the README now reflects the current implementation used by the codebase.

---

## Installation

### Conda environment

```bash
conda env create -f environment.yml
conda activate content-search-ai
```

### pip environment

```bash
pip install -r requirements.txt
```

### Model files

The repository does not store the full model assets inside Git because of their size.

During the first run, the application checks the `models/` folder and downloads the required model files from external storage only if they are missing.

If the model files are already present locally, they are reused and no download is performed.

---

## Main Dependencies

Key libraries used in the project:
- PyTorch
- Sentence Transformers
- OpenAI CLIP
- Faster-Whisper
- PyMuPDF
- Streamlit
- Watchdog
- SQLite

---

## Documentation Guide

Key project documentation:

- `docs/ARCHITECTURE.md`: system design, indexing flow, retrieval flow, and explainability
- `docs/RUNBOOK.md`: setup, startup, demo checklist, and troubleshooting
- `docs/EVALUATIONS_FULL_EL.md`: complete Greek-language evaluation analysis for thesis and oral defense
- `evaluation/README.md`: practical evaluation workspace guide
- `evaluation/EVALUATION_STATUS.md`: modality-by-modality evaluation status
- `docs/DEFENSE_PREP_EL.md`: practical Greek-language defense preparation notes
- `docs/THESIS_QA_EL.md`: likely thesis questions and answers in Greek
- `docs/THEORY_MODELS_QA_EL.md`: theory-focused model questions in Greek
- `docs/THESIS_QA_EL.md`: likely thesis questions and answers in Greek
- `docs/THEORY_MODELS_QA_EL.md`: theory-focused model questions in Greek

---

## Running the Application

```bash
python main.py
```

Then open:

`http://localhost:8501`

What happens on startup:
1. required model files are checked and downloaded if missing
2. initial indexing runs for images, PDFs, and audio
3. watchdog services start for real-time updates
4. the Streamlit interface is launched

---

## Data Folders

The application expects archive data in:
- `data/images`
- `data/pdfs`
- `data/audio`

Runtime query uploads are stored in:
- `data/query`
- `data/query_images`

---

## Explainability Layer

Each modality provides additional result context:
- computational summary
- top-k result tables
- confidence indicators
- evidence snippets or labels

Examples:
- images: similarity strength and confidence
- PDFs: best-matching paragraph per page
- audio: detected emotion and emotion probabilities

Explainability is designed for transparency only and does not alter the retrieval ranking.

---

## Current Scope

The current thesis implementation focuses on:
- small-to-medium archive indexing
- local execution
- multimodal retrieval across three media types
- explainable result presentation
- real-time monitoring of archive changes

---

## Limitations

Current practical limitations include:
- startup depends on local environment configuration
- model distribution currently relies on external Google Drive hosting because large model assets are not stored in the repository
- retrieval thresholds are heuristic and can be improved through evaluation
- large-scale indexing optimization is not yet the main focus

These are good candidates for future thesis improvements.

---

## Future Work

Planned or possible next steps:
- video retrieval support
- OCR for scanned PDFs
- larger-scale vector indexing with FAISS
- stronger evaluation metrics for retrieval quality
- improved architecture modularization
- richer explainability and result visualization

---

## Author

**Nikolaos Psaltakis**  
University of West Attica  
Department of Computer Science

---

## License

Academic use only.
