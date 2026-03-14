ABOUT_PROJECT_MARKDOWN = """
This system is a **unified multimodal semantic retrieval platform** capable of searching across  
**Images, PDFs, Audio, and Text**, all within a **single shared embedding space**.

The platform is designed with **research-grade architectural principles**, focusing on:
- **Pure embedding-based retrieval**
- **Strict separation between retrieval and explainability**
- **Multilingual support without translation models**
- **Database-first indexing and querying**
- **No heuristic rules, boosts, or hard constraints**

It supports the following retrieval modes:
- **Text → Image**
- **Image → Image**
- **Text → PDF**
- **PDF → PDF**
- **Text → Audio (via transcripts)**
- **Emotion-based Audio Filtering**

As of **v1.8**, the retrieval core is considered **final, stable, and locked**.

---
### 🧩 Technologies Used
- **Python 3.11**
- **Streamlit**
- **SQLite3 (Unified Multimodal DB)**
- **PyTorch**
- **Sentence-Transformers**
- **CLIP / M-CLIP (ViT-B/32)**
- **OpenAI Whisper & Faster-Whisper**
- **Emotion Model V5**
- **PyMuPDF**
- **Watchdog (real-time indexing)**

---
### ⚙️ Model Architecture Overview
- **M-CLIP (ViT-B/32)**  
  → Unified multilingual embedding space for text, images, PDFs, and audio transcripts

- **CLIP Image Encoder**  
  → Image → Image similarity using pure visual embeddings

- **Whisper-small / Faster-Whisper**  
  → Audio transcription (indexing only)

- **Emotion Model V5**  
  → 6-class emotion classification (angry, disgust, fearful, happy, neutral, sad)

- **PDF Page Encoder**  
  → Per-page semantic embeddings with paragraph-level explainability

All retrieval operations rely **exclusively on cosine similarity** between normalized embeddings.
"""


VERSION_HISTORY_MARKDOWN = """
## 🟢 **v1.8 — Retrieval Core Stabilization & Explainability Lock**  
**(December 2025)**

This release finalizes the **semantic retrieval architecture** and ensures full correctness,
consistency, and explainability across all supported modalities.

### 🔥 Key Improvements (This Session)

#### 🧠 Retrieval Core Finalization
- Confirmed **pure cosine similarity retrieval** across:
  - Images
  - PDFs
  - Audio
- No usage of:
  - keywords
  - filename rules
  - domain heuristics
  - task-specific boosts
- Adaptive similarity thresholding unified across all modalities.
- Retrieval logic is **modality-agnostic and symmetric**.

---

#### 📄 PDF Search — Explainability Completion
- Retrieval unit finalized as **PDF page embeddings**.
- Ranking based solely on **page-level semantic similarity**.
- Added **paragraph-level explainability**:
  - The most semantically similar paragraph is identified per page.
  - Paragraph selection does **not affect ranking**.
- Confidence score:
  - Derived from similarity distribution
  - Used **only for UI explainability**
  - Never affects ranking or filtering

---

#### 🎧 Audio Search — DB-First Architecture
- Fully migrated audio retrieval to **SQLite-only runtime**.
- Audio embeddings loaded exclusively from:
  - `audio_embeddings`
  - `audio_emotions`
- Whisper used **only during indexing**, never during search.
- Emotion metadata:
  - Stored as probabilities
  - Used optionally for filtering and explainability
- Safe and deterministic model loading (no meta tensors).

---

#### 🎭 Emotion Model V5 — Locked Integration
- Emotion inference finalized as **pure post-processing**.
- No interaction with semantic similarity.
- 6 fixed emotion classes.
- Emotion probabilities exposed for **explainability only**.

---

#### 🧩 Architectural Principles Enforced
- Strict separation between:
  - Retrieval core
  - Explainability layer
  - UI rendering
- Unified retrieval pipeline for all modalities:
  1. Encode
  2. Compare
  3. Rank
  4. Explain (non-intrusive)

This version marks the point where the system is considered:
- **Architecturally complete**
- **Retrieval-correct**
- **Explainable without bias**
- **Ready for academic documentation**

No further changes are planned for the retrieval core.

---
## 🟢 **v1.7 — Full Multimodal SQLite Integration & Real-Time Indexing**  
**(November 2025)**

- Unified SQLite database for all modalities:
  - `images`
  - `pdf_pages`
  - `audio_embeddings`
  - `audio_emotions`
- Removed all local embedding and transcript caches.
- Introduced Watchdog-based real-time indexing.
- Automatic DB updates on file create/delete.
- Full path normalization.
- Major codebase cleanup.

---
## 🟢 **v1.6 — Audio Search Integration**
- Whisper transcription
- M-CLIP audio semantic search
- Emotion Model V5
- Audio visualization

---
## 🟢 **v1.5 — Stable PDF Search**
- Page-level PDF processing
- Document similarity
- UI improvements

---
## 🟠 **v1.4 — Core Integration**
- Modular UI
- Cache system
- Layout refactor

---
## 🟡 **v1.3 — M-CLIP Adoption**
- Multilingual unified embeddings

---
## 🔵 **v1.2 — Visual Search Prototype**
- Text → Image
- Image → Image

---
## ⚫ **v1.0 — Project Initialization**
"""
