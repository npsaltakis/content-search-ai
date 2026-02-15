import time
import streamlit as st
import base64
from pathlib import Path
from core import ImageSearcher, PDFSearcher, Model, AudioSearcher
import psutil
from core.db.database_helper import DatabaseHelper
import os
import sqlite3
import pandas as pd
from core.explainability import (
    estimate_computational_summary,
    summary_to_lines,
    build_results_table,
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "content_search_ai.db"

db = DatabaseHelper(str(DB_PATH))

#lazy loading
def get_image_searcher():
    if "image_searcher" not in st.session_state:
        st.session_state.image_searcher = ImageSearcher()
    return st.session_state.image_searcher

def get_audio_searcher():
    if "audio_searcher" not in st.session_state:
        st.session_state.audio_searcher = AudioSearcher()
    return st.session_state.audio_searcher

def get_watchdog(db, name):
    row = db.get_watchdog_status(name)
    if not row:
        return None

    return {
        "status": row["status"],
        "last_event": row["last_event"],
        "last_updated": row["last_updated"],
        "processed": row["processed_count"],
        "error": row["error"],   # ✅ ΣΩΣΤΟ KEY
    }

# ======================================================
# 🧠 STREAMLIT CONFIGURATION
# ======================================================
st.set_page_config(
    page_title="Search Content in Multimedia Digital Archives using AI",
    layout="wide"
)

# ======================================================
# 🎨 CUSTOM CSS STYLING
# ======================================================
st.markdown("""
<style>
/* DASHBOARD GRID & CARDS */
.dashboard-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 20px;
    margin-top: 10px;
    margin-bottom: 20px;
}

.dash-card {
    background: #141414;
    border-radius: 18px;
    border: 1px solid #2a2a2a;
    padding: 18px 20px;
    box-shadow: 0 0 25px rgba(0,0,0,0.4);
    min-height: 120px;
}

.dash-card h3 {
    margin: 0 0 8px 0;
    font-size: 1.1rem;
}

.dash-card p {
    margin: 0;
    font-size: 0.9rem;
    color: #999;
}

[data-testid="stExpander"] {
    background-color: #141414;
    padding: 0;
    border-radius: 16px;
    border: 1px solid #2a2a2a;
    margin-bottom: 5px !important;
    box-shadow: 0 0 25px rgba(0,0,0,0.5), inset 0 0 12px rgba(255,255,255,0.03);
}

[data-testid="stExpander"] > details {
    border-radius: 16px !important;
}

[data-testid="stExpanderDetails"] {
    padding: 20px;
}

.section-title h2 {
    margin-bottom: 0;
}
.section-title p {
    margin-top: -5px;
    color: #aaa;
}

/* CARD */
.search-card {
    max-width: 900px;
    margin: 0 auto;
    padding: 25px;
    background: #141414;
    border-radius: 18px;
    border: 1px solid #2a2a2a;
    box-shadow: 0 0 35px rgba(0,0,0,0.45);
}

/* GRID */
.result-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 20px;
    margin-top: 25px;
}

/* IMAGE CARD */
.result-card {
    position: relative;
    background-color: #1b1b1b;
    border-radius: 16px;
    overflow: hidden;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}

.result-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 0 25px rgba(255,255,255,0.18);
}

.result-card img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

/* BADGE */
.badge {
    position: absolute;
    top: 10px;
    left: 10px;
    background: rgba(0,0,0,0.8);
    padding: 5px 10px;
    font-size: 0.85rem;
    border-radius: 8px;
    color: #ffd700;
    font-weight: bold;
}

/* OVERLAY */
.overlay {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 8px;
    background: linear-gradient(180deg, transparent, rgba(0,0,0,0.9));
    text-align: center;
    color: #ddd;
    font-size: 0.9rem;
}

/* ANIMATION */
.fade-in {
    animation: fadeIn 0.4s ease forwards;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}

/* 🟣 STYLE ONLY THE REAL INPUT FIELD */
div[data-testid="stTextInput"] input {
    background: #1c1c1c !important;
    border: 1px solid #2d2d2d !important;
    border-radius: 12px !important;
    padding: 12px 14px !important;
    color: #e6e6e6 !important;
    font-size: 1.05rem !important;
    box-shadow: inset 0 0 10px rgba(0,0,0,0.35) !important;
}

/* Prevent ugly wrapper from turning into dark box */
div[data-testid="stTextInput"] > div {
    background: transparent !important;
    padding: 0 !important;
    border: none !important;
    box-shadow: none !important;
}

/* Label styling */
div[data-testid="stTextInput"] label {
    font-size: 0.95rem !important;
    color: #ffb86c !important;
    margin-bottom: 6px !important;
    background: none !important;
}

.dash-card {
    margin-bottom: 18px;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# 🚀 INITIALIZATION
# ======================================================
# Path του logo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(BASE_DIR, "assets", "images", "logo.png")

# Μετατροπή εικόνας σε base64 για inline εμφάνιση
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode("utf-8")
else:
    st.warning(f"⚠️ Logo not found at {logo_path}")

# Εμφάνιση inline logo + text
st.markdown(f"""
<div style="display:flex;align-items:center;gap:25px;margin-top:-10px;margin-bottom:20px;">
    <img src="data:image/png;base64,{logo_base64}" width="100" style="border-radius:10px;"/>
    <div>
        <h1 style="margin-bottom:0;">Content Search AI</h1>
        <p style="margin-top:4px;color:#9aa0a6;font-size:1.1rem;">
            Search Content in Multimedia Digital Archives using AI
        </p>
        <p style="margin-top:-8px;color:#9aa0a6;font-size:0.9rem;">Version 1.8</p>
    </div>
</div>
""", unsafe_allow_html=True)

DATA_DIR = "./data"

# searcher = ImageSearcher()
# audio = AudioSearcher()
# pdf = PDFSearcher()
# ======================================================
# 🧭 TABS SETUP
# ======================================================
tabs = st.tabs([
    "📊 Dashboard",
    "ℹ️ Application Info",
    "⚙️ Application Settings",
    "💬 Search: Text → Image",
    "🖼️ Search: Image → Image",
    "💬 Search: Text → PDF",
    "📚 Search: PDF → PDF",
    "🎧 Search: Text → Audio"
])

# ======================================================
# 📊 DASHBOARD
# ======================================================
with tabs[0]:
    st.subheader("📊 System Dashboard")

    if st.button("🔄 Refresh Now"):
        st.rerun()

    cpu_percent = psutil.cpu_percent(interval=0.3)
    ram_percent = psutil.virtual_memory().percent

    images_wd = get_watchdog(db, "images")
    pdfs_wd   = get_watchdog(db, "pdfs")
    audio_wd  = get_watchdog(db, "audio")

    def status_badge(status):
        return {
            "Running": "🟡 Running",
            "Idle": "🟢 Idle",
            "Error": "🔴 Error",
        }.get(status, "⚪ Unknown")

    # ===============================
    # ROW 1 — SYSTEM + INDEX
    # ===============================
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="dash-card">
            <h3>🧠 System Overview</h3>
            <p><strong>CPU Usage:</strong> {:.1f}%</p>
            <p><strong>RAM Usage:</strong> {:.1f}%</p>
        </div>
        """.format(cpu_percent, ram_percent), unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="dash-card">
            <h3>📊 Indexed Content</h3>
            <p>🖼 Images: <strong>{db.count_images()}</strong></p>
            <p>📄 PDF Pages: <strong>{db.count_pdf_pages()}</strong></p>
            <p>🎧 Audio Files: <strong>{db.count_audio()}</strong></p>
        </div>
        """, unsafe_allow_html=True)

    # ===============================
    # ROW 2 — IMAGES + PDFS
    # ===============================
    col3, col4 = st.columns(2)

    with col3:
        st.markdown(f"""
        <div class="dash-card">
            <h3>🖼 Images Watchdog</h3>
            <p>Status: <strong>{status_badge(images_wd["status"])}</strong></p>
            <p>Last event: {images_wd["last_event"]}</p>
            <p>Processed files: {images_wd["processed"]}</p>
            {f"<p style='color:#ff6b6b'>Error: {images_wd['error']}</p>" if images_wd["error"] else ""}
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="dash-card">
            <h3>📄 PDFs Watchdog</h3>
            <p>Status: <strong>{status_badge(pdfs_wd["status"])}</strong></p>
            <p>Last event: {pdfs_wd["last_event"]}</p>
            <p>Processed pages: {pdfs_wd["processed"]}</p>
            {f"<p style='color:#ff6b6b'>Error: {pdfs_wd['error']}</p>" if pdfs_wd["error"] else ""}
        </div>
        """, unsafe_allow_html=True)

    # ===============================
    # ROW 3 — AUDIO (FULL WIDTH)
    # ===============================
    st.markdown(f"""
    <div class="dash-card">
        <h3>🎧 Audio Watchdog</h3>
        <p>Status: <strong>{status_badge(audio_wd["status"])}</strong></p>
        <p>Last event: {audio_wd["last_event"]}</p>
        <p>Processed files: {audio_wd["processed"]}</p>
        {f"<p style='color:#ff6b6b'>Error: {audio_wd['error']}</p>" if audio_wd["error"] else ""}
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# ℹ️ APPLICATION INFORMATION TAB
# ======================================================
with tabs[1]:
    st.subheader("ℹ️ Application Information")

    # ======================================================
    # 🧠 ABOUT THIS PROJECT
    # ======================================================
    with st.container():
        with st.expander("🧠 About This Project", expanded=True):
            st.markdown("""
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
                    """)

            # ======================================================
            # 📘 VERSION HISTORY
            # ======================================================
    with st.container():
        with st.expander("📘 Version History", expanded=False):
            st.markdown("""
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
        """)

# ======================================================
# ⚙️ SETTINGS TAB WITH ACCORDIONS
# ======================================================
with tabs[2]:
    st.subheader("⚙️ Application Settings")
    # ------------------------------------------------------
    # DISPLAY SETTINGS
    # ------------------------------------------------------
    with st.expander("🔧 Display Settings", expanded=True):
        top_k = st.slider("Select number of results per search", 3, 30, 5)

    st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# 💬 TEXT → IMAGE SEARCH
# ======================================================
with tabs[3]:
    st.subheader("💬 Text-to-Image Search")

    # ----------------------------------
    # State init
    # ----------------------------------
    if "run_text_search" not in st.session_state:
        st.session_state.run_text_search = False

    def trigger_text_search():
        st.session_state.run_text_search = True

    # ----------------------------------
    # Input
    # ----------------------------------
    query = st.text_input(
        "✍️ Enter your search query",
        value="",
        on_change=trigger_text_search
    )

    if st.button("🔎 Run Text Search"):
        st.session_state.run_text_search = True

    # ----------------------------------
    # Run search
    # ----------------------------------
    if st.session_state.run_text_search:
        if not query.strip():
            st.warning("⚠️ Please enter a search phrase.")
        else:
            st.info(f"🔍 Searching for: '{query}' ...")

            # -------------------------------
            # CORE SEARCH (NO CHANGES)
            # -------------------------------
            text_to_image_searcher = get_image_searcher()
            results = text_to_image_searcher.search(query, top_k=top_k)

            if not results:
                st.warning("❌ No results found.")
            else:
                # ======================================================
                # 🔎 TRUE INDEX SIZE FROM SQLITE (NO GUESSING)
                # ======================================================
                conn = sqlite3.connect(DB_PATH)
                cur = conn.cursor()

                cur.execute("SELECT COUNT(*) FROM images")
                indexed_items = cur.fetchone()[0]

                conn.close()

                # ======================================================
                # 🧠 EXPLAINABILITY: COMPUTATIONAL SUMMARY
                # ======================================================
                summary = estimate_computational_summary(
                    query=query,
                    results=results,
                    indexed_items=indexed_items,      # ✅ REAL N
                    embedding_dim=512,
                    compared_items=indexed_items,     # brute-force over archive
                    top_k=top_k
                )

                with st.expander("🧠 Computational Summary (Explainability)", expanded=False):
                    st.text("\n".join(summary_to_lines(summary)))

                    st.text("\nCosine similarity formula used:\n")
                    st.code(
                        "sim(q, i) = (t · v_i) / (||t|| · ||v_i||)\n"
                        "t = TextEncoder(query)\n"
                        "v_i = ImageEncoder(image_i)",
                        language="text"
                    )

                    st.text(
                        "\nNote on confidence values:\n"
                        "- Confidence is a relative measure based on similarity distribution.\n"
                        "- When few results are returned, confidence may reach high values.\n"
                        "- Confidence does NOT affect ranking."
                    )

                # ======================================================
                # 📊 NUMERICAL RESULTS (TOP-K)
                # ======================================================
                with st.expander("📊 Numerical Results (Top-K)", expanded=False):
                    rows = build_results_table(results)
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True, hide_index=True)

                # ======================================================
                # 🖼️ IMAGES
                # ======================================================
                cols = st.columns(len(results))

                for idx, r in enumerate(results):
                    score = r["score"]
                    confidence = r.get("confidence", None)

                    explain_text = ""
                    if confidence is not None:
                        if confidence >= 0.7:
                            explain_text = "🟢 High semantic relevance"
                        elif confidence >= 0.4:
                            explain_text = "🟡 Partial semantic match"
                        else:
                            explain_text = "🔴 Low confidence – weak semantic overlap"

                    caption = f"Cosine Similarity: {score:.4f}"
                    if confidence is not None:
                        caption += f"\nConfidence: {confidence * 100:.1f}%"
                        caption += f"\n{explain_text}"

                    cols[idx].image(
                        r["path"],
                        caption=caption,
                        use_container_width=True
                    )

    # ----------------------------------
    # Reset trigger
    # ----------------------------------
    st.session_state.run_text_search = False


# ======================================================
# 🖼️ IMAGE → IMAGE SEARCH
# ======================================================
with tabs[4]:
    st.subheader("🖼️ Image-to-Image Search")

    uploaded_file = st.file_uploader(
        "📤 Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        query_image_path = os.path.join("data/query_images", uploaded_file.name)
        os.makedirs(os.path.dirname(query_image_path), exist_ok=True)

        with open(query_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(query_image_path, caption="📸 Uploaded Image", width=250)

        if st.button("🔍 Run Image Search"):
            st.info("🔍 Analyzing and comparing image...")

            # -------------------------------
            # CORE SEARCH (NO CHANGES)
            # -------------------------------
            start = time.time()
            image_to_image_searcher = get_image_searcher()
            results = image_to_image_searcher.search_by_image(query_image_path, top_k=top_k)
            elapsed = time.time() - start

            if not results:
                st.warning("❌ No similar images found.")
            else:
                # --------------------------------------
                # ADAPTIVE THRESHOLD (NO CORE CHANGE)
                # --------------------------------------
                max_score = results[0]["score"]
                margin = 0.35  # safe margin for image-to-image
                threshold = max_score - margin

                valid_results = [r for r in results if r["score"] >= threshold]

                if not valid_results:
                    st.warning("⚠️ No strongly similar images found.")
                else:
                    st.success(
                        f"✅ Found {len(valid_results)} strongly similar images "
                        f"(filtered from {len(results)}) in {elapsed:.2f}s"
                    )

                results = valid_results

                # ======================================================
                # 🔎 TRUE INDEX SIZE FROM SQLITE
                # ======================================================
                conn = sqlite3.connect(DB_PATH)
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM images")
                indexed_items = cur.fetchone()[0]
                conn.close()

                # ======================================================
                # 🧠 EXPLAINABILITY: COMPUTATIONAL SUMMARY
                # ======================================================
                summary = estimate_computational_summary(
                    query=f"IMAGE: {os.path.basename(query_image_path)}",
                    results=results,
                    indexed_items=indexed_items,
                    embedding_dim=512,
                    compared_items=indexed_items,   # brute-force over archive
                    top_k=top_k
                )

                with st.expander("🧠 Computational Summary (Explainability)", expanded=False):
                    st.text("\n".join(summary_to_lines(summary)))

                    st.text("\nCosine similarity formula used:\n")
                    st.code(
                        "sim(q_img, i) = (v_q · v_i) / (||v_q|| · ||v_i||)\n"
                        "v_q = ImageEncoder(query_image)\n"
                        "v_i = ImageEncoder(image_i)",
                        language="text"
                    )

                    st.text(
                        "\nNote on confidence values:\n"
                        "- Confidence is a relative measure based on similarity distribution.\n"
                        "- When few results are returned, confidence may reach high values.\n"
                        "- Confidence does NOT affect ranking."
                    )

                # ======================================================
                # 📊 NUMERICAL RESULTS (TOP-K)
                # ======================================================
                with st.expander("📊 Numerical Results (Top-K)", expanded=False):
                    rows = build_results_table(results)
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True, hide_index=True)

                # ======================================================
                # 🖼️ VISUAL RESULTS
                # ======================================================
                cols = st.columns(len(results))

                for idx, r in enumerate(results):
                    score = r["score"]
                    confidence = r.get("confidence", None)

                    caption = f"Cosine Similarity: {score:.4f}"
                    if confidence is not None:
                        caption += f"\nConfidence: {confidence * 100:.1f}%"

                    caption += "\nReason: visual embedding similarity"

                    cols[idx].image(
                        r["path"],
                        caption=caption,
                        use_container_width=True
                    )

# ======================================================
# 💬 TEXT → PDF SEARCH
# ======================================================
with tabs[5]:
    st.subheader("💬 Text-to-PDF Semantic Search")

    query_text = st.text_area(
        "✍️ Enter your search text:",
        placeholder="e.g. deep learning in medical imaging"
    )

    if st.button("🔍 Run Text → PDF Search"):
        if not query_text.strip():
            st.warning("⚠️ Please enter text before searching.")
        else:
            st.info(f"🔍 Searching for: '{query_text}' ...")

            text_to_pdf_searcher = PDFSearcher(db_path="content_search_ai.db")

            with st.spinner("Processing and comparing PDFs..."):
                results = text_to_pdf_searcher.search_by_text(
                    query_text=query_text,
                    top_k=top_k
                )

            if not results:
                st.warning("❌ No matching PDFs found.")
            else:
                st.success(f"✅ Found {len(results)} relevant PDF pages")

                # ======================================================
                # 🧠 COMPUTATIONAL SUMMARY (EXPLAINABILITY)
                # ======================================================
                indexed_items = len(results)

                summary = estimate_computational_summary(
                    query=query_text,
                    results=results,
                    indexed_items=indexed_items,
                    embedding_dim=512,
                    compared_items=indexed_items,
                    top_k=top_k
                )

                with st.expander("🧠 Computational Summary (Explainability)", expanded=False):
                    st.text("\n".join(summary_to_lines(summary)))

                    st.text("\nCosine similarity formula used:\n")
                    st.code(
                        "sim(q, p_i) = (t · v_i) / (||t|| · ||v_i||)\n"
                        "t = TextEncoder(query_text)\n"
                        "v_i = TextEncoder(pdf_page_i)",
                        language="text"
                    )

                # ======================================================
                # 📊 NUMERICAL RESULTS TABLE (TOP-K)
                # ======================================================
                with st.expander("📊 Numerical Results (Top-K)", expanded=False):
                    table_rows = []
                    for i, r in enumerate(results, start=1):
                        table_rows.append({
                            "Rank": i,
                            "PDF": os.path.basename(r["pdf"]),
                            "Page": r["page"],
                            "Similarity (%)": round(r["score"] * 100, 2),
                            "Confidence (%)": round(r["confidence"] * 100, 1),
                        })

                    df = pd.DataFrame(table_rows)
                    st.dataframe(df, use_container_width=True, hide_index=True)

                # ======================================================
                # 📄 DETAILED RESULTS (WITH PARAGRAPH EXPLAINABILITY)
                # ======================================================
                for r in results:
                    filename = os.path.basename(r["pdf"])

                    st.markdown(
                        f"""
                        ### 📄 {filename} — Page {r['page']}
                        **Similarity:** `{r['score'] * 100:.2f}%`  
                        **Confidence:** `{r['confidence'] * 100:.1f}%`  
                        **Reason:** semantic text embedding similarity
                        """
                    )

                    # -------- PARAGRAPH-LEVEL EXPLAINABILITY --------
                    if r.get("matched_paragraph"):
                        st.markdown("**Most semantically similar paragraph:**")
                        st.info(r["matched_paragraph"])
                    else:
                        st.caption("No paragraph-level match available.")
                    # ------------------------------------------------

                    with open(r["pdf"], "rb") as f:
                        pdf_data = f.read()

                    st.download_button(
                        label=f"⬇️ Download {filename}",
                        data=pdf_data,
                        file_name=filename,
                        mime="application/pdf",
                        key=f"download_{filename}_{r['page']}"
                    )

                    st.markdown("---")

# ======================================================
# 📚 PDF → PDF SEARCH
# ======================================================
with tabs[6]:
    st.subheader("📚 PDF-to-PDF Similarity Search")

    uploaded_pdf = st.file_uploader(
        "📤 Upload a PDF to compare",
        type=["pdf"]
    )

    base_folder = "./data/pdfs"
    query_folder = "./data/query"

    os.makedirs(base_folder, exist_ok=True)
    os.makedirs(query_folder, exist_ok=True)

    if uploaded_pdf is not None:
        # ----------------------------------
        # Save uploaded PDF
        # ----------------------------------
        query_path = os.path.join(query_folder, uploaded_pdf.name)
        with open(query_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())

        st.success(f"✅ Uploaded: {uploaded_pdf.name}")
        st.info("🔍 Analyzing document similarity...")

        php_to_pdf_searcher = PDFSearcher(db_path="content_search_ai.db")

        with st.spinner("Processing and comparing PDFs..."):
            results = php_to_pdf_searcher.search_similar_pdfs(
                query_pdf_path=query_path,
                top_k=top_k
            )

        if not results:
            st.warning("❌ No strong matches found.")
        else:
            st.success(f"✅ Found {len(results)} similar documents")

            # ======================================================
            # 🧠 COMPUTATIONAL SUMMARY (EXPLAINABILITY)
            # ======================================================
            indexed_items = len(results)

            summary = estimate_computational_summary(
                query=f"PDF: {uploaded_pdf.name}",
                results=results,
                indexed_items=indexed_items,
                embedding_dim=512,
                compared_items=indexed_items,
                top_k=top_k
            )

            with st.expander("🧠 Computational Summary (Explainability)", expanded=False):
                st.text("\n".join(summary_to_lines(summary)))

                st.text("\nCosine similarity formula used:\n")
                st.code(
                    "sim(q_pdf, d_i) = (v_q · v_i) / (||v_q|| · ||v_i||)\n"
                    "v_q = PDFEncoder(query_document)\n"
                    "v_i = PDFEncoder(document_i)",
                    language="text"
                )

            # ======================================================
            # 📊 NUMERICAL RESULTS TABLE (TOP-K)
            # ======================================================
            with st.expander("📊 Numerical Results (Top-K)", expanded=False):
                table_rows = []
                for i, r in enumerate(results, start=1):
                    table_rows.append({
                        "Rank": i,
                        "PDF": os.path.basename(r["pdf"]),
                        "Page": r["page"],
                        "Similarity (%)": round(r["score"] * 100, 2),
                        "Confidence (%)": round(r["confidence"] * 100, 1),
                    })

                df = pd.DataFrame(table_rows)
                st.dataframe(df, use_container_width=True, hide_index=True)

            # ======================================================
            # 📄 DETAILED RESULTS (WITH PARAGRAPH EXPLAINABILITY)
            # ======================================================
            for r in results:
                filename = os.path.basename(r["pdf"])

                color = (
                    "🟢" if r["score"] >= 0.98
                    else "🟠" if r["score"] >= 0.95
                    else "🔴"
                )

                st.markdown(
                    f"""
                    ### {color} {filename} — Page {r['page']}
                    **Similarity:** `{r['score'] * 100:.2f}%`  
                    **Confidence:** `{r['confidence'] * 100:.1f}%`  
                    **Reason:** document-level semantic embedding similarity
                    """
                )

                if os.path.basename(r["pdf"]) == uploaded_pdf.name:
                    st.caption("ℹ️ Self-match: the query PDF exists in the archive.")

                # -------- PARAGRAPH-LEVEL EXPLAINABILITY --------
                if r.get("matched_paragraph"):
                    st.markdown("**Most semantically similar paragraph:**")
                    st.info(r["matched_paragraph"])
                else:
                    st.caption("No paragraph-level match available.")
                # ------------------------------------------------

                with open(r["pdf"], "rb") as f:
                    pdf_data = f.read()

                st.download_button(
                    label=f"⬇️ Download {filename}",
                    data=pdf_data,
                    file_name=filename,
                    mime="application/pdf",
                    key=f"download_{filename}_{r['page']}"
                )

                st.markdown("---")

# ======================================================
# 🎧 AUDIO SEARCH
# ======================================================
with tabs[7]:
    st.subheader("🎧 Text-to-Audio Search (Semantic + Emotion)")

    audio_searcher = get_audio_searcher()

    st.markdown("""
    #### 🎨 Color Guide
    - 🎭 Emotion color shows detected dominant emotion
    """)

    # -------------------------------
    # STATE
    # -------------------------------
    if "run_audio_search" not in st.session_state:
        st.session_state.run_audio_search = False

    def trigger_audio_search():
        st.session_state.run_audio_search = True

    # -------------------------------
    # INPUT
    # -------------------------------
    query = st.text_input(
        "🔎 Enter search text or emotion (e.g. happy, θυμός)",
        on_change=trigger_audio_search
    )

    if st.button("Run Audio Search", use_container_width=True):
        st.session_state.run_audio_search = True

    # -------------------------------
    # SEARCH
    # -------------------------------
    if st.session_state.run_audio_search:

        results = []                 # ✅ ALWAYS defined
        emotion_only = False         # ✅ ALWAYS defined

        if not query.strip():
            st.warning("⚠️ Please enter a query.")
        else:
            with st.spinner("Searching audio…"):

                emotion_keywords = {
                    "happy", "sad", "angry", "fearful",
                    "disgust", "neutral",
                    "χαρά", "λύπη", "θυμός", "φόβος", "αηδία"
                }

                q_norm = query.lower().strip()
                emotion_only = q_norm in emotion_keywords

                if emotion_only:
                    results = audio_searcher.search_by_emotion(query, top_k=top_k) or []
                else:
                    results = audio_searcher.search_semantic(query, top_k=top_k) or []

        if not results:
            st.error("❌ No matching audio found.")
        else:
            st.success(f"✅ Found {len(results)} audio matches!")

            # ======================================================
            # 🧠 COMPUTATIONAL SUMMARY (EXPLAINABILITY)
            # ======================================================
            # Try to infer total indexed audio items without assuming DB tables
            indexed_items = None
            for attr in ("num_items", "n_items", "total_items", "index_size", "audio_count"):
                if hasattr(audio_searcher, attr):
                    try:
                        indexed_items = int(getattr(audio_searcher, attr))
                        break
                    except Exception:
                        pass

            # Fallback: at least show we computed over something (safe)
            compared_items = indexed_items if indexed_items is not None else len(results)

            # Embedding dim for transcript text embeddings (keep 512 as per your system)
            embedding_dim = 512

            summary = estimate_computational_summary(
                query=f"Audio query: {query}",
                results=results,
                indexed_items=(indexed_items if indexed_items is not None else len(results)),
                embedding_dim=embedding_dim,
                compared_items=compared_items,
                top_k=top_k
            )

            with st.expander("🧠 Computational Summary (Explainability)", expanded=False):
                st.text("\n".join(summary_to_lines(summary)))

                if emotion_only:
                    st.text("\nSearch mode: Emotion-only classification\n")
                else:
                    st.text("\nSearch mode: Semantic text → transcript similarity\n")

                st.code(
                    "Semantic mode:\n"
                    "sim(q, a_i) = cosine(TextEncoder(query), TextEncoder(transcript_i))\n\n"
                    "Emotion mode:\n"
                    "emotion_i = argmax EmotionClassifier(audio_i)",
                    language="text"
                )

            # ======================================================
            # 📊 NUMERICAL RESULTS (TOP-K)
            # ======================================================
            with st.expander("📊 Numerical Results (Top-K)", expanded=False):
                rows = []
                for i, r in enumerate(results, start=1):
                    audio_path = r.get("audio_path", "")
                    rows.append({
                        "Rank": i,
                        "Audio": Path(audio_path).name if audio_path else "n/a",
                        "Similarity": round(float(r.get("similarity", 0) or 0), 3),
                        "Emotion": r.get("emotion", "unknown"),
                        "Language": r.get("language", "n/a"),
                    })

                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)

            # ======================================================
            # 🎧 DETAILED RESULTS
            # ======================================================
            for r in results:
                audio_path = r.get("audio_path", "")
                full_path = Path(audio_path).as_posix() if audio_path else ""
                fname = Path(full_path).name if full_path else "unknown.wav"

                st.markdown(f"""
                ### 🎵 {fname}
                🔊 **Similarity:** `{float(r.get("similarity", 0) or 0):.3f}`  
                🎭 **Emotion:** `{r.get("emotion", "unknown")}`  
                🌐 **Query Language:** `{r.get("language", "n/a")}`
                """)

                # AUDIO PLAYER
                if full_path:
                    try:
                        with open(full_path, "rb") as f:
                            st.audio(f.read(), format="audio/wav")
                        st.caption(full_path)
                    except Exception as e:
                        st.error(f"Could not load audio: {e}")
                else:
                    st.warning("⚠️ Missing audio_path in result.")

                # EMOTION PROBABILITIES (EXPLAINABILITY)
                if r.get("emotion_probs"):
                    with st.expander("🎭 Emotion probabilities"):
                        st.json(r["emotion_probs"])

                st.markdown("---")

    st.session_state.run_audio_search = False

