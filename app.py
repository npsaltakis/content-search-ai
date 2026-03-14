import os
import sqlite3
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from core import AudioSearcher, ImageSearcher, PDFSearcher
from core.db.database_helper import DatabaseHelper
from core.explainability import (
    estimate_computational_summary,
    summary_to_lines,
    build_results_table,
)
from ui import (
    ABOUT_PROJECT_MARKDOWN,
    VERSION_HISTORY_MARKDOWN,
    apply_global_styles,
    render_app_header,
    render_dashboard,
    render_image_to_image_tab,
    render_pdf_to_pdf_tab,
    render_text_to_image_tab,
    render_text_to_pdf_tab,
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

apply_global_styles()

# ======================================================
# 🚀 INITIALIZATION
# ======================================================
render_app_header(BASE_DIR)

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
    render_dashboard(db, get_watchdog)

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
            st.markdown(ABOUT_PROJECT_MARKDOWN)

            # ======================================================
            # 📘 VERSION HISTORY
            # ======================================================
    with st.container():
        with st.expander("📘 Version History", expanded=False):
            st.markdown(VERSION_HISTORY_MARKDOWN)

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
    render_text_to_image_tab(DB_PATH, get_image_searcher, top_k)


# ======================================================
# 🖼️ IMAGE → IMAGE SEARCH
# ======================================================
with tabs[4]:
    render_image_to_image_tab(DB_PATH, get_image_searcher, top_k)

# ======================================================
# 💬 TEXT → PDF SEARCH
# ======================================================
with tabs[5]:
    render_text_to_pdf_tab(top_k)

# ======================================================
# 📚 PDF → PDF SEARCH
# ======================================================
with tabs[6]:
    render_pdf_to_pdf_tab(top_k)

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

