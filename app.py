from pathlib import Path

import streamlit as st
from core import AudioSearcher, ImageSearcher
from core.db.database_helper import DatabaseHelper
from ui import (
    ABOUT_PROJECT_MARKDOWN,
    VERSION_HISTORY_MARKDOWN,
    apply_global_styles,
    render_app_header,
    render_archive_tab,
    render_audio_tab,
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
    "🎧 Search: Text → Audio",
    "🗂️ Archive Browser",
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
    render_audio_tab(get_audio_searcher, top_k)

# ======================================================
# 🗂️ ARCHIVE BROWSER
# ======================================================
with tabs[8]:
    render_archive_tab(str(DB_PATH), BASE_DIR)

