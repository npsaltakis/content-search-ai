import sqlite3
from pathlib import Path

import psutil
import streamlit as st

_DEFAULT_DB = Path(__file__).resolve().parents[1] / "content_search_ai.db"

MODALITY_ICON = {
    "Text → Image":   "💬🖼️",
    "Text → PDF":     "💬📄",
    "Text → Audio":   "💬🎧",
    "Audio → Emotion":"🎭🎧",
}


def _load_recent_searches(db_path: str, limit: int = 10):
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("""
            SELECT query, modality, searched_at
            FROM search_logs
            ORDER BY searched_at DESC
            LIMIT ?
        """, (limit,))
        rows = cur.fetchall()
        conn.close()
        return rows
    except Exception:
        return []


def _status_badge(status):
    return {
        "Running": "🟡 Running",
        "Idle": "🟢 Idle",
        "Error": "🔴 Error",
    }.get(status, "⚪ Unknown")


def render_dashboard(db, get_watchdog, db_path: str = str(_DEFAULT_DB)):
    st.subheader("📊 System Dashboard")

    if st.button("🔄 Refresh Now"):
        st.rerun()

    cpu_percent = psutil.cpu_percent(interval=0.3)
    ram_percent = psutil.virtual_memory().percent

    images_wd = get_watchdog(db, "images")
    pdfs_wd = get_watchdog(db, "pdfs")
    audio_wd = get_watchdog(db, "audio")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="dash-card">
                <h3>🧠 System Overview</h3>
                <p><strong>CPU Usage:</strong> {:.1f}%</p>
                <p><strong>RAM Usage:</strong> {:.1f}%</p>
            </div>
            """.format(cpu_percent, ram_percent),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="dash-card">
                <h3>📊 Indexed Content</h3>
                <p>🖼 Images: <strong>{db.count_images()}</strong></p>
                <p>📄 PDF Pages: <strong>{db.count_pdf_pages()}</strong></p>
                <p>🎧 Audio Files: <strong>{db.count_audio()}</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col3, col4 = st.columns(2)

    with col3:
        st.markdown(
            f"""
            <div class="dash-card">
                <h3>🖼 Images Watchdog</h3>
                <p>Status: <strong>{_status_badge(images_wd["status"])}</strong></p>
                <p>Last event: {images_wd["last_event"]}</p>
                <p>Processed files: {images_wd["processed"]}</p>
                {f"<p style='color:#ff6b6b'>Error: {images_wd['error']}</p>" if images_wd["error"] else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="dash-card">
                <h3>📄 PDFs Watchdog</h3>
                <p>Status: <strong>{_status_badge(pdfs_wd["status"])}</strong></p>
                <p>Last event: {pdfs_wd["last_event"]}</p>
                <p>Processed pages: {pdfs_wd["processed"]}</p>
                {f"<p style='color:#ff6b6b'>Error: {pdfs_wd['error']}</p>" if pdfs_wd["error"] else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="dash-card">
            <h3>🎧 Audio Watchdog</h3>
            <p>Status: <strong>{_status_badge(audio_wd["status"])}</strong></p>
            <p>Last event: {audio_wd["last_event"]}</p>
            <p>Processed files: {audio_wd["processed"]}</p>
            {f"<p style='color:#ff6b6b'>Error: {audio_wd['error']}</p>" if audio_wd["error"] else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Recent Search History ─────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🕓 Recent Search History")

    recent = _load_recent_searches(db_path)

    if not recent:
        st.info("No searches yet. Run a query from any search tab.")
    else:
        for query, modality, searched_at in recent:
            icon = MODALITY_ICON.get(modality, "🔍")
            ts   = searched_at[:16].replace("T", " ") if searched_at else "—"
            st.markdown(
                f"{icon} &nbsp; **{query}** &nbsp;&nbsp; "
                f"<span style='color:gray;font-size:0.85em'>{modality} · {ts}</span>",
                unsafe_allow_html=True,
            )
