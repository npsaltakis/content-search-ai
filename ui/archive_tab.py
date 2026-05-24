"""
Archive Browser Tab
Displays all indexed files (images, PDFs, audio) from the SQLite database.
Read-only — does not touch the retrieval core.
"""

import sqlite3
from pathlib import Path

import streamlit as st


# ── helpers ──────────────────────────────────────────────────────────────────

EMOTION_COLORS = {
    "happy":   "🟡",
    "sad":     "🔵",
    "angry":   "🔴",
    "disgust": "🟤",
    "fearful": "🟣",
    "neutral": "⚪",
}


def _load_images(db_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT filename, filepath FROM images ORDER BY filename")
    rows = cur.fetchall()
    conn.close()
    return rows


def _load_pdfs(db_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT pdf_path, COUNT(*) as pages
        FROM pdf_pages
        GROUP BY pdf_path
        ORDER BY pdf_path
    """)
    rows = cur.fetchall()
    conn.close()
    return rows


def _load_audio(db_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT a.audio_path, e.emotion
        FROM audio_embeddings a
        LEFT JOIN audio_emotions e ON a.audio_path = e.audio_path
        ORDER BY a.audio_path
    """)
    rows = cur.fetchall()
    conn.close()
    return rows


# ── main render ───────────────────────────────────────────────────────────────

def render_archive_tab(db_path: str, base_dir: Path):
    st.subheader("🗂️ Archive Browser")
    st.caption("Read-only view of all indexed files in the system.")

    images = _load_images(db_path)
    pdfs   = _load_pdfs(db_path)
    audios = _load_audio(db_path)

    # ── top counters ──────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("🖼️ Images",    len(images))
    c2.metric("📄 PDFs",      len(pdfs))
    c3.metric("🎧 Audio files", len(audios))

    st.divider()

    # ── tabs per modality ─────────────────────────────────────────────────────
    img_tab, pdf_tab, aud_tab = st.tabs(["🖼️ Images", "📄 PDFs", "🎧 Audio"])

    # ── IMAGES ───────────────────────────────────────────────────────────────
    with img_tab:
        if not images:
            st.info("No images indexed yet.")
        else:
            cols_per_row = st.select_slider(
                "Thumbnails per row", options=[2, 3, 4, 5, 6], value=6
            )

            cols = st.columns(cols_per_row)
            for idx, (filename, filepath) in enumerate(images):
                full_path = base_dir / filepath
                with cols[idx % cols_per_row]:
                    if full_path.exists():
                        st.image(
                            str(full_path),
                            caption=filename,
                            use_container_width=True,
                        )
                    else:
                        st.warning(f"Missing:\n{filename}", icon="⚠️")

    # ── PDFs ─────────────────────────────────────────────────────────────────
    with pdf_tab:
        if not pdfs:
            st.info("No PDFs indexed yet.")
        else:
            search_pdf = st.text_input("🔍 Filter by name", key="archive_pdf_filter")

            for pdf_path, pages in pdfs:
                name = Path(pdf_path).name
                if search_pdf and search_pdf.lower() not in name.lower():
                    continue

                full_path = base_dir / pdf_path
                col_name, col_pages, col_dl = st.columns([5, 1, 1])

                col_name.markdown(f"📄 **{name}**")
                col_pages.markdown(f"`{pages} σελ.`")

                if full_path.exists():
                    with open(full_path, "rb") as f:
                        col_dl.download_button(
                            label="⬇️",
                            data=f,
                            file_name=name,
                            mime="application/pdf",
                            key=f"dl_{pdf_path}",
                        )
                else:
                    col_dl.markdown("—")

    # ── AUDIO ─────────────────────────────────────────────────────────────────
    with aud_tab:
        if not audios:
            st.info("No audio files indexed yet.")
        else:
            # Emotion filter
            emotions_available = sorted({e for _, e in audios if e})
            sel_emotion = st.selectbox(
                "Filter by emotion",
                ["All"] + emotions_available,
                key="archive_emotion_filter"
            )

            for audio_path, emotion in audios:
                if sel_emotion != "All" and emotion != sel_emotion:
                    continue

                name     = Path(audio_path).name
                full_path = base_dir / audio_path
                emoji    = EMOTION_COLORS.get(emotion or "", "❓")

                col_name, col_emotion, col_play = st.columns([4, 1, 2])
                col_name.markdown(f"🎵 **{name}**")
                col_emotion.markdown(f"{emoji} `{emotion or 'unknown'}`")

                if full_path.exists():
                    fmt = "audio/mpeg" if name.endswith(".mp3") else "audio/wav"
                    with open(full_path, "rb") as f:
                        col_play.audio(f.read(), format=fmt)
                else:
                    col_play.markdown("*file missing*")
