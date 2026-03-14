import base64
from pathlib import Path

import streamlit as st


def render_app_header(base_dir: Path):
    logo_path = base_dir / "assets" / "images" / "logo.png"
    logo_base64 = None

    if logo_path.exists():
        with open(logo_path, "rb") as file_obj:
            logo_base64 = base64.b64encode(file_obj.read()).decode("utf-8")
    else:
        st.warning(f"⚠️ Logo not found at {logo_path}")

    if not logo_base64:
        return

    st.markdown(
        f"""
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
        """,
        unsafe_allow_html=True,
    )
