import streamlit as st


APP_CSS = """
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

.search-card {
    max-width: 900px;
    margin: 0 auto;
    padding: 25px;
    background: #141414;
    border-radius: 18px;
    border: 1px solid #2a2a2a;
    box-shadow: 0 0 35px rgba(0,0,0,0.45);
}

.result-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 20px;
    margin-top: 25px;
}

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

.fade-in {
    animation: fadeIn 0.4s ease forwards;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}

div[data-testid="stTextInput"] input {
    background: #1c1c1c !important;
    border: 1px solid #2d2d2d !important;
    border-radius: 12px !important;
    padding: 12px 14px !important;
    color: #e6e6e6 !important;
    font-size: 1.05rem !important;
    box-shadow: inset 0 0 10px rgba(0,0,0,0.35) !important;
}

div[data-testid="stTextInput"] > div {
    background: transparent !important;
    padding: 0 !important;
    border: none !important;
    box-shadow: none !important;
}

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
"""


def apply_global_styles():
    st.markdown(APP_CSS, unsafe_allow_html=True)
