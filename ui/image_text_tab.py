import sqlite3

import pandas as pd
import streamlit as st

from core.explainability import (
    build_results_table,
    estimate_computational_summary,
    summary_to_lines,
)


def render_text_to_image_tab(db_path, get_image_searcher, top_k):
    st.subheader("💬 Text-to-Image Search")

    if "run_text_search" not in st.session_state:
        st.session_state.run_text_search = False

    def trigger_text_search():
        st.session_state.run_text_search = True

    query = st.text_input(
        "✍️ Enter your search query",
        value="",
        on_change=trigger_text_search,
    )

    if st.button("🔎 Run Text Search"):
        st.session_state.run_text_search = True

    if st.session_state.run_text_search:
        if not query.strip():
            st.warning("⚠️ Please enter a search phrase.")
        else:
            st.info(f"🔍 Searching for: '{query}' ...")

            from ui.search_logger import log_search
            text_to_image_searcher = get_image_searcher()
            results = text_to_image_searcher.search(query, top_k=top_k)
            log_search(query, "Text → Image")

            if not results:
                st.warning("❌ No results found.")
            else:
                conn = sqlite3.connect(db_path)
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM images")
                indexed_items = cur.fetchone()[0]
                conn.close()

                summary = estimate_computational_summary(
                    query=query,
                    results=results,
                    indexed_items=indexed_items,
                    embedding_dim=512,
                    compared_items=indexed_items,
                    top_k=top_k,
                )

                with st.expander("🧠 Computational Summary (Explainability)", expanded=False):
                    st.text("\n".join(summary_to_lines(summary)))

                    st.text("\nCosine similarity formula used:\n")
                    st.code(
                        "sim(q, i) = (t · v_i) / (||t|| · ||v_i||)\n"
                        "t = TextEncoder(query)\n"
                        "v_i = ImageEncoder(image_i)",
                        language="text",
                    )

                    st.text(
                        "\nNote on confidence values:\n"
                        "- Confidence is a relative measure based on similarity distribution.\n"
                        "- When few results are returned, confidence may reach high values.\n"
                        "- Confidence does NOT affect ranking."
                    )

                with st.expander("📊 Numerical Results (Top-K)", expanded=False):
                    rows = build_results_table(results)
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True, hide_index=True)

                cols = st.columns(len(results))

                for idx, result in enumerate(results):
                    score = result["score"]
                    confidence = result.get("confidence")

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
                        result["path"],
                        caption=caption,
                        use_container_width=True,
                    )

    st.session_state.run_text_search = False
