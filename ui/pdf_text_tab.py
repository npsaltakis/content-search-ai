import os

import pandas as pd
import streamlit as st

from core import PDFSearcher
from core.explainability import estimate_computational_summary, summary_to_lines


def render_text_to_pdf_tab(top_k):
    st.subheader("💬 Text-to-PDF Semantic Search")

    query_text = st.text_area(
        "✍️ Enter your search text:",
        placeholder="e.g. deep learning in medical imaging",
    )

    if st.button("🔍 Run Text → PDF Search"):
        if not query_text.strip():
            st.warning("⚠️ Please enter text before searching.")
        else:
            st.info(f"🔍 Searching for: '{query_text}' ...")

            from ui.search_logger import log_search
            text_to_pdf_searcher = PDFSearcher(db_path="content_search_ai.db")

            with st.spinner("Processing and comparing PDFs..."):
                results = text_to_pdf_searcher.search_by_text(
                    query_text=query_text,
                    top_k=top_k,
                )
            log_search(query_text, "Text → PDF")

            if not results:
                st.warning("❌ No matching PDFs found.")
            else:
                st.success(f"✅ Found {len(results)} relevant PDF pages")

                indexed_items = len(results)

                summary = estimate_computational_summary(
                    query=query_text,
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
                        "sim(q, p_i) = (t · v_i) / (||t|| · ||v_i||)\n"
                        "t = TextEncoder(query_text)\n"
                        "v_i = TextEncoder(pdf_page_i)",
                        language="text",
                    )

                with st.expander("📊 Numerical Results (Top-K)", expanded=False):
                    table_rows = []
                    for idx, result in enumerate(results, start=1):
                        table_rows.append({
                            "Rank": idx,
                            "PDF": os.path.basename(result["pdf"]),
                            "Page": result["page"],
                            "Similarity (%)": round(result["score"] * 100, 2),
                            "Confidence (%)": round(result["confidence"] * 100, 1),
                        })

                    df = pd.DataFrame(table_rows)
                    st.dataframe(df, use_container_width=True, hide_index=True)

                for result in results:
                    filename = os.path.basename(result["pdf"])

                    st.markdown(
                        f"""
                        ### 📄 {filename} — Page {result['page']}
                        **Similarity:** `{result['score'] * 100:.2f}%`  
                        **Confidence:** `{result['confidence'] * 100:.1f}%`  
                        **Reason:** semantic text embedding similarity
                        """
                    )

                    if result.get("matched_paragraph"):
                        st.markdown("**Most semantically similar paragraph:**")
                        st.info(result["matched_paragraph"])
                    else:
                        st.caption("No paragraph-level match available.")

                    with open(result["pdf"], "rb") as file_obj:
                        pdf_data = file_obj.read()

                    st.download_button(
                        label=f"⬇️ Download {filename}",
                        data=pdf_data,
                        file_name=filename,
                        mime="application/pdf",
                        key=f"download_{filename}_{result['page']}",
                    )

                    st.markdown("---")
