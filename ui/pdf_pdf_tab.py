import os

import pandas as pd
import streamlit as st

from core import PDFSearcher
from core.explainability import estimate_computational_summary, summary_to_lines


def render_pdf_to_pdf_tab(top_k):
    st.subheader("📚 PDF-to-PDF Similarity Search")

    uploaded_pdf = st.file_uploader(
        "📤 Upload a PDF to compare",
        type=["pdf"],
    )

    base_folder = "./data/pdfs"
    query_folder = "./data/query"

    os.makedirs(base_folder, exist_ok=True)
    os.makedirs(query_folder, exist_ok=True)

    if uploaded_pdf is not None:
        query_path = os.path.join(query_folder, uploaded_pdf.name)
        with open(query_path, "wb") as file_obj:
            file_obj.write(uploaded_pdf.getbuffer())

        st.success(f"✅ Uploaded: {uploaded_pdf.name}")
        st.info("🔍 Analyzing document similarity...")

        pdf_to_pdf_searcher = PDFSearcher(db_path="content_search_ai.db")

        with st.spinner("Processing and comparing PDFs..."):
            results = pdf_to_pdf_searcher.search_similar_pdfs(
                query_pdf_path=query_path,
                top_k=top_k,
            )

        if not results:
            st.warning("❌ No strong matches found.")
        else:
            st.success(f"✅ Found {len(results)} similar documents")

            indexed_items = len(results)

            summary = estimate_computational_summary(
                query=f"PDF: {uploaded_pdf.name}",
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
                    "sim(q_pdf, d_i) = (v_q · v_i) / (||v_q|| · ||v_i||)\n"
                    "v_q = PDFEncoder(query_document)\n"
                    "v_i = PDFEncoder(document_i)",
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
                color = (
                    "🟢" if result["score"] >= 0.98
                    else "🟠" if result["score"] >= 0.95
                    else "🔴"
                )

                st.markdown(
                    f"""
                    ### {color} {filename} — Page {result['page']}
                    **Similarity:** `{result['score'] * 100:.2f}%`  
                    **Confidence:** `{result['confidence'] * 100:.1f}%`  
                    **Reason:** document-level semantic embedding similarity
                    """
                )

                if os.path.basename(result["pdf"]) == uploaded_pdf.name:
                    st.caption("ℹ️ Self-match: the query PDF exists in the archive.")

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
