import os
import sqlite3
import time

import pandas as pd
import streamlit as st

from core.explainability import (
    build_results_table,
    estimate_computational_summary,
    summary_to_lines,
)


def render_image_to_image_tab(db_path, get_image_searcher, top_k):
    st.subheader("🖼️ Image-to-Image Search")

    uploaded_file = st.file_uploader(
        "📤 Upload an image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        query_image_path = os.path.join("data/query_images", uploaded_file.name)
        os.makedirs(os.path.dirname(query_image_path), exist_ok=True)

        with open(query_image_path, "wb") as file_obj:
            file_obj.write(uploaded_file.getbuffer())

        st.image(query_image_path, caption="📸 Uploaded Image", width=250)

        if st.button("🔍 Run Image Search"):
            st.info("🔍 Analyzing and comparing image...")

            start = time.time()
            image_to_image_searcher = get_image_searcher()
            results = image_to_image_searcher.search_by_image(query_image_path, top_k=top_k)
            elapsed = time.time() - start

            if not results:
                st.warning("❌ No similar images found.")
            else:
                max_score = results[0]["score"]
                margin = 0.35
                threshold = max_score - margin

                valid_results = [result for result in results if result["score"] >= threshold]

                if not valid_results:
                    st.warning("⚠️ No strongly similar images found.")
                else:
                    st.success(
                        f"✅ Found {len(valid_results)} strongly similar images "
                        f"(filtered from {len(results)}) in {elapsed:.2f}s"
                    )

                results = valid_results

                conn = sqlite3.connect(db_path)
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM images")
                indexed_items = cur.fetchone()[0]
                conn.close()

                summary = estimate_computational_summary(
                    query=f"IMAGE: {os.path.basename(query_image_path)}",
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
                        "sim(q_img, i) = (v_q · v_i) / (||v_q|| · ||v_i||)\n"
                        "v_q = ImageEncoder(query_image)\n"
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

                    caption = f"Cosine Similarity: {score:.4f}"
                    if confidence is not None:
                        caption += f"\nConfidence: {confidence * 100:.1f}%"

                    caption += "\nReason: visual embedding similarity"

                    cols[idx].image(
                        result["path"],
                        caption=caption,
                        use_container_width=True,
                    )
