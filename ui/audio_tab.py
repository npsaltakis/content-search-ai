from pathlib import Path

import pandas as pd
import streamlit as st

from core.explainability import estimate_computational_summary, summary_to_lines


def render_audio_tab(get_audio_searcher, top_k):
    st.subheader("🎧 Text-to-Audio Search (Semantic + Emotion)")

    audio_searcher = get_audio_searcher()

    st.markdown(
        """
        #### 🎨 Color Guide
        - 🎭 Emotion color shows detected dominant emotion
        """
    )

    if "run_audio_search" not in st.session_state:
        st.session_state.run_audio_search = False

    def trigger_audio_search():
        st.session_state.run_audio_search = True

    query = st.text_input(
        "🔎 Enter search text or emotion (e.g. happy, θυμός)",
        on_change=trigger_audio_search,
    )

    if st.button("Run Audio Search", use_container_width=True):
        st.session_state.run_audio_search = True

    if st.session_state.run_audio_search:
        results = []
        emotion_only = False

        if not query.strip():
            st.warning("⚠️ Please enter a query.")
        else:
            with st.spinner("Searching audio…"):
                emotion_keywords = {
                    "happy", "sad", "angry", "fearful",
                    "disgust", "neutral",
                    "χαρά", "λύπη", "θυμός", "φόβος", "αηδία",
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

            indexed_items = None
            for attr in ("num_items", "n_items", "total_items", "index_size", "audio_count"):
                if hasattr(audio_searcher, attr):
                    try:
                        indexed_items = int(getattr(audio_searcher, attr))
                        break
                    except Exception:
                        pass

            compared_items = indexed_items if indexed_items is not None else len(results)
            embedding_dim = 512

            summary = estimate_computational_summary(
                query=f"Audio query: {query}",
                results=results,
                indexed_items=(indexed_items if indexed_items is not None else len(results)),
                embedding_dim=embedding_dim,
                compared_items=compared_items,
                top_k=top_k,
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
                    language="text",
                )

            with st.expander("📊 Numerical Results (Top-K)", expanded=False):
                rows = []
                for idx, result in enumerate(results, start=1):
                    audio_path = result.get("audio_path", "")
                    rows.append({
                        "Rank": idx,
                        "Audio": Path(audio_path).name if audio_path else "n/a",
                        "Similarity": round(float(result.get("similarity", 0) or 0), 3),
                        "Emotion": result.get("emotion", "unknown"),
                        "Language": result.get("language", "n/a"),
                    })

                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)

            for result in results:
                audio_path = result.get("audio_path", "")
                full_path = Path(audio_path).as_posix() if audio_path else ""
                fname = Path(full_path).name if full_path else "unknown.wav"

                st.markdown(
                    f"""
                    ### 🎵 {fname}
                    🔊 **Similarity:** `{float(result.get('similarity', 0) or 0):.3f}`  
                    🎭 **Emotion:** `{result.get('emotion', 'unknown')}`  
                    🌐 **Query Language:** `{result.get('language', 'n/a')}`
                    """
                )

                if full_path:
                    try:
                        with open(full_path, "rb") as file_obj:
                            st.audio(file_obj.read(), format="audio/wav")
                        st.caption(full_path)
                    except Exception as exc:
                        st.error(f"Could not load audio: {exc}")
                else:
                    st.warning("⚠️ Missing audio_path in result.")

                if result.get("emotion_probs"):
                    with st.expander("🎭 Emotion probabilities"):
                        st.json(result["emotion_probs"])

                st.markdown("---")

    st.session_state.run_audio_search = False
