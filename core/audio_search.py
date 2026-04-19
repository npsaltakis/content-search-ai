import sqlite3
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# ==================================================
# Utils
# ==================================================
def detect_language(text: str) -> str:
    for ch in text:
        if 'α' <= ch <= 'ω' or 'Α' <= ch <= 'Ω':
            return "el"
    return "en"


def normalize_emotion_query(q: str) -> str:
    q = q.lower().strip()
    mapping = {
        "happy": "happy", "joy": "happy", "χαρά": "happy",
        "sad": "sad", "λυπημένος": "sad", "λύπη": "sad",
        "angry": "angry", "θυμός": "angry",
        "fear": "fearful", "fearful": "fearful", "φόβος": "fearful",
        "disgust": "disgust", "αηδία": "disgust",
        "neutral": "neutral", "ουδέτερος": "neutral",
    }
    return mapping.get(q, q)


# ==================================================
# AudioSearcher
# ==================================================
class AudioSearcher:
    """
    DB-first Audio Searcher
    - Semantic retrieval via M-CLIP (text -> audio transcript embeddings)
    - Emotion metadata for filtering & explainability
    """

    def __init__(
        self,
        db_path: str = "content_search_ai.db",
        mclip_model_path: str = "models/mclip_finetuned_coco_ready",
    ):
        self.db_path = db_path

        # M-CLIP ALWAYS CPU (safe loading)
        self.mclip = SentenceTransformer(
            mclip_model_path,
            device="cpu"
        )

    # ==================================================
    # LOAD AUDIO DATA FROM DB
    # ==================================================
    def _load_audio_rows(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("""
            SELECT
                a.audio_path,
                a.embedding,
                em.emotion,
                em.emotion_probs
            FROM audio_embeddings a
            LEFT JOIN audio_emotions em
                ON a.audio_path = em.audio_path
        """)

        rows = cur.fetchall()
        conn.close()

        results = []

        for audio_path, emb_blob, emotion, emotion_probs_json in rows:
            if emb_blob is None:
                continue

            vec = np.frombuffer(emb_blob, dtype=np.float32)

            emotion_probs = None
            if emotion_probs_json:
                try:
                    emotion_probs = json.loads(emotion_probs_json)
                except Exception:
                    emotion_probs = None

            results.append({
                "audio_path": audio_path,
                "vector": vec,
                "emotion": emotion,
                "emotion_probs": emotion_probs
            })

        return results

    # ==================================================
    # SEMANTIC SEARCH
    # ==================================================
    def search_semantic(self, query: str, top_k: int = 5):
        rows = self._load_audio_rows()
        if not rows:
            return []

        # Encode query (normalized)
        qvec = self.mclip.encode(
            query,
            normalize_embeddings=True
        ).astype(np.float32)

        # Cosine similarity (dot since normalized)
        sims = np.array([
            float(np.dot(qvec, r["vector"]))
            for r in rows
        ], dtype=np.float32)

        mean = float(sims.mean())
        std = float(sims.std())

        # Safe adaptive threshold
        MIN_SIM = mean if std == 0 else mean + 0.3 * std

        results = []
        for r, sim in zip(rows, sims):
            if sim < MIN_SIM:
                continue

            results.append({
                "audio_path": r["audio_path"],
                "similarity": float(sim),
                "emotion": r["emotion"],
                "emotion_probs": r["emotion_probs"],
                "language": detect_language(query)
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    # ==================================================
    # EMOTION SEARCH
    # ==================================================
    def search_by_emotion(self, emotion_query: str, top_k: int = 5):
        emotion = normalize_emotion_query(emotion_query)

        rows = self._load_audio_rows()
        matches = [
            r for r in rows
            if r["emotion"] == emotion
        ]

        return [{
            "audio_path": r["audio_path"],
            "emotion": r["emotion"],
            "emotion_probs": r["emotion_probs"]
        } for r in matches[:top_k]]
