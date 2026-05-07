import sqlite3
import numpy as np
import torch
import clip
from PIL import Image
from sentence_transformers import SentenceTransformer

# =========================================================
# QUERY NORMALIZATION (deterministic, multilingual-safe)
# =========================================================
def normalize_text_image_query(query: str, lang: str) -> str:
    query = query.strip().lower()

    if lang == "el":
        mappings = {
            "άντρας": "man",
            "γυναίκα": "woman",
            "μηχανή": "motorcycle",
            "μοτοσυκλέτα": "motorcycle",
            "καβαλάει": "riding",
            "οδηγεί": "riding",
            "πάνω σε": "on",
        }
        for gr, en in mappings.items():
            query = query.replace(gr, en)

    return query

# =========================================================
# PROMPT TEMPLATES (LOCKED)
# =========================================================
PROMPT_TEMPLATES = [
    "a photo of {q}",
    "a picture of {q}",
    "a photo showing {q}",
    "an image of {q}",
]


def build_prompt_variants(q: str):
    prompts = [t.format(q=q) for t in PROMPT_TEMPLATES]

    variants = [q]

    if "man" in q:
        variants.append(q.replace("man", "person"))
    if "woman" in q:
        variants.append(q.replace("woman", "person"))
    if "riding" in q:
        variants.append(q.replace("riding", "on"))

    variants = list(dict.fromkeys(variants))

    for v in variants:
        prompts.extend(t.format(q=v) for t in PROMPT_TEMPLATES)

    return list(dict.fromkeys(prompts))

def detect_language(text: str):
    for ch in text:
        if 'α' <= ch <= 'ω' or 'Α' <= ch <= 'Ω':
            return "el"
    return "en"

# =========================================================
# IMAGE SEARCHER (PURE RETRIEVAL CORE)
# =========================================================
class ImageSearcher:

    def __init__(self, db_path="content_search_ai.db"):
        self.db_path = db_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.image_model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)

        self.text_model = SentenceTransformer(
            "./models/mclip_finetuned_coco_ready",
            device=self.device
        )

    # -----------------------------------------------------
    def _load_sqlite_embeddings(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT filename, filepath, embedding FROM images")
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "filename": filename,
                "path": image_path,
                "vector": np.frombuffer(blob, dtype=np.float32)
            }
            for filename, image_path, blob in rows
        ]

    # -----------------------------------------------------
    # TEXT → IMAGE (GENERIC, PURE SIMILARITY)
    # -----------------------------------------------------
    def search(self, query: str, top_k=5):
        images = self._load_sqlite_embeddings()
        if not images:
            raise RuntimeError("No image embeddings found")

        lang = detect_language(query)
        q_norm = normalize_text_image_query(query, lang)
        prompts = build_prompt_variants(q_norm)

        query_embs = self.text_model.encode(
            prompts,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        # -----------------------------
        # Coarse similarity distribution
        # -----------------------------
        sims = np.array([
            max(float(np.dot(q, img["vector"])) for q in query_embs)
            for img in images
        ])

        mean, std = sims.mean(), sims.std()
        MIN_SIM = mean + 0.3 * std   # adaptive, generic

        # -----------------------------
        # Final ranking
        # -----------------------------
        results = []
        for img, sim in zip(images, sims):
            if sim < MIN_SIM:
                continue

            confidence = (sim - mean) / (std + 1e-6)
            confidence = float(np.clip(confidence, 0.0, 1.0))

            results.append({
                "filename": img["filename"],
                "score": float(sim),
                "confidence": confidence,
                "path": img["path"]
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    # -----------------------------------------------------
    # IMAGE → IMAGE (PURE)
    # -----------------------------------------------------
    def search_by_image(self, query_image_path: str, top_k=5):
        images = self._load_sqlite_embeddings()

        image = self.preprocess(
            Image.open(query_image_path).convert("RGB")
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q = self.image_model.encode_image(image)
            q /= q.norm(dim=-1, keepdim=True)
            q = q.cpu().numpy().flatten()

        sims = np.array([
            float(np.dot(q, img["vector"]))
            for img in images
        ])

        mean, std = sims.mean(), sims.std()
        MIN_SIM = mean + 0.3 * std  # ίδιο philosophy με Text→Image

        results = []
        for img, sim in zip(images, sims):
            if sim < MIN_SIM:
                continue

            confidence = (sim - mean) / (std + 1e-6)
            confidence = float(np.clip(confidence, 0.0, 1.0))

            results.append({
                "filename": img["filename"],
                "score": float(sim),
                "confidence": confidence,
                "path": img["path"],
                "explain": "Matched via global visual similarity using CLIP image embeddings"
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
