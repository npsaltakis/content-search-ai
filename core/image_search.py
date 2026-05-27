import sqlite3
import numpy as np
import torch
import open_clip
from open_clip.tokenizer import HFTokenizer
from PIL import Image
from core.model import ModelManager


# =========================================================
# PROMPT TEMPLATES  (CLIP-style zero-shot)
# =========================================================
PROMPT_TEMPLATES = [
    "a photo of {q}",
    "a picture of {q}",
    "a photo showing {q}",
    "an image of {q}",
]


def build_prompt_variants(q: str) -> list[str]:
    """
    Build multiple prompt variants for a query.
    The model is natively multilingual — no Greek→English translation needed.
    Including the raw query first lets the model use its full multilingual ability.
    """
    q = q.strip()
    prompts = [q]  # raw query (Greek or English)
    prompts += [t.format(q=q) for t in PROMPT_TEMPLATES]
    return list(dict.fromkeys(prompts))  # deduplicate, preserve order


# =========================================================
# IMAGE SEARCHER
# =========================================================
class ImageSearcher:
    """
    Unified multilingual image searcher using OpenCLIP.

    Model: xlm-roberta-large-ViT-H-14  (frozen_laion5b_s13b_b90k)
    - Single model handles BOTH text and image encoding
    - Natively multilingual text encoder (Greek, English, 100+ languages)
    - ViT-H/14 backbone: 1024-dim embeddings, 14px patch size
    - ~+19% zero-shot accuracy vs previous ViT-B/32 setup
    """

    MODEL_NAME     = "xlm-roberta-large-ViT-H-14"
    MODEL_PATH     = "./models/openclip_vit_h14/open_clip_pytorch_model.bin"
    TOKENIZER_PATH = "./models/openclip_vit_h14/tokenizer"

    def __init__(self, db_path="content_search_ai.db"):
        self.db_path = db_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        ModelManager().ensure("openclip_vit_h14")

        print(f"[IMAGE] Loading OpenCLIP '{self.MODEL_NAME}' on {self.device}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.MODEL_NAME,
            pretrained=self.MODEL_PATH,
            device=self.device
        )
        self.model.eval()
        self.tokenizer = HFTokenizer(self.TOKENIZER_PATH, context_length=77)

        # Backwards-compat alias used by watchdog & sync_manager
        self.image_model = self.model
        print("[IMAGE] OpenCLIP ready (local).")

    # --------------------------------------------------
    def _load_sqlite_embeddings(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT filename, filepath, embedding FROM images")
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "filename": filename,
                "path":     image_path,
                "vector":   np.frombuffer(blob, dtype=np.float32)
            }
            for filename, image_path, blob in rows
        ]

    # --------------------------------------------------
    # TEXT → IMAGE
    # --------------------------------------------------
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        images = self._load_sqlite_embeddings()
        if not images:
            raise RuntimeError("No image embeddings found in DB")

        prompts = build_prompt_variants(query)

        # Encode all prompt variants in one batch
        tokens = self.tokenizer(prompts).to(self.device)
        with torch.no_grad():
            query_embs = self.model.encode_text(tokens)
            query_embs = query_embs / query_embs.norm(dim=-1, keepdim=True)
            query_embs = query_embs.cpu().numpy().astype(np.float32)

        # Per-image: max similarity across all prompt variants
        sims = np.array([
            max(float(np.dot(q, img["vector"])) for q in query_embs)
            for img in images
        ])

        mean, std = sims.mean(), sims.std()
        MIN_SIM   = mean + 0.3 * std   # α=0.3 — thesis-validated default
        CONF_LOW  = mean + 0.3 * std
        CONF_HIGH = mean + 3.0 * std

        results = []
        for img, sim in zip(images, sims):
            if sim < MIN_SIM:
                continue

            confidence = float(np.clip(
                (sim - CONF_LOW) / (CONF_HIGH - CONF_LOW + 1e-6),
                0.0, 1.0
            ))
            results.append({
                "filename":   img["filename"],
                "score":      float(sim),
                "confidence": confidence,
                "path":       img["path"],
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    # --------------------------------------------------
    # IMAGE → IMAGE
    # --------------------------------------------------
    def search_by_image(self, query_image_path: str, top_k: int = 5) -> list[dict]:
        images = self._load_sqlite_embeddings()

        image = self.preprocess(
            Image.open(query_image_path).convert("RGB")
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q = self.model.encode_image(image)
            q /= q.norm(dim=-1, keepdim=True)
            q = q.cpu().numpy().flatten()

        sims = np.array([
            float(np.dot(q, img["vector"]))
            for img in images
        ])

        mean, std = sims.mean(), sims.std()
        MIN_SIM   = mean + 0.3 * std   # α=0.3 — thesis-validated default
        CONF_LOW  = mean + 0.3 * std
        CONF_HIGH = mean + 3.0 * std

        results = []
        for img, sim in zip(images, sims):
            if sim < MIN_SIM:
                continue

            confidence = float(np.clip(
                (sim - CONF_LOW) / (CONF_HIGH - CONF_LOW + 1e-6),
                0.0, 1.0
            ))
            results.append({
                "filename":   img["filename"],
                "score":      float(sim),
                "confidence": confidence,
                "path":       img["path"],
                "explain":    "Matched via global visual similarity using OpenCLIP ViT-H/14 embeddings",
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
