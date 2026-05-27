import sqlite3
import numpy as np
import torch
import re
import fitz
from sentence_transformers import SentenceTransformer
from core.model import ModelManager


# =========================================================
# E5 PREFIX HELPERS
# E5 models require "query: " / "passage: " prefixes
# for optimal retrieval performance.
# =========================================================
def e5_query(text: str) -> str:
    return f"query: {text.strip()}"

def e5_passage(text: str) -> str:
    return f"passage: {text.strip()}"


def split_into_paragraphs(text, min_len=80):
    raw = re.split(r"\n\s*\n", text)
    return [p.strip() for p in raw if len(p.strip()) >= min_len]


def aggregate_embeddings(vectors, mode="mean"):
    v = np.mean(vectors, axis=0)
    return v / (np.linalg.norm(v) + 1e-8)


class PDFSearcher:
    """
    PDF Searcher — Pure Semantic Retrieval
    Page-level embeddings with adaptive thresholding.

    Model: intfloat/multilingual-e5-large  (text-only semantic model)
    - Replaces M-CLIP (image-text model, suboptimal for text search)
    - 100+ languages including Greek
    - 1024-dim embeddings
    - Requires E5 prefixes: 'query: ' for search, 'passage: ' for indexing
    """

    MODEL_PATH = "./models/multilingual_e5_large"

    def __init__(
        self,
        db_path="content_search_ai.db",
        model_path=None
    ):
        self.db_path = db_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        path = model_path or self.MODEL_PATH
        ModelManager().ensure("multilingual_e5_large")
        print(f"[PDF] Loading multilingual-e5-large from {path} on {self.device}")

        self.model = SentenceTransformer(path, device=self.device)
        self.min_chars = 50

    # ==================================================
    # LOAD PDF PAGE EMBEDDINGS FROM SQLITE
    # ==================================================
    def _load_pdf_embeddings(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT pdf_path, page_number, text_content, embedding FROM pdf_pages"
        )
        rows = cursor.fetchall()
        conn.close()

        pages = []
        for pdf_path, page_number, text_content, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float32)
            pages.append({
                "pdf_path": pdf_path,
                "page":     page_number,
                "text":     text_content,
                "vector":   vec
            })
        return pages

    # ==================================================
    # EXTRACT EMBEDDINGS FOR PDF (INDEXING ONLY)
    # "passage: " prefix — E5 convention for stored docs
    # ==================================================
    def get_pdf_pages_embeddings(self, pdf_path):
        pages = []
        valid_pages = 0

        try:
            with fitz.open(pdf_path) as doc:
                for i, page in enumerate(doc):
                    text = page.get_text("text").strip()
                    if not text or len(text) < self.min_chars:
                        continue

                    clean_text = re.sub(r"[^A-Za-zΑ-Ωα-ω\s]", " ", text)
                    words = [w for w in clean_text.split() if len(w) > 2]
                    if len(words) < 10:
                        continue

                    letters = sum(c.isalpha() for c in text)
                    ratio = letters / (len(text) + 1)
                    if ratio < 0.6:
                        continue

                    valid_pages += 1

                    emb = self.model.encode(
                        e5_passage(text),           # ← "passage: " prefix
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    ).astype(np.float32)

                    pages.append((i + 1, emb, text))

                if valid_pages < 1:
                    print("[PDF] Low text density, skipping PDF.")
                    return []

        except Exception as e:
            print(f"[PDF] Error processing {pdf_path}: {e}")

        return pages

    # ==================================================
    # PDF → PDF SEARCH
    # ==================================================
    def search_similar_pdfs(self, query_pdf_path, top_k=5, top_docs=10):
        query_pages = self.get_pdf_pages_embeddings(query_pdf_path)
        if not query_pages:
            return []

        query_page_vecs = [p[1] for p in query_pages]
        query_doc_vec = aggregate_embeddings(query_page_vecs)

        stored_pages = self._load_pdf_embeddings()
        if not stored_pages:
            return []

        pdf_groups = {}
        for p in stored_pages:
            pdf_groups.setdefault(p["pdf_path"], []).append(p)

        # Pass 1 — document-level similarity
        doc_scores = []
        for pdf_path, pages in pdf_groups.items():
            page_vecs = [pg["vector"] for pg in pages]
            doc_vec = aggregate_embeddings(page_vecs)
            sim = float(np.dot(query_doc_vec, doc_vec))
            doc_scores.append((pdf_path, sim))

        sims = np.array([s for _, s in doc_scores])
        mean, std = sims.mean(), sims.std()
        MIN_SIM = mean + 0.3 * std   # α=0.3 — thesis-validated default

        top_docs_ranked = sorted(
            [(p, s) for p, s in doc_scores if s >= MIN_SIM],
            key=lambda x: x[1],
            reverse=True
        )[:top_docs]

        # Pass 2 — page & paragraph explainability
        results = []
        for pdf_path, doc_sim in top_docs_ranked:
            pages = pdf_groups[pdf_path]

            for page in pages:
                page_sim = float(np.dot(query_doc_vec, page["vector"]))
                if page_sim < MIN_SIM:
                    continue

                paragraphs = split_into_paragraphs(page["text"])
                best_para, best_para_score = None, None

                if paragraphs:
                    para_embs = self.model.encode(
                        [e5_passage(p) for p in paragraphs],   # ← prefix
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                    para_sims = para_embs @ query_doc_vec
                    idx = int(np.argmax(para_sims))
                    best_para = paragraphs[idx]
                    best_para_score = float(para_sims[idx])

                CONF_LOW  = mean + 1.0 * std
                CONF_HIGH = mean + 4.0 * std
                confidence = float(np.clip(
                    (doc_sim - CONF_LOW) / (CONF_HIGH - CONF_LOW + 1e-6),
                    0.0, 1.0
                ))

                results.append({
                    "pdf":               pdf_path,
                    "page":              page["page"],
                    "score":             doc_sim,
                    "confidence":        confidence,
                    "matched_paragraph": best_para,
                    "paragraph_score":   best_para_score
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    # ==================================================
    # TEXT → PDF SEARCH
    # "query: " prefix — E5 convention for search queries
    # ==================================================
    def search_by_text(self, query_text, top_k=5):
        pdf_pages = self._load_pdf_embeddings()
        if not pdf_pages:
            return []

        query_vec = self.model.encode(
            e5_query(query_text),               # ← "query: " prefix
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        sims = np.array([
            float(np.dot(query_vec, page["vector"]))
            for page in pdf_pages
        ])

        mean, std = sims.mean(), sims.std()
        MIN_SIM   = mean + 0.3 * std   # α=0.3 — thesis-validated default
        CONF_LOW  = mean + 0.3 * std
        CONF_HIGH = mean + 4.0 * std

        results = []
        for page, sim in zip(pdf_pages, sims):
            if sim < MIN_SIM:
                continue

            # paragraph explainability
            paragraphs = split_into_paragraphs(page["text"])
            best_para, best_para_score = None, None

            if paragraphs:
                para_embs = self.model.encode(
                    [e5_passage(p) for p in paragraphs],   # ← prefix
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                para_sims = para_embs @ query_vec
                idx = int(np.argmax(para_sims))
                if para_sims[idx] >= MIN_SIM:
                    best_para = paragraphs[idx]
                    best_para_score = float(para_sims[idx])

            confidence = float(np.clip(
                (sim - CONF_LOW) / (CONF_HIGH - CONF_LOW + 1e-6),
                0.0, 1.0
            ))

            results.append({
                "pdf":               page["pdf_path"],
                "page":              page["page"],
                "score":             sim,
                "confidence":        confidence,
                "matched_paragraph": best_para,
                "paragraph_score":   best_para_score
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
