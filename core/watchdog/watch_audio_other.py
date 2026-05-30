import time
import json
from pathlib import Path

import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import torch
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer

from core.db.database_helper import DatabaseHelper
from core.emotion_model_v5 import EmotionModelV5


# ======================================================
# 🔧 Convert absolute → relative DB path
# ======================================================
def make_relative(full_path: str) -> str | None:
    full_path = full_path.replace("\\", "/")

    if "/data/" in full_path:
        rel = full_path.split("/data/")[1]
        return f"data/{rel}"

    print("❌ [AUDIO] Could not compute relative path:", full_path)
    return None


# ======================================================
# ⏳ Wait until file is fully written
# ======================================================
def wait_until_file_ready(path: str, timeout: int = 10) -> bool:
    last_size = -1

    for _ in range(timeout):
        try:
            size = Path(path).stat().st_size
            if size == last_size and size > 0:
                return True
            last_size = size
        except FileNotFoundError:
            pass

        time.sleep(1)

    return False


# ======================================================
# 🎧 AUDIO WATCHDOG HANDLER
# ======================================================
class AudioOtherHandler(FileSystemEventHandler):
    def __init__(self):
        # root: content-search-ai
        self.base_dir = Path(__file__).resolve().parents[2]
        print("📌 AUDIO BASE DIR =", self.base_dir)

        # -------------------------
        # DB
        # -------------------------
        db_path = self.base_dir / "content_search_ai.db"
        self.db = DatabaseHelper(str(db_path))
        print("📌 AUDIO DB =", db_path)

        # -------------------------
        # Paths
        # -------------------------
        self.audio_dir = self.base_dir / "data/audio"
        self.watch_dir = str(self.audio_dir)

        # -------------------------
        # State
        # -------------------------
        self.processed_files: set[str] = set()

        # -------------------------
        # Models
        # -------------------------
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("🔹 Loading Faster-Whisper (small, CPU)...")
        self.whisper = WhisperModel(
            "small",
            device="cpu",
            compute_type="int8"
        )

        print("🔹 Loading EmotionModelV5...")
        self.emotion_model = EmotionModelV5(
            ckpt_path=str(self.base_dir / "models/best_model_audio_emotion_v5.pt")
        )

        print("🔹 Loading multilingual-e5-large text encoder...")
        self.text_model = SentenceTransformer(
            str(self.base_dir / "models/multilingual_e5_large"),
            device="cpu"
        )

        print("✅ AUDIO watchdog ready.\n")

    # --------------------------------------------------
    # 🗑️ FILE DELETED
    # --------------------------------------------------
    def on_deleted(self, event):
        if event.is_directory:
            return

        if not event.src_path.lower().endswith((".wav", ".mp3")):
            return

        rel_path = make_relative(event.src_path)
        if not rel_path:
            return

        print(f"🗑️ AUDIO deleted → {rel_path}")
        self.db.delete_audio(rel_path)

    # --------------------------------------------------
    # 🆕 FILE CREATED
    # --------------------------------------------------
    def on_created(self, event):
        if event.is_directory:
            return

        if not event.src_path.lower().endswith((".wav", ".mp3")):
            return

        full_path = event.src_path.replace("\\", "/")

        # Debounce
        if full_path in self.processed_files:
            return

        print(f"🆕 AUDIO detected → {full_path}")

        # Wait until fully written
        if not wait_until_file_ready(full_path):
            print(f"⚠️ AUDIO not ready, skipping → {full_path}")
            return

        self.processed_files.add(full_path)

        rel_path = make_relative(full_path)
        if not rel_path:
            return

        try:
            # 1️⃣ Transcription
            segments, _ = self.whisper.transcribe(
                full_path,
                beam_size=1
            )
            transcript = " ".join(seg.text for seg in segments).strip()

            if not transcript:
                transcript = "[NO_TRANSCRIPT]"

            # 2️⃣ Text embedding  ("passage: " prefix — E5 convention for stored docs)
            emb = self.text_model.encode(
                f"passage: {transcript}",
                normalize_embeddings=True
            ).astype(np.float32)

            # 3️⃣ Emotion detection
            emotion, emotion_probs = self.emotion_model.predict(full_path)

            # 4️⃣ Save to DB
            self.db.insert_audio_embedding(
                rel_path,
                emb.tobytes()
            )

            self.db.insert_audio_emotion(
                rel_path,
                emotion,
                json.dumps(emotion_probs)
            )

            print(f"💾 AUDIO indexed → {Path(full_path).name}")
            print(f"   • emotion : {emotion}")
            print(f"   • path    : {rel_path}")

        except Exception as e:
            print(f"❌ AUDIO processing error → {full_path}")
            print(e)


# ======================================================
# 🚀 START WATCHDOG
# ======================================================
def start_watch():
    handler = AudioOtherHandler()
    observer = Observer()

    print("🎧 Watching AUDIO folder:")
    print(handler.watch_dir)

    observer.schedule(
        handler,
        handler.watch_dir,
        recursive=False
    )
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()


if __name__ == "__main__":
    start_watch()
