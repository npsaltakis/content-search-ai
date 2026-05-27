import os
from pathlib import Path


MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


# =========================================================
# MODEL REGISTRY
# Each entry describes one model and how to obtain it.
# =========================================================
MODEL_REGISTRY = {

    # --------------------------------------------------
    # OpenCLIP ViT-H/14  (image search)
    # --------------------------------------------------
    "openclip_vit_h14": {
        "description": "OpenCLIP xlm-roberta-large ViT-H/14 — image search (text+image)",
        "required_files": [
            "open_clip_pytorch_model.bin",
            "tokenizer/tokenizer.json",
            "tokenizer/sentencepiece.bpe.model",
            "tokenizer/tokenizer_config.json",
            "tokenizer/config.json",
        ],
        "download": {
            "type": "huggingface",
            "repo_id": "laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k",
            "files": {
                "open_clip_pytorch_model.bin": "open_clip_pytorch_model.bin",
            },
            "tokenizer_repo": "xlm-roberta-large",
            "tokenizer_subdir": "tokenizer",
        },
    },

    # --------------------------------------------------
    # multilingual-e5-large  (PDF + audio search)
    # --------------------------------------------------
    "multilingual_e5_large": {
        "description": "multilingual-e5-large — PDF & audio semantic search",
        "required_files": [
            "model.safetensors",
            "tokenizer.json",
            "config.json",
        ],
        "download": {
            "type": "sentence_transformers",
            "model_name": "intfloat/multilingual-e5-large",
        },
    },

    # --------------------------------------------------
    # Audio emotion model  (custom trained checkpoint)
    # --------------------------------------------------
    "audio_emotion": {
        "description": "Audio emotion classifier v5 (Whisper encoder + classifier)",
        "required_files": [
            "best_model_audio_emotion_v5.pt",
        ],
        "download": {
            "type": "gdrive",
            "id": "1aDNpqCu1afl-7Cdn7UjCIp2x2iEU5jme",
            "zip_name": "extra_model_files.zip",
            "dest_dir": "",   # extract directly into models/
        },
    },
}


# =========================================================
# MODEL MANAGER
# =========================================================
class ModelManager:
    """
    Checks that all required model files exist in models/.
    Downloads any missing models automatically.
    """

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.models_dir.mkdir(exist_ok=True)

    # --------------------------------------------------
    def is_ready(self, model_key: str) -> bool:
        """Return True if all required files exist for this model."""
        entry = MODEL_REGISTRY[model_key]
        # audio_emotion lives directly under models/, all others in models/<key>/
        base = self.models_dir if model_key == "audio_emotion" else self.models_dir / model_key
        for rel_file in entry["required_files"]:
            if not (base / rel_file).exists():
                return False
        return True

    # --------------------------------------------------
    def ensure(self, model_key: str):
        """Download model if not already present."""
        if self.is_ready(model_key):
            print(f"[Models] {model_key}: OK (already exists)")
            return

        print(f"[Models] {model_key}: missing — downloading...")
        entry = MODEL_REGISTRY[model_key]
        dl = entry["download"]

        if dl["type"] == "sentence_transformers":
            self._download_sentence_transformers(model_key, dl)
        elif dl["type"] == "huggingface":
            self._download_huggingface(model_key, dl)
        elif dl["type"] == "gdrive":
            self._download_gdrive(dl)
        else:
            raise ValueError(f"Unknown download type: {dl['type']}")

        print(f"[Models] {model_key}: download complete.")

    # --------------------------------------------------
    def ensure_all(self):
        """Ensure every model in the registry is present."""
        for key in MODEL_REGISTRY:
            self.ensure(key)

    # --------------------------------------------------
    # DOWNLOADERS
    # --------------------------------------------------
    def _download_sentence_transformers(self, model_key: str, dl: dict):
        from sentence_transformers import SentenceTransformer
        dest = self.models_dir / model_key
        print(f"  Downloading {dl['model_name']} from HuggingFace...")
        model = SentenceTransformer(dl["model_name"])
        model.save(str(dest))
        print(f"  Saved to {dest}")

    def _download_huggingface(self, model_key: str, dl: dict):
        from huggingface_hub import hf_hub_download

        dest = self.models_dir / model_key
        dest.mkdir(exist_ok=True)

        # Download main model file(s)
        for local_name, remote_name in dl["files"].items():
            out_path = dest / local_name
            if not out_path.exists():
                print(f"  Downloading {remote_name} from {dl['repo_id']}...")
                hf_hub_download(
                    repo_id=dl["repo_id"],
                    filename=remote_name,
                    local_dir=str(dest),
                )

        # Download tokenizer
        if "tokenizer_repo" in dl:
            tok_dest = dest / dl["tokenizer_subdir"]
            tok_dest.mkdir(exist_ok=True)
            tok_files = [
                "tokenizer.json",
                "sentencepiece.bpe.model",
                "tokenizer_config.json",
                "config.json",
            ]
            for fname in tok_files:
                out_path = tok_dest / fname
                if not out_path.exists():
                    try:
                        print(f"  Downloading tokenizer/{fname}...")
                        hf_hub_download(
                            repo_id=dl["tokenizer_repo"],
                            filename=fname,
                            local_dir=str(tok_dest),
                        )
                    except Exception:
                        pass  # optional files

    def _download_gdrive(self, dl: dict):
        import zipfile
        from tqdm import tqdm
        try:
            import gdown
        except ImportError:
            raise RuntimeError("gdown is required: pip install gdown")

        zip_path = self.models_dir / dl["zip_name"]
        url = f"https://drive.google.com/uc?id={dl['id']}"

        print(f"  Downloading {dl['zip_name']} from Google Drive...")
        gdown.download(url, str(zip_path), quiet=False, fuzzy=True)

        if not zipfile.is_zipfile(zip_path):
            raise RuntimeError(f"Invalid ZIP: {zip_path}")

        dest = self.models_dir / dl["dest_dir"] if dl["dest_dir"] else self.models_dir
        print(f"  Extracting to {dest}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            for f in tqdm(z.namelist(), desc="Extracting", unit="file"):
                z.extract(f, dest)

        os.remove(zip_path)
