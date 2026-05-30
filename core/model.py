from pathlib import Path


MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


MODEL_REGISTRY = {

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

    "audio_emotion": {
        "description": "Audio emotion classifier v5 (Whisper encoder + classifier)",
        "required_files": [
            "best_model_audio_emotion_v5.pt",
        ],
        "download": {
            "type": "huggingface",
            "repo_id": "nickpsal/audio-emotion-v5",
            "files": {
                "best_model_audio_emotion_v5.pt": "best_model_audio_emotion_v5.pt",
            },
        },
    },
}


class ModelManager:
    """
    Checks that all required model files exist in models/.
    Downloads any missing models automatically.
    """

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.models_dir.mkdir(exist_ok=True)

    def is_ready(self, model_key: str) -> bool:
        entry = MODEL_REGISTRY[model_key]
        base = self.models_dir / model_key
        for rel_file in entry["required_files"]:
            if not (base / rel_file).exists():
                return False
        return True

    def ensure(self, model_key: str):
        if self.is_ready(model_key):
            print(f"[Models] {model_key}: OK (already exists)")
            return

        print(f"[Models] {model_key}: missing — downloading...")
        dl = MODEL_REGISTRY[model_key]["download"]

        if dl["type"] == "sentence_transformers":
            self._download_sentence_transformers(model_key, dl)
        elif dl["type"] == "huggingface":
            self._download_huggingface(model_key, dl)
        else:
            raise ValueError(f"Unknown download type: {dl['type']}")

        print(f"[Models] {model_key}: download complete.")

    def ensure_all(self):
        for key in MODEL_REGISTRY:
            self.ensure(key)

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

        for local_name, remote_name in dl["files"].items():
            if not (dest / local_name).exists():
                print(f"  Downloading {remote_name} from {dl['repo_id']}...")
                hf_hub_download(
                    repo_id=dl["repo_id"],
                    filename=remote_name,
                    local_dir=str(dest),
                )

        if "tokenizer_repo" in dl:
            tok_dest = dest / dl["tokenizer_subdir"]
            tok_dest.mkdir(exist_ok=True)
            for fname in ["tokenizer.json", "sentencepiece.bpe.model", "tokenizer_config.json", "config.json"]:
                if not (tok_dest / fname).exists():
                    try:
                        print(f"  Downloading tokenizer/{fname}...")
                        hf_hub_download(
                            repo_id=dl["tokenizer_repo"],
                            filename=fname,
                            local_dir=str(tok_dest),
                        )
                    except Exception:
                        pass
