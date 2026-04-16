import os

import ollama
import requests
from tqdm import tqdm

from core.config import Config


class ModelManager:
    @staticmethod
    def setup_all():
        """Main entry point to ensure all models are ready."""
        Config.ensure_dirs()
        ModelManager.ensure_tts()
        ModelManager.ensure_llm()
        # STT is handled by faster-whisper directly, but we ensure its dir exists

    @staticmethod
    def ensure_tts():
        """Check and download TTS models if missing."""
        model_path = Config.TTS["MODEL_PATH"]
        config_path = Config.TTS["CONFIG_PATH"]
        urls = Config.TTS["DOWNLOAD_URLS"]

        if not os.path.exists(model_path):
            print(f"- TTS model missing. Downloading to {model_path}...")
            ModelManager.download_file(urls["model"], model_path)

        if not os.path.exists(config_path):
            print(f"- TTS config missing. Downloading to {config_path}...")
            ModelManager.download_file(urls["config"], config_path)

    @staticmethod
    def ensure_llm():
        """Check and pull LLM model from Ollama."""
        model_name = Config.LLM["MODEL"]
        try:
            print(f"- Checking LLM model: {model_name}...")
            ollama.show(model_name)
        except ollama.ResponseError:
            print(
                f"- LLM model {model_name} not found. Pulling from Ollama (this may take a while)..."  # noqa
            )
            # We use stream=True to show progress if we were in a terminal,
            # but for now we'll just pull it.
            ollama.pull(model_name)
            print(f"- LLM model {model_name} pulled successfully.")

    @staticmethod
    def download_file(url, destination):
        """Helper to download a file with a progress bar."""
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024

        with open(destination, "wb") as f, tqdm(
            total=total_size,
            unit="iB",
            unit_scale=True,
            desc=os.path.basename(destination),
        ) as bar:
            for data in response.iter_content(block_size):
                f.write(data)
                bar.update(len(data))
