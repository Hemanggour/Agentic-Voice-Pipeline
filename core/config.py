import os

class Config:
    # Global Settings
    DEBUG = False
    MODEL_DIR = "./models"
    
    # Audio Settings
    AUDIO = {
        "SAMPLE_RATE": 22050,
        "CHANNELS": 1,
        "RECORD_RATE": 16000,
        "CHUNK_SIZE": 1024,
    }
    
    # STT Settings
    STT = {
        "MODEL_SIZE": "tiny",
        "DEVICE": "cpu",
        "COMPUTE_TYPE": "int8",
        "CACHE_DIR": "./models/stt",
    }
    
    # LLM Settings
    LLM = {
        "MODEL": "gemma3:270m",
        "TEMPERATURE": 0.6,
        "MEMORY_LIMIT": 10,
        "CHUNK_TOKEN_THRESHOLD": 20,
    }

    # TTS Settings
    TTS = {
        "MODEL_NAME": "en_US-lessac-medium",
        "MODEL_PATH": "./models/tts/en_US-lessac-medium.onnx",
        "CONFIG_PATH": "./models/tts/en_US-lessac-medium.onnx.json",
        "DOWNLOAD_URLS": {
            "model": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
            "config": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
        }
    }

    @classmethod
    def ensure_dirs(cls):
        """Ensure model directories exist."""
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        os.makedirs(cls.STT["CACHE_DIR"], exist_ok=True)
