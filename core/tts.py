import time

from piper import PiperVoice

from core.config import Config


class TTSAgent:
    def __init__(self, model_path=None):
        path = model_path or Config.TTS["MODEL_PATH"]
        self.voice = PiperVoice.load(path)

    def generate(self, text: str):
        """
        Generate full audio (non-streaming)
        Returns: {
            "audio": bytes,
            "metrics": {...}
        }
        """
        start_time = time.perf_counter()
        first_chunk_time = None

        full_audio = b""
        chunk_count = 0

        for chunk in self.voice.synthesize(text):
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter()

            audio_bytes = chunk.audio_int16_bytes
            full_audio += audio_bytes
            chunk_count += 1

        end_time = time.perf_counter()

        metrics = self._calculate_metrics(
            start_time, end_time, first_chunk_time, len(full_audio), chunk_count
        )

        return {"audio": full_audio, "metrics": metrics}

    def stream(self, text: str):
        """
        Stream audio chunks (generator)
        Yields:
            {"type": "chunk", "data": bytes}
            {"type": "end", "metrics": {...}}
        """
        start_time = time.perf_counter()
        first_chunk_time = None

        total_audio_bytes = 0
        chunk_count = 0

        for chunk in self.voice.synthesize(text):
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter()

            audio_bytes = chunk.audio_int16_bytes

            total_audio_bytes += len(audio_bytes)
            chunk_count += 1

            yield {"type": "chunk", "audio": audio_bytes}

        end_time = time.perf_counter()

        metrics = self._calculate_metrics(
            start_time, end_time, first_chunk_time, total_audio_bytes, chunk_count
        )

        yield {"type": "end", "metrics": metrics}

    def _calculate_metrics(self, start, end, first_chunk, total_bytes, chunk_count):
        total_time = end - start
        ttfb = (first_chunk - start) if first_chunk else 0

        sample_rate = Config.AUDIO.get("SAMPLE_RATE", 22050)
        bytes_per_sample = 2

        audio_duration = total_bytes / (sample_rate * bytes_per_sample)
        rtf = total_time / audio_duration if audio_duration > 0 else 0

        return {
            "total_time": total_time,
            "ttfb": ttfb,
            "chunks": chunk_count,
            "audio_duration": audio_duration,
            "rtf": rtf,
        }
