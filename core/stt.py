import time

from faster_whisper import WhisperModel

from core.config import Config


class STTAgent:
    def __init__(self, model_size=None, device=None, compute_type=None):
        self.model = WhisperModel(
            model_size or Config.STT["MODEL_SIZE"],
            device=device or Config.STT["DEVICE"],
            compute_type=compute_type or Config.STT["COMPUTE_TYPE"],
            download_root=Config.STT["CACHE_DIR"],
        )

    def generate(self, audio_path: str):
        """
        Full transcription (non-streaming)
        Returns: {
            "text": str,
            "metrics": {...}
        }
        """
        start_time = time.time()

        segments, info = self.model.transcribe(
            audio_path,
            beam_size=1,
            chunk_length=5,
            vad_filter=True,
            condition_on_previous_text=True,
        )

        first_token_time = None
        last_emit_time = start_time

        chunk_count = 0
        compute_blocks = []
        full_text = ""

        for segment in segments:
            now = time.time()

            if first_token_time is None:
                first_token_time = now - start_time

            delta = now - last_emit_time

            if delta > 0.05:
                compute_blocks.append(delta)

            last_emit_time = now
            chunk_count += 1

            full_text += segment.text + " "

        end_time = time.time()

        metrics = self._calculate_metrics(
            start_time,
            end_time,
            first_token_time,
            chunk_count,
            compute_blocks,
            info.duration,
        )

        return {"text": full_text.strip(), "metrics": metrics}

    def stream(self, audio_path: str):
        """
        Streaming transcription (generator)
        Yields:
            {"type": "segment", "text": str}
            {"type": "end", "metrics": {...}}
        """
        start_time = time.time()

        segments, info = self.model.transcribe(
            audio_path,
            beam_size=1,
            chunk_length=5,
            vad_filter=True,
            condition_on_previous_text=True,
        )

        first_token_time = None
        last_emit_time = start_time

        chunk_count = 0
        compute_blocks = []
        full_text = ""

        for segment in segments:
            now = time.time()

            if first_token_time is None:
                first_token_time = now - start_time

            delta = now - last_emit_time

            if delta > 0.05:
                compute_blocks.append(delta)

            last_emit_time = now
            chunk_count += 1

            full_text += segment.text + " "

            yield {"type": "segment", "text": segment.text.strip(), "delta": delta}

        end_time = time.time()

        metrics = self._calculate_metrics(
            start_time,
            end_time,
            first_token_time,
            chunk_count,
            compute_blocks,
            info.duration,
        )

        yield {"type": "end", "text": full_text.strip(), "metrics": metrics}

    def _calculate_metrics(
        self, start, end, first_token_time, chunk_count, compute_blocks, audio_duration
    ):
        total_time = end - start
        rtf = total_time / audio_duration if audio_duration > 0 else 0

        avg_compute = sum(compute_blocks) / len(compute_blocks) if compute_blocks else 0

        return {
            "ttft": first_token_time or 0,
            "total_time": total_time,
            "chunks": chunk_count,
            "compute_blocks": len(compute_blocks),
            "avg_compute_block": avg_compute,
            "rtf": rtf,
        }
