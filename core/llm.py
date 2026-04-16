import time

from langchain_ollama import ChatOllama

from core.config import Config


class ChatAgent:
    def __init__(self, model=None, temperature=None, memory_limit=None):
        self.llm = ChatOllama(
            model=model or Config.LLM["MODEL"],
            temperature=(
                temperature if temperature is not None else Config.LLM["TEMPERATURE"]
            ),
        )
        self.history = []
        self.memory_limit = memory_limit or Config.LLM["MEMORY_LIMIT"]

    def _add_to_history(self, role, content):
        self.history.append({"role": role, "content": content})
        self.history = self.history[-self.memory_limit :]

    def _calculate_metrics(self, start, end, first_token, token_count):
        total_time = end - start
        ttft = (first_token - start) if first_token else total_time

        avg_token_time = total_time / token_count if token_count > 0 else 0

        return {
            "ttft": ttft,
            "total_time": total_time,
            "tokens": token_count,
            "avg_token_time": avg_token_time,
        }

    def generate(self, prompt: str):
        """
        Generate full response (non-streaming)
        Returns: {
            "text": str,
            "metrics": {...}
        }
        """
        self._add_to_history("user", prompt)

        start_time = time.perf_counter()
        response = self.llm.invoke(self.history)
        end_time = time.perf_counter()

        full_response = response.content
        self._add_to_history("assistant", full_response)

        token_count = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            token_count = response.usage_metadata.get("output_tokens", 0)

        metrics = self._calculate_metrics(start_time, end_time, None, token_count)

        return {"text": full_response, "metrics": metrics}

    def stream(self, prompt: str):
        """
        Stream response token by token (generator)
        Yields:
            {"type": "token", "text": str}
            {"type": "end", "text": str, "metrics": {...}}
        """
        self._add_to_history("user", prompt)

        start_time = time.perf_counter()
        first_token_time = None
        full_response = ""
        token_count = 0

        for chunk in self.llm.stream(self.history):
            if first_token_time is None:
                first_token_time = time.perf_counter()

            token = chunk.content
            full_response += token
            token_count += 1

            yield {"type": "token", "text": token}

        end_time = time.perf_counter()
        self._add_to_history("assistant", full_response)

        metrics = self._calculate_metrics(
            start_time, end_time, first_token_time, token_count
        )

        yield {"type": "end", "text": full_response, "metrics": metrics}
