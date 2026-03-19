from __future__ import annotations

from importlib import import_module

from kernel.exceptions import ModelDependencyError, ModelInferenceError


class LlamaCppModel:
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: int = 8,
        n_gpu_layers: int = 0,
        temperature: float = 0.2,
        max_tokens: int = 900,
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm = None

    def _load_backend_class(self):
        try:
            module = import_module("llama_cpp")
        except ImportError as exc:
            raise ModelDependencyError(
                "llama-cpp-python is required to use the local llama.cpp backend"
            ) from exc

        llama_cls = getattr(module, "Llama", None)
        if llama_cls is None:
            raise ModelDependencyError("llama_cpp.Llama is unavailable; reinstall llama-cpp-python")
        return llama_cls

    def _ensure_loaded(self):
        if self._llm is None:
            llama_cls = self._load_backend_class()
            self._llm = llama_cls(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
            )

    def generate(self, prompt: str) -> str:
        try:
            self._ensure_loaded()

            result = self._llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "Ты точный технический ассистент. Отвечай по существу, структурированно, без выдумывания фактов и только на русском языке.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            message = result["choices"][0]["message"]["content"]
            return (message or "").strip()
        except ModelDependencyError:
            raise
        except Exception as e:
            raise ModelInferenceError(f"Failed to generate with llama.cpp model {self.model_path}: {e}") from e
