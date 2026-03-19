from configs.settings import (
    DEFAULT_MODEL,
    MINISTRAL_MODEL_PATH,
    QWEN2_MODEL_PATH,
    MISTRAL7B_MODEL_PATH,
    LLM_N_CTX,
    LLM_N_THREADS,
    LLM_N_GPU_LAYERS,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)
from kernel.exceptions import ModelNotFoundError


def _build_llama_cpp_model(**kwargs):
    from services.models.llama_cpp_model import LlamaCppModel

    return LlamaCppModel(**kwargs)


class ModelManager:
    def __init__(self):
        self.models = {}
        self.default_model = None

    def register(self, name, model):
        self.models[name] = model
        if self.default_model is None:
            self.default_model = name

    def set_default(self, name: str):
        if name not in self.models:
            raise ModelNotFoundError(f"Model {name} not registered")
        self.default_model = name

    def get(self, name=None):
        name = name or self.default_model
        model = self.models.get(name)
        if not model:
            raise ModelNotFoundError(f"Model {name} not registered")
        return model

    def list_models(self):
        return {
            "default_model": self.default_model,
            "models": sorted(list(self.models.keys())),
        }

    def generate(self, prompt: str, model_name: str | None = None):
        model = self.get(model_name)
        return model.generate(prompt)


def create_default_manager():
    manager = ModelManager()

    manager.register(
        "ministral",
        _build_llama_cpp_model(
            model_path=MINISTRAL_MODEL_PATH,
            n_ctx=LLM_N_CTX,
            n_threads=LLM_N_THREADS,
            n_gpu_layers=LLM_N_GPU_LAYERS,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        ),
    )

    manager.register(
        "qwen2",
        _build_llama_cpp_model(
            model_path=QWEN2_MODEL_PATH,
            n_ctx=LLM_N_CTX,
            n_threads=LLM_N_THREADS,
            n_gpu_layers=LLM_N_GPU_LAYERS,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        ),
    )

    manager.register(
        "mistral7b",
        _build_llama_cpp_model(
            model_path=MISTRAL7B_MODEL_PATH,
            n_ctx=LLM_N_CTX,
            n_threads=LLM_N_THREADS,
            n_gpu_layers=LLM_N_GPU_LAYERS,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        ),
    )

    manager.set_default(DEFAULT_MODEL)
    return manager
