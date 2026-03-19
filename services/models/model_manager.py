from __future__ import annotations

from configs.settings import Settings, load_settings
from kernel.exceptions import ModelNotFoundError


DEFAULT_MODEL_SPECS = {
    "ministral": "ministral_model_path",
    "qwen2": "qwen2_model_path",
    "mistral7b": "mistral7b_model_path",
}


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


def create_default_manager(settings: Settings | None = None):
    settings = settings or load_settings()
    manager = ModelManager()

    for model_name, model_path_attr in DEFAULT_MODEL_SPECS.items():
        manager.register(
            model_name,
            _build_llama_cpp_model(
                model_path=getattr(settings, model_path_attr),
                n_ctx=settings.llm_n_ctx,
                n_threads=settings.llm_n_threads,
                n_gpu_layers=settings.llm_n_gpu_layers,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            ),
        )

    manager.set_default(settings.default_model)
    return manager
