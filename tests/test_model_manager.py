import sys
import types

import pytest

from kernel.exceptions import ModelNotFoundError
from services.models.model_manager import ModelManager, create_default_manager


class StubModel:
    def __init__(self, response: str):
        self.response = response
        self.prompts = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return f"{self.response}::{prompt}"


def test_model_manager_register_get_list_and_generate():
    manager = ModelManager()
    alpha = StubModel("alpha")
    beta = StubModel("beta")

    manager.register("alpha", alpha)
    manager.register("beta", beta)

    assert manager.default_model == "alpha"
    assert manager.get() is alpha
    assert manager.get("beta") is beta
    assert manager.list_models() == {
        "default_model": "alpha",
        "models": ["alpha", "beta"],
    }
    assert manager.generate("hello") == "alpha::hello"
    assert alpha.prompts == ["hello"]


def test_model_manager_set_default_and_missing_model_errors():
    manager = ModelManager()
    manager.register("alpha", StubModel("alpha"))

    manager.set_default("alpha")
    assert manager.default_model == "alpha"

    with pytest.raises(ModelNotFoundError):
        manager.get("missing")

    with pytest.raises(ModelNotFoundError):
        manager.set_default("missing")


def test_create_default_manager_registers_expected_models(monkeypatch):
    created = []

    class FakeLlamaCppModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.append(kwargs)

        def generate(self, prompt: str) -> str:
            return prompt

    monkeypatch.setitem(
        sys.modules,
        "services.models.llama_cpp_model",
        types.SimpleNamespace(LlamaCppModel=FakeLlamaCppModel),
    )
    monkeypatch.setitem(
        sys.modules,
        "configs.settings",
        types.SimpleNamespace(
            DEFAULT_MODEL="ministral",
            MINISTRAL_MODEL_PATH="/tmp/ministral.gguf",
            QWEN2_MODEL_PATH="/tmp/qwen2.gguf",
            MISTRAL7B_MODEL_PATH="/tmp/mistral7b.gguf",
            LLM_N_CTX=2048,
            LLM_N_THREADS=4,
            LLM_N_GPU_LAYERS=0,
            LLM_TEMPERATURE=0.1,
            LLM_MAX_TOKENS=256,
        ),
    )

    manager = create_default_manager()

    assert manager.default_model == "ministral"
    assert manager.list_models() == {
        "default_model": "ministral",
        "models": ["ministral", "mistral7b", "qwen2"],
    }
    assert len(created) == 3
    assert created[0]["model_path"]
    assert created[0]["n_ctx"] > 0
