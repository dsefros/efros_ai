import sys
import types

import pytest

sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: False))

from kernel.exceptions import ModelNotFoundError
from services.models.model_manager import ModelManager, create_default_manager


class FakeModel:
    def __init__(self, name):
        self.name = name
        self.prompts = []

    def generate(self, prompt: str):
        self.prompts.append(prompt)
        return f"{self.name}::{prompt}"


def test_model_manager_register_get_list_and_generate():
    manager = ModelManager()
    alpha = FakeModel("alpha")
    beta = FakeModel("beta")

    manager.register("alpha", alpha)
    manager.register("beta", beta)

    assert manager.default_model == "alpha"
    assert manager.get() is alpha
    assert manager.get("beta") is beta
    assert manager.list_models() == {
        "default_model": "alpha",
        "models": ["alpha", "beta"],
    }
    assert manager.generate("hello", model_name="beta") == "beta::hello"
    assert beta.prompts == ["hello"]


def test_model_manager_set_default_and_missing_model_errors():
    manager = ModelManager()
    manager.register("alpha", FakeModel("alpha"))

    manager.set_default("alpha")
    assert manager.default_model == "alpha"

    with pytest.raises(ModelNotFoundError, match="missing"):
        manager.get("missing")

    with pytest.raises(ModelNotFoundError, match="missing"):
        manager.set_default("missing")


def test_create_default_manager_registers_expected_models(monkeypatch):
    created = []

    def stub_build_llama_cpp_model(**kwargs):
        created.append(kwargs)
        return FakeModel(kwargs["model_path"])

    monkeypatch.setattr("services.models.model_manager._build_llama_cpp_model", stub_build_llama_cpp_model)
    monkeypatch.setattr("services.models.model_manager.DEFAULT_MODEL", "qwen2")
    monkeypatch.setattr("services.models.model_manager.MINISTRAL_MODEL_PATH", "/tmp/ministral.gguf")
    monkeypatch.setattr("services.models.model_manager.QWEN2_MODEL_PATH", "/tmp/qwen2.gguf")
    monkeypatch.setattr("services.models.model_manager.MISTRAL7B_MODEL_PATH", "/tmp/mistral7b.gguf")
    monkeypatch.setattr("services.models.model_manager.LLM_N_CTX", 1024)
    monkeypatch.setattr("services.models.model_manager.LLM_N_THREADS", 2)
    monkeypatch.setattr("services.models.model_manager.LLM_N_GPU_LAYERS", 0)
    monkeypatch.setattr("services.models.model_manager.LLM_TEMPERATURE", 0.1)
    monkeypatch.setattr("services.models.model_manager.LLM_MAX_TOKENS", 256)

    manager = create_default_manager()

    assert manager.default_model == "qwen2"
    assert manager.list_models() == {
        "default_model": "qwen2",
        "models": ["ministral", "mistral7b", "qwen2"],
    }
    assert len(created) == 3
    assert created[0]["model_path"] == "/tmp/ministral.gguf"
    assert created[1]["model_path"] == "/tmp/qwen2.gguf"
    assert created[2]["model_path"] == "/tmp/mistral7b.gguf"
