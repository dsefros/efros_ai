import importlib
import sys
import types

import pytest

sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: False))

from configs.settings import Settings
from kernel.exceptions import ModelDependencyError, ModelNotFoundError
from services.models.llama_cpp_model import LlamaCppModel
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
    settings = Settings(
        default_model="qwen2",
        ministral_model_path="/tmp/ministral.gguf",
        qwen2_model_path="/tmp/qwen2.gguf",
        mistral7b_model_path="/tmp/mistral7b.gguf",
        llm_n_ctx=1024,
        llm_n_threads=2,
        llm_n_gpu_layers=0,
        llm_temperature=0.1,
        llm_max_tokens=256,
    )

    manager = create_default_manager(settings=settings)

    assert manager.default_model == "qwen2"
    assert manager.list_models() == {
        "default_model": "qwen2",
        "models": ["ministral", "mistral7b", "qwen2"],
    }
    assert len(created) == 3
    assert created[0]["model_path"] == "/tmp/ministral.gguf"
    assert created[1]["model_path"] == "/tmp/qwen2.gguf"
    assert created[2]["model_path"] == "/tmp/mistral7b.gguf"


def test_create_default_manager_wraps_backend_initialization_failures(monkeypatch):
    def fail_build_llama_cpp_model(**kwargs):
        raise RuntimeError(f"missing dependency for {kwargs['model_path']}")

    monkeypatch.setattr("services.models.model_manager._build_llama_cpp_model", fail_build_llama_cpp_model)

    settings = Settings(
        default_model="ministral",
        ministral_model_path="/tmp/ministral.gguf",
        qwen2_model_path="/tmp/qwen2.gguf",
        mistral7b_model_path="/tmp/mistral7b.gguf",
    )

    with pytest.raises(ModelDependencyError, match="Failed to initialize model 'ministral'"):
        create_default_manager(settings=settings)


def test_llama_cpp_model_reports_missing_dependency_without_import_time_failure(monkeypatch):
    monkeypatch.setattr("services.models.llama_cpp_model.import_module", lambda name: (_ for _ in ()).throw(ImportError("no llama_cpp")))

    model = LlamaCppModel(model_path="/tmp/demo.gguf")

    with pytest.raises(ModelDependencyError, match="llama-cpp-python is required"):
        model.generate("hello")


def test_llama_cpp_model_reports_missing_llama_class(monkeypatch):
    monkeypatch.setattr("services.models.llama_cpp_model.import_module", lambda name: types.SimpleNamespace())

    model = LlamaCppModel(model_path="/tmp/demo.gguf")

    with pytest.raises(ModelDependencyError, match="llama_cpp.Llama is unavailable"):
        model.generate("hello")


def test_run_module_exits_cleanly_for_model_startup_errors(monkeypatch):
    sys.modules["uvicorn"] = types.SimpleNamespace(run=lambda *args, **kwargs: None)
    sys.modules["api.server"] = types.SimpleNamespace(create_app=lambda kernel, model_manager, runtime=None: None)
    sys.modules["kernel.register_knowledge"] = types.SimpleNamespace(register_knowledge=lambda kernel: None)
    sys.modules["services.models.model_manager"] = types.SimpleNamespace(
        create_default_manager=lambda settings=None: (_ for _ in ()).throw(ModelDependencyError("llama backend missing"))
    )
    sys.modules["yaml"] = types.SimpleNamespace(safe_load=lambda *args, **kwargs: {})

    sys.modules.pop("run", None)
    run = importlib.import_module("run")

    with pytest.raises(SystemExit, match="Startup error: llama backend missing"):
        run.main()
