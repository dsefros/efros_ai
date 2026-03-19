import sys
import types

import run


class DummyWorker:
    def __init__(self, kernel, queue):
        self.kernel = kernel
        self.queue = queue
        self.started = False

    def start(self):
        self.started = True


class DummyModelManager:
    def __init__(self):
        self.default_model = "mock"
        self.models = {"mock": object()}


def test_bootstrap_wires_kernel_and_returns_app(monkeypatch):
    calls = {"register_knowledge": 0, "load_module": []}
    worker_instances = []
    app = object()

    def fake_create_default_manager():
        return DummyModelManager()

    def fake_register_knowledge(kernel):
        calls["register_knowledge"] += 1
        kernel.knowledge = "knowledge"

    def fake_load_module(kernel, path):
        calls["load_module"].append(path)
        kernel.executors["support_agent"] = object()
        kernel.pipeline_engine.register_pipeline("demo", {"steps": []})
        kernel.tools.tools["search"] = object()
        return {"name": "support_module"}

    def fake_create_app(kernel, model_manager):
        assert kernel.model_manager is model_manager
        assert kernel.knowledge == "knowledge"
        return app

    def fake_worker_factory(kernel, queue):
        worker = DummyWorker(kernel, queue)
        worker_instances.append(worker)
        return worker

    monkeypatch.setattr(run, "Worker", fake_worker_factory)
    monkeypatch.setitem(
        sys.modules,
        "services.models.model_manager",
        types.SimpleNamespace(create_default_manager=fake_create_default_manager),
    )
    monkeypatch.setitem(
        sys.modules,
        "kernel.module_loader",
        types.SimpleNamespace(load_module=fake_load_module),
    )
    monkeypatch.setitem(
        sys.modules,
        "kernel.register_knowledge",
        types.SimpleNamespace(register_knowledge=fake_register_knowledge),
    )
    monkeypatch.setitem(
        sys.modules,
        "api.server",
        types.SimpleNamespace(create_app=fake_create_app),
    )

    bootstrapped_app = run.bootstrap()

    assert bootstrapped_app is app
    assert calls["register_knowledge"] == 1
    assert calls["load_module"] == ["modules/support_module"]
    assert len(worker_instances) == 1
    assert worker_instances[0].started is True
