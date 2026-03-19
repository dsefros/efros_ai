import importlib
import sys
import types


class DummyWorker:
    def __init__(self, kernel, queue):
        self.kernel = kernel
        self.queue = queue
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True
        return "thread"

    def stop(self):
        self.stopped = True


class DummyModelManager:
    default_model = "mock"
    models = {"mock": object()}


class DummyApp(dict):
    pass


def test_bootstrap_wires_kernel_dependencies(monkeypatch):
    sys.modules["uvicorn"] = types.SimpleNamespace(run=lambda *args, **kwargs: None)
    sys.modules["api.server"] = types.SimpleNamespace(create_app=lambda kernel, model_manager, runtime=None: None)
    sys.modules["kernel.register_knowledge"] = types.SimpleNamespace(register_knowledge=lambda kernel: None)
    sys.modules["services.models.model_manager"] = types.SimpleNamespace(
        create_default_manager=lambda: DummyModelManager()
    )
    sys.modules["yaml"] = types.SimpleNamespace(safe_load=lambda *args, **kwargs: {})
    sys.modules["dotenv"] = types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None)

    sys.modules.pop("run", None)
    run = importlib.import_module("run")

    calls = {}
    worker_instances = []

    def build_worker(kernel, queue):
        worker = DummyWorker(kernel, queue)
        worker_instances.append(worker)
        return worker

    def fake_register_knowledge(kernel):
        calls["knowledge_kernel"] = kernel
        kernel.knowledge = object()

    def fake_load_module(kernel, path):
        calls["load_module"] = (kernel, path)
        kernel.register_executor("support_agent", object())
        kernel.pipeline_engine.register_pipeline("demo", [])
        kernel.tools.register(types.SimpleNamespace(name="search"))
        return {"name": "support_module"}

    def fake_create_app(kernel, model_manager, runtime=None):
        calls["create_app"] = (kernel, model_manager, runtime)
        return DummyApp(kernel=kernel, model_manager=model_manager, runtime=runtime)

    monkeypatch.setattr(run, "EventBus", lambda: "events")
    monkeypatch.setattr(run, "JobQueue", lambda: "jobs")
    monkeypatch.setattr(run, "Worker", build_worker)
    monkeypatch.setattr(run, "create_default_manager", lambda: DummyModelManager())
    monkeypatch.setattr(run, "register_knowledge", fake_register_knowledge)
    monkeypatch.setattr(run, "load_module", fake_load_module)
    monkeypatch.setattr(run, "create_app", fake_create_app)

    app = run.bootstrap()
    kernel = app["kernel"]

    assert isinstance(app, DummyApp)
    assert kernel.events == "events"
    assert kernel.jobs == "jobs"
    assert kernel.model_manager.default_model == "mock"
    assert calls["knowledge_kernel"] is kernel
    assert calls["load_module"] == (kernel, "modules/support_module")
    assert calls["create_app"][0] is kernel
    assert calls["create_app"][1] is kernel.model_manager
    assert calls["create_app"][2].kernel is kernel
    assert app["runtime"] is calls["create_app"][2]
    assert worker_instances and worker_instances[0].started is True
    assert "support_agent" in kernel.executors
    assert "demo" in kernel.pipeline_engine.pipelines
    assert "search" in kernel.tools.tools


def test_create_app_lifespan_shuts_down_runtime(monkeypatch):
    sys.modules["uvicorn"] = types.SimpleNamespace(run=lambda *args, **kwargs: None)
    sys.modules["api.server"] = types.SimpleNamespace(create_app=lambda kernel, model_manager, runtime=None: None)
    sys.modules["kernel.register_knowledge"] = types.SimpleNamespace(register_knowledge=lambda kernel: None)
    sys.modules["services.models.model_manager"] = types.SimpleNamespace(
        create_default_manager=lambda: DummyModelManager()
    )
    sys.modules["yaml"] = types.SimpleNamespace(safe_load=lambda *args, **kwargs: {})
    sys.modules["dotenv"] = types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None)

    sys.modules.pop("run", None)
    run = importlib.import_module("run")

    worker_instances = []

    def build_worker(kernel, queue):
        worker = DummyWorker(kernel, queue)
        worker_instances.append(worker)
        return worker

    monkeypatch.setattr(run, "EventBus", lambda: "events")
    monkeypatch.setattr(run, "JobQueue", lambda: "jobs")
    monkeypatch.setattr(run, "Worker", build_worker)
    monkeypatch.setattr(run, "create_default_manager", lambda: DummyModelManager())
    monkeypatch.setattr(run, "register_knowledge", lambda kernel: setattr(kernel, "knowledge", object()))
    monkeypatch.setattr(run, "load_module", lambda kernel, path: None)

    runtime = run.build_runtime()
    runtime.shutdown()

    assert worker_instances and worker_instances[0].started is True
    assert worker_instances[0].stopped is True
    assert runtime.kernel.events == "events"
    assert runtime.kernel.jobs == "jobs"
    assert runtime.kernel.model_manager is runtime.model_manager
