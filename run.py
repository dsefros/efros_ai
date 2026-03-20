from __future__ import annotations

import logging
from dataclasses import dataclass

import uvicorn

from api.server import create_app
from configs.settings import Settings, SettingsError, load_settings
from kernel.ai_kernel import AIKernel
from kernel.exceptions import ModelError
from kernel.module_loader import load_module
from kernel.register_knowledge import register_knowledge
from services.events.event_bus import EventBus
from services.history import HISTORY_SERVICE_NAME, HistoryPersistenceError, build_history_repository
from services.integrations import REDMINE_SERVICE_NAME, build_redmine_service
from services.jobs.job_queue import JobQueue, Worker
from services.models.model_manager import create_default_manager


@dataclass
class AppRuntime:
    kernel: AIKernel
    model_manager: object
    events: EventBus
    jobs: JobQueue
    worker: Worker
    settings: Settings
    history_repository: object | None = None

    def shutdown(self) -> None:
        self.worker.stop()


def configure_logging(settings: Settings | None = None):
    settings = settings or load_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )



def build_runtime(settings: Settings | None = None) -> AppRuntime:
    settings = settings or load_settings()
    kernel = AIKernel()

    events = EventBus()
    jobs = JobQueue()
    worker = Worker(kernel, jobs)

    kernel.events = events
    kernel.jobs = jobs
    kernel.worker = worker

    model_manager = create_default_manager(settings=settings)
    kernel.model_manager = model_manager

    register_knowledge(kernel)
    history_repository = build_history_repository(settings)
    if history_repository is not None:
        kernel.register_service(HISTORY_SERVICE_NAME, history_repository)
    redmine_service = build_redmine_service(settings)
    if redmine_service is not None:
        kernel.register_service(REDMINE_SERVICE_NAME, redmine_service)
    load_module(kernel, "modules/support_module")
    worker.start()

    return AppRuntime(
        kernel=kernel,
        model_manager=model_manager,
        events=events,
        jobs=jobs,
        worker=worker,
        settings=settings,
        history_repository=history_repository,
    )



def bootstrap(settings: Settings | None = None):
    logger = logging.getLogger(__name__)

    runtime = build_runtime(settings=settings)
    kernel = runtime.kernel
    model_manager = runtime.model_manager

    app = create_app(kernel, model_manager, runtime=runtime)

    logger.info(
        "bootstrap_completed",
        extra={
            "default_model": model_manager.default_model,
            "executors_count": len(kernel.executors),
            "pipelines_count": len(kernel.pipeline_engine.pipelines),
            "tools_count": len(kernel.tools.tools),
        },
    )

    return app


def main() -> None:
    try:
        settings = load_settings()
        configure_logging(settings)
        app = bootstrap(settings=settings)
    except SettingsError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc
    except (ModelError, HistoryPersistenceError) as exc:
        raise SystemExit(f"Startup error: {exc}") from exc

    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
    )


if __name__ == "__main__":
    main()
