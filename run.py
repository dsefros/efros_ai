import logging
import uvicorn

from kernel.ai_kernel import AIKernel
from kernel.module_loader import load_module
from kernel.register_knowledge import register_knowledge

from services.events.event_bus import EventBus
from services.jobs.job_queue import JobQueue, Worker
from services.models.model_manager import create_default_manager

from api.server import create_app
from configs.settings import API_HOST, API_PORT, LOG_LEVEL


def configure_logging():
    level = getattr(logging, str(LOG_LEVEL).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def bootstrap():
    logger = logging.getLogger(__name__)

    kernel = AIKernel()

    kernel.events = EventBus()
    kernel.jobs = JobQueue()

    worker = Worker(kernel, kernel.jobs)
    worker.start()

    model_manager = create_default_manager()
    kernel.model_manager = model_manager

    register_knowledge(kernel)
    load_module(kernel, "modules/support_module")

    app = create_app(kernel, model_manager)

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


if __name__ == "__main__":
    configure_logging()
    app = bootstrap()

    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT
    )
