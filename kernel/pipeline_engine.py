import time
import logging

from kernel.exceptions import PipelineNotFoundError, PipelineStepError

logger = logging.getLogger(__name__)


class PipelineEngine:
    def __init__(self, kernel):
        self.kernel = kernel
        self.pipelines = {}

    def register_pipeline(self, name, pipeline):
        self.pipelines[name] = pipeline
        logger.info("pipeline_registered", extra={"pipeline_name": name, "steps": len(pipeline)})

    def run_pipeline(self, name, payload):
        if name not in self.pipelines:
            raise PipelineNotFoundError(f"Pipeline {name} not found")

        pipeline = self.pipelines[name]
        context = {"payload": payload}

        logger.info("pipeline_started", extra={"pipeline_name": name})

        for step in pipeline:
            start = time.time()
            updates = step(context, self.kernel) or {}

            if not isinstance(updates, dict):
                raise PipelineStepError(
                    f"Pipeline step {step.__name__} must return dict, got {type(updates).__name__}"
                )

            context.update(updates)
            duration = time.time() - start

            logger.info(
                "pipeline_step_completed",
                extra={
                    "pipeline_name": name,
                    "step_name": step.__name__,
                    "duration_sec": round(duration, 4),
                },
            )

        logger.info("pipeline_completed", extra={"pipeline_name": name})
        return context
