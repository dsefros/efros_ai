import logging
import queue
import threading
import uuid


logger = logging.getLogger(__name__)


class Job:
    def __init__(self, type, payload):
        self.id = str(uuid.uuid4())
        self.type = type
        self.payload = payload
        self.retries = 0


class JobQueue:
    def __init__(self):
        self.q = queue.Queue()

    def push(self, job):
        self.q.put(job)
        logger.info(
            "job_enqueued",
            extra={
                "job_id": job.id,
                "job_type": job.type,
                "job_retries": job.retries,
                "queue_size": self.q.qsize(),
            },
        )

    def pop(self, timeout=None):
        job = self.q.get(timeout=timeout)
        logger.info(
            "job_dequeued",
            extra={
                "job_id": job.id,
                "job_type": job.type,
                "job_retries": job.retries,
                "queue_size": self.q.qsize(),
            },
        )
        return job


class Worker:
    MAX_RETRIES = 3

    def __init__(self, kernel, queue):
        self.kernel = kernel
        self.queue = queue
        self.thread = None
        self._stop_event = threading.Event()

    def start(self):
        if self.thread and self.thread.is_alive():
            logger.debug("worker_start_skipped_already_running")
            return self.thread

        self._stop_event.clear()

        def run():
            logger.info("worker_started")
            while not self._stop_event.is_set():
                try:
                    job = self.queue.pop(timeout=0.1)
                except queue.Empty:
                    continue

                logger.info(
                    "job_processing_started",
                    extra={
                        "job_id": job.id,
                        "job_type": job.type,
                        "job_retries": job.retries,
                    },
                )

                try:
                    self.kernel.run_executor(job.type, job.payload)
                except Exception:
                    job.retries += 1
                    attempts = job.retries + 1
                    retries_remaining = max(self.MAX_RETRIES - job.retries, 0)
                    logger.exception(
                        "job_processing_failed",
                        extra={
                            "job_id": job.id,
                            "job_type": job.type,
                            "job_retries": job.retries,
                            "job_attempts": attempts,
                            "max_retries": self.MAX_RETRIES,
                            "retries_remaining": retries_remaining,
                        },
                    )

                    if job.retries < self.MAX_RETRIES and not self._stop_event.is_set():
                        logger.warning(
                            "job_retry_scheduled",
                            extra={
                                "job_id": job.id,
                                "job_type": job.type,
                                "job_retries": job.retries,
                                "job_attempts": attempts,
                                "max_retries": self.MAX_RETRIES,
                            },
                        )
                        self.queue.push(job)
                    else:
                        logger.error(
                            "job_dropped",
                            extra={
                                "job_id": job.id,
                                "job_type": job.type,
                                "job_retries": job.retries,
                                "job_attempts": attempts,
                                "max_retries": self.MAX_RETRIES,
                                "drop_reason": "stop_requested" if self._stop_event.is_set() else "max_retries_exhausted",
                            },
                        )
                else:
                    logger.info(
                        "job_processing_succeeded",
                        extra={
                            "job_id": job.id,
                            "job_type": job.type,
                            "job_retries": job.retries,
                            "job_attempts": job.retries + 1,
                        },
                    )

            logger.info("worker_stopped")

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        return self.thread

    def stop(self, timeout=1.0):
        logger.info("worker_stop_requested")
        self._stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=timeout)
        return self.thread
