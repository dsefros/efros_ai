import logging
import time

from services.jobs.job_queue import JobQueue, Job, Worker


def test_queue_push_pop_round_trip_preserves_job_data():
    queue = JobQueue()
    job = Job("demo", {"a": 1})

    queue.push(job)
    popped = queue.pop()

    assert popped is job
    assert popped.type == "demo"
    assert popped.payload == {"a": 1}
    assert popped.retries == 0
    assert popped.id


def test_queue_is_fifo():
    queue = JobQueue()
    first = Job("first", {"position": 1})
    second = Job("second", {"position": 2})

    queue.push(first)
    queue.push(second)

    assert queue.pop() is first
    assert queue.pop() is second


class _KernelStub:
    def run_executor(self, name, payload):
        return {"name": name, "payload": payload}


def test_worker_start_is_idempotent():
    from services.jobs.job_queue import Worker

    worker = Worker(_KernelStub(), JobQueue())

    first = worker.start()
    second = worker.start()

    assert first is second
    assert worker.thread is first
    assert worker.thread.daemon is True
    worker.stop()


class _RecordingKernelStub:
    def __init__(self):
        self.calls = []

    def run_executor(self, name, payload):
        self.calls.append((name, payload))
        return {"name": name, "payload": payload}


def test_worker_stop_exits_background_thread():
    kernel = _RecordingKernelStub()
    queue = JobQueue()
    worker = Worker(kernel, queue)

    worker.start()
    queue.push(Job("demo", {"a": 1}))

    deadline = time.time() + 1
    while time.time() < deadline and not kernel.calls:
        time.sleep(0.01)

    worker.stop()

    assert kernel.calls == [("demo", {"a": 1})]
    assert worker.thread is not None
    assert not worker.thread.is_alive()


class _FlakyKernelStub:
    def __init__(self, failures_before_success=0):
        self.failures_before_success = failures_before_success
        self.calls = []

    def run_executor(self, name, payload):
        self.calls.append((name, payload))
        if len(self.calls) <= self.failures_before_success:
            raise RuntimeError("boom")
        return {"name": name, "payload": payload}


def test_worker_logs_retry_then_success(caplog):
    kernel = _FlakyKernelStub(failures_before_success=1)
    queue = JobQueue()
    worker = Worker(kernel, queue)
    job = Job("demo", {"a": 1})

    caplog.set_level(logging.INFO)

    worker.start()
    queue.push(job)

    deadline = time.time() + 1
    while time.time() < deadline and len(kernel.calls) < 2:
        time.sleep(0.01)

    worker.stop()

    messages = [record.message for record in caplog.records]

    assert kernel.calls == [("demo", {"a": 1}), ("demo", {"a": 1})]
    assert job.retries == 1
    assert "job_enqueued" in messages
    assert "job_dequeued" in messages
    assert "job_processing_failed" in messages
    assert "job_retry_scheduled" in messages
    assert "job_processing_succeeded" in messages


class _AlwaysFailKernelStub:
    def __init__(self):
        self.calls = []

    def run_executor(self, name, payload):
        self.calls.append((name, payload))
        raise RuntimeError("boom")


def test_worker_logs_terminal_drop_after_max_retries(caplog):
    kernel = _AlwaysFailKernelStub()
    queue = JobQueue()
    worker = Worker(kernel, queue)
    job = Job("demo", {"a": 1})

    caplog.set_level(logging.INFO)

    worker.start()
    queue.push(job)

    deadline = time.time() + 2
    while time.time() < deadline and len(kernel.calls) < worker.MAX_RETRIES:
        time.sleep(0.01)

    worker.stop()

    messages = [record.message for record in caplog.records]

    assert len(kernel.calls) == worker.MAX_RETRIES
    assert job.retries == worker.MAX_RETRIES
    assert messages.count("job_retry_scheduled") == worker.MAX_RETRIES - 1
    assert "job_dropped" in messages
