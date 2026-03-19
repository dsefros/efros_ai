from services.jobs.job_queue import JobQueue, Job


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
