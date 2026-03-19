from services.jobs.job_queue import JobQueue, Job

def test_job_queue_pop_returns_pushed_job():
    q = JobQueue()
    job = Job("demo", {"a": 1})

    q.push(job)
    popped = q.pop()

    assert popped is job
    assert popped.type == "demo"
    assert popped.payload == {"a": 1}
