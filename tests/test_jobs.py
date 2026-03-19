from services.jobs.job_queue import JobQueue, Job

def test_queue():

    q = JobQueue()

    q.push(Job("demo", {"a":1}))

    job = q.pop()

    print("JOB:", job.id, job.payload)

if __name__ == "__main__":
    test_queue()
