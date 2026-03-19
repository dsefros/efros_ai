import queue
import threading
import uuid

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

    def pop(self):
        return self.q.get()

class Worker:

    def __init__(self, kernel, queue):
        self.kernel = kernel
        self.queue = queue

    def start(self):

        def run():
            while True:

                job = self.queue.pop()

                try:
                    self.kernel.run_executor(job.type, job.payload)
                except Exception:
                    job.retries += 1

                    if job.retries < 3:
                        self.queue.push(job)

        t = threading.Thread(target=run)
        t.daemon = True
        t.start()
