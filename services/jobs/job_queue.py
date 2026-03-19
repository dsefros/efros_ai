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

    def pop(self, timeout=None):
        return self.q.get(timeout=timeout)


class Worker:
    def __init__(self, kernel, queue):
        self.kernel = kernel
        self.queue = queue
        self.thread = None
        self._stop_event = threading.Event()

    def start(self):
        if self.thread and self.thread.is_alive():
            return self.thread

        self._stop_event.clear()

        def run():
            while not self._stop_event.is_set():
                try:
                    job = self.queue.pop(timeout=0.1)
                except queue.Empty:
                    continue

                try:
                    self.kernel.run_executor(job.type, job.payload)
                except Exception:
                    job.retries += 1

                    if job.retries < 3 and not self._stop_event.is_set():
                        self.queue.push(job)

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        return self.thread

    def stop(self, timeout=1.0):
        self._stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=timeout)
        return self.thread
