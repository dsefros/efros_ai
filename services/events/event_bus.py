from collections import defaultdict
import time

class Event:
    def __init__(self, type, payload, source=None):
        self.type = type
        self.payload = payload
        self.source = source
        self.timestamp = time.time()

class EventBus:

    def __init__(self):
        self.subscribers = defaultdict(list)

    def subscribe(self, topic, handler):
        self.subscribers[topic].append(handler)

    def publish(self, event):
        handlers = self.subscribers.get(event.type, [])
        for h in handlers:
            h(event)
