from collections import defaultdict
import logging
import time


logger = logging.getLogger(__name__)


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
        logger.info(
            "event_handler_subscribed",
            extra={
                "event_type": topic,
                "handler_name": getattr(handler, "__name__", handler.__class__.__name__),
                "handler_count": len(self.subscribers[topic]),
            },
        )

    def publish(self, event):
        handlers = list(self.subscribers.get(event.type, []))
        logger.info(
            "event_published",
            extra={
                "event_type": event.type,
                "event_source": event.source,
                "handler_count": len(handlers),
            },
        )

        errors = []
        for handler in handlers:
            handler_name = getattr(handler, "__name__", handler.__class__.__name__)
            try:
                handler(event)
            except Exception as exc:
                errors.append(exc)
                logger.exception(
                    "event_handler_failed",
                    extra={
                        "event_type": event.type,
                        "event_source": event.source,
                        "handler_name": handler_name,
                    },
                )
            else:
                logger.info(
                    "event_handler_succeeded",
                    extra={
                        "event_type": event.type,
                        "event_source": event.source,
                        "handler_name": handler_name,
                    },
                )

        if errors:
            raise RuntimeError(
                f"One or more handlers failed while publishing event '{event.type}'"
            ) from errors[0]
