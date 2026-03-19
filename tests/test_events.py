import logging

import pytest

from services.events.event_bus import EventBus, Event


def test_event_bus_delivers_event_to_subscribers():
    bus = EventBus()
    received = []

    def handler(event):
        received.append(event)

    bus.subscribe("test", handler)

    event = Event("test", {"msg": "hello"}, source="unit-test")
    bus.publish(event)

    assert received == [event]
    assert received[0].payload == {"msg": "hello"}
    assert received[0].source == "unit-test"
    assert isinstance(received[0].timestamp, float)


def test_event_bus_ignores_topics_without_subscribers(caplog):
    bus = EventBus()

    caplog.set_level(logging.INFO)
    bus.publish(Event("unused", {"msg": "hello"}))

    assert bus.subscribers["unused"] == []
    assert [record.message for record in caplog.records].count("event_published") == 1


def test_event_bus_logs_and_raises_after_handler_failures(caplog):
    bus = EventBus()
    received = []

    def failing_handler(event):
        raise RuntimeError("bad handler")

    def succeeding_handler(event):
        received.append(event.payload)

    bus.subscribe("test", failing_handler)
    bus.subscribe("test", succeeding_handler)

    caplog.set_level(logging.INFO)

    with pytest.raises(RuntimeError, match="One or more handlers failed while publishing event 'test'"):
        bus.publish(Event("test", {"msg": "hello"}, source="unit-test"))

    messages = [record.message for record in caplog.records]

    assert received == [{"msg": "hello"}]
    assert "event_handler_failed" in messages
    assert "event_handler_succeeded" in messages
