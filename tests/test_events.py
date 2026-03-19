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


def test_event_bus_ignores_topics_without_subscribers():
    bus = EventBus()

    bus.publish(Event("unused", {"msg": "hello"}))

    assert bus.subscribers["unused"] == []
