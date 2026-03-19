from services.events.event_bus import EventBus, Event

def test_publish_delivers_event_to_subscriber():
    bus = EventBus()
    received = []

    def handler(event):
        received.append(event)

    bus.subscribe("test", handler)

    event = Event("test", {"msg": "hello"})
    bus.publish(event)

    assert received == [event]
    assert received[0].payload == {"msg": "hello"}
