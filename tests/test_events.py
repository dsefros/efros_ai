from services.events.event_bus import EventBus, Event

def test_event():

    bus = EventBus()

    def handler(event):
        print("EVENT RECEIVED:", event.payload)

    bus.subscribe("test", handler)

    bus.publish(Event("test", {"msg": "hello"}))

if __name__ == "__main__":
    test_event()
