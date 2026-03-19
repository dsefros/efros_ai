class ServiceAdapter:

    def __init__(self, service):
        self.service = service

    def call(self, method, *args, **kwargs):

        fn = getattr(self.service, method)

        return fn(*args, **kwargs)
