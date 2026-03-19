class PipelineAdapter:

    def __init__(self, service, method):
        self.service = service
        self.method = method

    def run(self, ctx, kernel):

        fn = getattr(self.service, self.method)

        result = fn(**ctx["payload"])

        return {"result": result}
