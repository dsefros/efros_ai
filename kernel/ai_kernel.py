from kernel.tool_manager import ToolManager
from kernel.pipeline_engine import PipelineEngine
from kernel.service_registry import ServiceRegistry
from kernel.exceptions import ExecutorNotFoundError


class AIKernel:
    def __init__(self):
        self.tools = ToolManager()
        self.pipeline_engine = PipelineEngine(self)
        self.executors = {}
        self.services = ServiceRegistry()

        self.events = None
        self.jobs = None
        self.knowledge = None
        self.model_manager = None

    def register_executor(self, name, executor):
        self.executors[name] = executor

    def run_executor(self, name, payload):
        executor = self.executors.get(name)
        if not executor:
            raise ExecutorNotFoundError(f"Executor {name} not found")
        return executor.run(payload, self)

    def register_service(self, name, service):
        self.services.register(name, service)

    def get_service(self, name):
        return self.services.get(name)
