from typing import Dict, Any

class ModuleContext:
    def __init__(self, kernel):
        self.kernel = kernel

    def register_tool(self, tool):
        self.kernel.tools.register(tool)

    def register_pipeline(self, name, pipeline):
        self.kernel.pipeline_engine.register_pipeline(name, pipeline)

    def register_agent(self, name, agent):
        self.kernel.register_executor(name, agent)
