class Tool:
    def __init__(self, name, description, handler):
        self.name = name
        self.description = description
        self.handler = handler

    def execute(self, **kwargs):
        return self.handler(**kwargs)

class ToolManager:
    def __init__(self):
        self.tools = {}

    def register(self, tool):
        self.tools[tool.name] = tool

    def get(self, name):
        return self.tools[name]

    def call(self, name, **args):
        return self.tools[name].execute(**args)
