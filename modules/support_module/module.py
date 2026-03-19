from kernel.tool_manager import Tool
from modules.support_module.pipeline import register_pipeline

def register(ctx):

    def search_tool(query):
        return {"result": f"found {query}"}

    tool = Tool("search", "simple search", search_tool)

    ctx.register_tool(tool)

    class SupportAgent:
        def run(self, payload, kernel):
            tool = kernel.tools.get("search")
            return tool.execute(query=payload["query"])

    ctx.register_agent("support_agent", SupportAgent())

    register_pipeline(ctx.kernel)
