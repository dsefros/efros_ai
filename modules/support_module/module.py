from kernel.tool_manager import Tool
from modules.support_module.pipeline import register_pipeline
from modules.tools.redmine_tool import build_get_issue_handler


def register(ctx):

    def search_tool(query):
        return {"result": f"found {query}"}

    tool = Tool("search", "simple search", search_tool)
    redmine_tool = Tool("redmine_get_issue", "fetch a Redmine issue by id", build_get_issue_handler(ctx.kernel))

    ctx.register_tool(tool)
    ctx.register_tool(redmine_tool)

    class SupportAgent:
        def run(self, payload, kernel):
            tool = kernel.tools.get("search")
            return tool.execute(query=payload["query"])

    ctx.register_agent("support_agent", SupportAgent())

    register_pipeline(ctx.kernel)
