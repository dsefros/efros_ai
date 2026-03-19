import json
import re

JSON_RE = re.compile(r"\{.*\}", re.S)

class LLMRunner:
    def __init__(self, llm, tool_manager):
        self.llm = llm
        self.tools = tool_manager

    def run(self, prompt):
        resp = self.llm(prompt)

        m = JSON_RE.search(resp)
        if not m:
            return resp

        data = json.loads(m.group())
        tool = data.get("tool")
        args = data.get("args", {})

        if tool:
            result = self.tools.call(tool, **args)
            return {"tool_result": result}

        return resp
