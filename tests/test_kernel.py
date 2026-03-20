import sys
import types

import pytest


def _safe_load(stream):
    manifest = {}
    for raw_line in stream:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition(":")
        manifest[key.strip()] = value.strip().strip('"')
    return manifest


sys.modules.setdefault("yaml", types.SimpleNamespace(safe_load=_safe_load))

from kernel.ai_kernel import AIKernel
from kernel.exceptions import ExecutorNotFoundError
from kernel.module_loader import load_module


def test_support_module_registers_executor_tool_and_pipeline():
    kernel = AIKernel()

    manifest = load_module(kernel, "modules/support_module")

    assert manifest["name"] == "support_module"
    assert "support_agent" in kernel.executors
    assert "search" in kernel.tools.tools
    assert "redmine_get_issue" in kernel.tools.tools
    assert "demo" in kernel.pipeline_engine.pipelines


def test_support_agent_executes_registered_tool():
    kernel = AIKernel()
    load_module(kernel, "modules/support_module")

    result = kernel.run_executor("support_agent", {"query": "test"})

    assert result == {"result": "found test"}


def test_run_executor_raises_for_unknown_executor():
    kernel = AIKernel()

    with pytest.raises(ExecutorNotFoundError, match="missing"):
        kernel.run_executor("missing", {"query": "test"})
