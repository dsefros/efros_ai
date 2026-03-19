from kernel.ai_kernel import AIKernel
from kernel.module_loader import load_module
from kernel.exceptions import ExecutorNotFoundError

def test_load_module_registers_support_agent_pipeline_and_tool():
    k = AIKernel()
    manifest = load_module(k, "modules/support_module")

    assert manifest["name"] == "support_module"
    assert "support_agent" in k.executors
    assert "demo" in k.pipeline_engine.pipelines
    assert "search" in k.tools.tools

    res = k.run_executor("support_agent", {"query": "test"})
    assert res == {"result": "found test"}


def test_run_executor_raises_for_unknown_executor():
    k = AIKernel()

    try:
        k.run_executor("missing", {"query": "test"})
    except ExecutorNotFoundError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("Expected ExecutorNotFoundError for missing executor")
