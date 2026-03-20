from pathlib import Path
import textwrap

import pytest

from kernel.ai_kernel import AIKernel
from kernel.module_loader import load_module


def _write_module(tmp_path: Path, manifest: str, module_body: str) -> Path:
    module_dir = tmp_path / "demo_module"
    module_dir.mkdir()
    (module_dir / "manifest.yaml").write_text(textwrap.dedent(manifest).strip() + "\n", encoding="utf-8")
    (module_dir / "module.py").write_text(textwrap.dedent(module_body).strip() + "\n", encoding="utf-8")
    return module_dir


def test_load_module_uses_default_entrypoint_and_registers_executor(tmp_path):
    module_dir = _write_module(
        tmp_path,
        """
        name: demo_module
        version: '1.0'
        """,
        """
        def register(ctx):
            class DemoExecutor:
                def run(self, payload, kernel):
                    return {"echo": payload["value"]}

            ctx.register_agent("demo", DemoExecutor())
        """,
    )
    kernel = AIKernel()

    manifest = load_module(kernel, str(module_dir))

    assert manifest["name"] == "demo_module"
    assert kernel.run_executor("demo", {"value": "ok"}) == {"echo": "ok"}


def test_load_module_raises_clear_error_when_entrypoint_file_is_missing(tmp_path):
    module_dir = tmp_path / "broken_module"
    module_dir.mkdir()
    (module_dir / "manifest.yaml").write_text("name: broken\nentrypoint: missing.py\n", encoding="utf-8")
    kernel = AIKernel()

    with pytest.raises(FileNotFoundError):
        load_module(kernel, str(module_dir))


@pytest.mark.parametrize(
    ("manifest_text", "module_body", "error_type", "message"),
    [
        ("entrypoint: module.py\n", "def register(ctx):\n    pass\n", KeyError, "name"),
        ("name: no_register\n", "VALUE = 1\n", AttributeError, "register"),
    ],
)
def test_load_module_surfaces_invalid_manifest_or_register_contracts(
    tmp_path,
    manifest_text,
    module_body,
    error_type,
    message,
):
    module_dir = _write_module(tmp_path, manifest_text, module_body)
    kernel = AIKernel()

    with pytest.raises(error_type, match=message):
        load_module(kernel, str(module_dir))
