import importlib.util
import os
import yaml
from .module_sdk import ModuleContext

def load_module(kernel, path):
    manifest_path = os.path.join(path, "manifest.yaml")
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    module_file = os.path.join(path, manifest.get("entrypoint", "module.py"))

    spec = importlib.util.spec_from_file_location(manifest["name"], module_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ctx = ModuleContext(kernel)
    mod.register(ctx)

    return manifest
