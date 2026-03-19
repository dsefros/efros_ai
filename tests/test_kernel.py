from kernel.ai_kernel import AIKernel
from kernel.module_loader import load_module

def test():
    k = AIKernel()
    load_module(k, "modules/support_module")
    res = k.run_executor("support_agent", {"query":"test"})
    print(res)

if __name__ == "__main__":
    test()
