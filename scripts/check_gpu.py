"""Execute real Torch and Warp CUDA kernels and save a machine-readable result."""
import json
from pathlib import Path

import torch
import warp as wp


@wp.kernel
def square_kernel(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    i = wp.tid()
    y[i] = x[i] * x[i]


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this process")
    x = torch.arange(1024, dtype=torch.float32, device="cuda")
    y = torch.zeros_like(x)
    wp.init()
    wp.launch(square_kernel, dim=x.numel(), inputs=[wp.from_torch(x), wp.from_torch(y)], device="cuda:0")
    wp.synchronize()
    torch.testing.assert_close(y, x.square())
    a = torch.randn(256, 256, device="cuda", requires_grad=True)
    (a @ a.T).square().mean().backward()
    assert torch.isfinite(a.grad).all()
    result = {
        "gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda, "warp": wp.__version__,
        "warp_kernel_passed": True, "torch_matmul_backward_passed": True,
    }
    Path("reports/gpu-validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
