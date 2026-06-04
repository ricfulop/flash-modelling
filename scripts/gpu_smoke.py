#!/usr/bin/env python
"""Small CUDA smoke test for the DGX Tier A environment."""
from __future__ import annotations

import torch


def main():
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_device_count={torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to torch")
    print(f"cuda_device_0={torch.cuda.get_device_name(0)}")
    x = torch.ones((1024, 1024), device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print(f"matmul_00={float(y[0, 0].detach().cpu())}")


if __name__ == "__main__":
    main()
