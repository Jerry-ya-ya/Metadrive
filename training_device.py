"""Resolve the training device without silently ignoring a requested GPU."""

import os


def device_status():
    import torch

    requested = os.environ.get("TRAIN_DEVICE", "auto").strip().lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("TRAIN_DEVICE must be auto, cpu, or cuda.")
    available = torch.cuda.is_available()
    return {
        "requested": requested,
        "selected": "cuda" if requested == "cuda" or (requested == "auto" and available) else "cpu",
        "cuda_available": available,
        "gpu_name": torch.cuda.get_device_name(0) if available else None,
        "torch_cuda_version": torch.version.cuda,
    }


def select_training_device():
    status = device_status()
    if status["selected"] == "cuda" and not status["cuda_available"]:
        raise RuntimeError(
            "TRAIN_DEVICE=cuda, but PyTorch cannot access CUDA. "
            "Check the CUDA PyTorch build, Docker GPU allocation, and NVIDIA/WSL drivers."
        )
    return status["selected"]
