"""
Device detection utilities for PyTorch.

Provides a unified way to detect the best available device
for model inference across all providers.
"""

from typing import Literal

DeviceType = Literal["auto", "cuda", "cpu", "mps"]


def get_best_device(preferred: str = "auto") -> str:
    """
    Determine the best available device for model inference.

    Args:
        preferred: Preferred device ("auto", "cuda", "cpu", "mps").
                   If "auto", automatically selects the best available.

    Returns:
        Device string compatible with PyTorch/Transformers.

    Examples:
        >>> device = get_best_device()  # Returns "cuda" if available
        >>> device = get_best_device("cpu")  # Forces CPU
    """
    if preferred != "auto":
        return preferred

    try:
        import torch

        # Check CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            return "cuda"

        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

    except ImportError:
        pass

    return "cpu"


def get_device_info() -> dict:
    """
    Get detailed information about available devices.

    Returns:
        Dictionary with device availability information.

    Example:
        >>> info = get_device_info()
        >>> print(info)
        {'cuda': True, 'cuda_devices': 1, 'mps': False, 'cpu': True, 'recommended': 'cuda'}
    """
    info = {
        "cuda": False,
        "cuda_devices": 0,
        "cuda_name": None,
        "mps": False,
        "cpu": True,
        "recommended": "cpu",
    }

    try:
        import torch

        # CUDA info
        if torch.cuda.is_available():
            info["cuda"] = True
            info["cuda_devices"] = torch.cuda.device_count()
            info["cuda_name"] = torch.cuda.get_device_name(0)
            info["recommended"] = "cuda"

        # MPS info
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["mps"] = True
            if not info["cuda"]:
                info["recommended"] = "mps"

    except ImportError:
        pass

    return info


def print_device_info():
    """Print device information to console."""
    info = get_device_info()

    print("🖥️  Información del Dispositivo:")
    print(f"   CUDA disponible: {info['cuda']}")
    if info["cuda"]:
        print(f"   Dispositivos CUDA: {info['cuda_devices']}")
        print(f"   Nombre CUDA: {info['cuda_name']}")
    print(f"   MPS disponible: {info['mps']}")
    print(f"   Recomendado: {info['recommended']}")
