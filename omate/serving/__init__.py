"""
Omate serving layer — NVIDIA Triton Inference Server client.

Falls back to direct PyTorch inference when Triton is not running.
"""
from .triton_client import (
    TritonClient,
    MockTritonClient,
    TritonInferResult,
    ModelStatus,
    get_triton_client,
)

__all__ = [
    "TritonClient",
    "MockTritonClient",
    "TritonInferResult",
    "ModelStatus",
    "get_triton_client",
]
