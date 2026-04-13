"""
Tests for omate/serving — Triton client with mock fallback.

Tests that call .infer() delegate to AnomalyDetector (PyTorch) and are
skipped automatically when torch is not installed.
"""
from __future__ import annotations

import importlib
import numpy as np
import pytest

from omate.serving.triton_client import (
    MockTritonClient,
    TritonClient,
    TritonInferResult,
    ModelStatus,
    get_triton_client,
)

# Skip infer-level tests when PyTorch is not installed
requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch not installed",
)


# ---------------------------------------------------------------------------
# MockTritonClient
# ---------------------------------------------------------------------------

class TestMockTritonClient:

    def _signal(self, n: int = 2500) -> np.ndarray:
        return np.random.randn(n).astype(np.float32)

    @requires_torch
    def test_infer_returns_result(self):
        client = MockTritonClient()
        result = client.infer(self._signal())
        assert isinstance(result, TritonInferResult)
        assert result.logits.shape == (5,)

    @requires_torch
    def test_infer_latency_positive(self):
        client = MockTritonClient()
        result = client.infer(self._signal())
        assert result.latency_ms > 0

    @requires_torch
    def test_infer_backend_is_mock(self):
        client = MockTritonClient()
        result = client.infer(self._signal())
        assert result.backend == "mock_pytorch"

    @requires_torch
    def test_infer_model_name_matches(self):
        client = MockTritonClient(model_name="my_model")
        result = client.infer(self._signal())
        assert result.model_name == "my_model"

    def test_server_always_live(self):
        client = MockTritonClient()
        assert client.server_live() is True
        assert client.server_ready() is True

    def test_model_ready(self):
        client = MockTritonClient()
        assert client.model_ready() is True
        assert client.model_ready("ecg_patchtst") is True
        assert client.model_ready("nonexistent_model") is False

    @requires_torch
    def test_metrics_accumulate(self):
        client = MockTritonClient()
        sig = self._signal()
        for _ in range(5):
            client.infer(sig)
        m = client.get_metrics()
        assert m["models"]["ecg_patchtst"]["total_inferences"] == 5
        assert m["models"]["ecg_patchtst"]["avg_latency_ms"] > 0

    @requires_torch
    def test_metrics_p50_p99(self):
        client = MockTritonClient()
        sig = self._signal()
        for _ in range(20):
            client.infer(sig)
        m = client.get_metrics()
        ms = m["models"]["ecg_patchtst"]
        assert ms["p50_latency_ms"] > 0
        assert ms["p99_latency_ms"] >= ms["p50_latency_ms"]

    def test_get_model_status(self):
        client = MockTritonClient()
        status = client.get_model_status()
        assert isinstance(status, ModelStatus)
        assert status.state == "READY"

    def test_list_models(self):
        client = MockTritonClient()
        models = client.list_models()
        assert len(models) == 1
        assert models[0].model_name == "ecg_patchtst"

    def test_backend_flag_in_metrics(self):
        client = MockTritonClient()
        m = client.get_metrics()
        assert m["backend"] == "mock_pytorch"
        assert m["server_live"] is False   # Triton server not actually running


# ---------------------------------------------------------------------------
# TritonClient (auto-fallback to mock when server is unreachable)
# ---------------------------------------------------------------------------

class TestTritonClientFallback:

    def _signal(self, n: int = 2500) -> np.ndarray:
        return np.random.randn(n).astype(np.float32)

    def test_falls_back_to_mock_on_bad_url(self):
        client = TritonClient(
            url="http://localhost:19999",   # port nothing listens on
            auto_fallback=True,
        )
        assert client._mock is not None

    @requires_torch
    def test_infer_via_fallback(self):
        client = TritonClient(
            url="http://localhost:19999",
            auto_fallback=True,
        )
        result = client.infer(self._signal())
        assert isinstance(result, TritonInferResult)
        assert result.backend == "mock_pytorch"

    def test_server_live_returns_false_when_mock(self):
        client = TritonClient(
            url="http://localhost:19999",
            auto_fallback=True,
        )
        assert client.server_live() is False

    def test_raises_on_no_fallback(self):
        with pytest.raises(ConnectionError):
            TritonClient(
                url="http://localhost:19999",
                auto_fallback=False,
            )

    def test_metrics_from_fallback(self):
        client = TritonClient(
            url="http://localhost:19999",
            auto_fallback=True,
        )
        m = client.get_metrics()
        assert "backend" in m
        assert "models" in m


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_get_triton_client_returns_instance():
    client = get_triton_client()
    assert client is not None

def test_get_triton_client_is_usable():
    client = get_triton_client()
    m = client.get_metrics()
    assert "backend" in m
    assert "models" in m
