"""
ECG Anomaly Detection — PatchTST-inspired Transformer.

A lightweight version of the architecture described in:
  Nie et al. (2023) "A Time Series is Worth 64 Words"
  ICLR 2023 — https://openreview.net/forum?id=Jbdc0vTOcol

The POC uses a smaller model that runs on CPU without a GPU.
For production: use the full PatchTST architecture, export to ONNX,
serve via NVIDIA Triton with target <100ms p99 on A10G GPU.
"""

import warnings
warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")
warnings.filterwarnings("ignore", message=".*is deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from enum import Enum


class AnomalyClass(str, Enum):
    NORMAL = "Normal"
    AFIB = "Atrial Fibrillation"
    ST_ELEVATION = "ST Elevation"
    LBBB = "Left Bundle Branch Block"
    OTHER = "Other Anomaly"


ANOMALY_CLASSES = [e.value for e in AnomalyClass]


@dataclass
class AnomalyResult:
    predicted_class: str
    confidence: float              # probability of predicted class
    probabilities: dict            # all class probabilities
    is_anomaly: bool               # True if not Normal
    risk_score: float              # 0–1, used for escalation routing
    window_samples: int            # number of samples analyzed


class PatchECG(nn.Module):
    """
    Lightweight PatchTST for ECG anomaly detection.

    Treats ECG as a sequence of 200ms patches (50 samples @ 250Hz).
    Uses Transformer encoder to capture long-range dependencies
    between heartbeats — critical for arrhythmia detection.

    POC size: 128d model, 4 heads, 2 layers (~500K params, runs on CPU)
    Production size: 256d, 8 heads, 3 layers, ONNX export to Triton
    """

    def __init__(
        self,
        seq_len: int = 2500,     # 10 seconds at 250Hz
        patch_len: int = 50,      # 200ms per patch (one beat ≈ 4 patches)
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        n_classes: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_len = patch_len
        n_patches = seq_len // patch_len

        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, d_model))
        self.norm_input = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) ECG samples
        Returns:
            logits: (batch, n_classes)
        """
        # Reshape into patches: (B, n_patches, patch_len)
        B, L = x.shape
        x = x.unfold(-1, self.patch_len, self.patch_len)
        # Embed patches
        x = self.patch_embed(x) + self.pos_embed
        x = self.norm_input(x)
        # Transformer encoding
        x = self.transformer(x)
        # Global average pooling over patches
        x = x.mean(dim=1)
        return self.classifier(x)


class AnomalyDetector:
    """
    Wrapper around PatchECG with inference logic.

    For POC: uses randomly-initialized weights (no training data needed).
    For production: load pre-trained weights from model registry.
    """

    def __init__(self, model: PatchECG | None = None, device: str = "cpu"):
        self.device = device
        self.model = model or PatchECG()
        self.model.to(device)
        self.model.eval()
        self.seq_len = 2500  # 10s @ 250Hz

    def warmup(self) -> None:
        """
        JIT-compile the model and stabilize first-inference latency.

        The first PyTorch inference is typically 10–100x slower than
        subsequent ones due to kernel compilation and memory allocation.
        Call this once after construction, before any timed measurements.
        """
        dummy = np.zeros(self.seq_len, dtype=np.float32)
        self.predict(dummy)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "AnomalyDetector":
        """Load pre-trained weights."""
        model = PatchECG()
        model.load_state_dict(torch.load(path, map_location=device))
        return cls(model=model, device=device)

    def predict(self, ecg_window: np.ndarray) -> AnomalyResult:
        """
        Run anomaly detection on a 10-second ECG window.

        Args:
            ecg_window: np.ndarray of shape (2500,) — 10s @ 250Hz
                        Should be denoised before calling this.

        Returns:
            AnomalyResult with class prediction and risk score
        """
        if len(ecg_window) < self.seq_len:
            # Pad if short
            ecg_window = np.pad(
                ecg_window, (0, self.seq_len - len(ecg_window)), mode="edge"
            )
        else:
            ecg_window = ecg_window[: self.seq_len]

        # Normalize per-window
        mu, sigma = ecg_window.mean(), ecg_window.std()
        if sigma > 1e-6:
            ecg_window = (ecg_window - mu) / sigma

        x = torch.tensor(ecg_window, dtype=torch.float32).unsqueeze(0)
        x = x.to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_class = ANOMALY_CLASSES[pred_idx]
        confidence = float(probs[pred_idx])
        is_anomaly = pred_class != AnomalyClass.NORMAL.value

        # Risk score: combines anomaly confidence with class severity
        severity_weights = {
            AnomalyClass.NORMAL.value:       0.0,
            AnomalyClass.AFIB.value:         0.7,
            AnomalyClass.ST_ELEVATION.value: 0.9,
            AnomalyClass.LBBB.value:         0.6,
            AnomalyClass.OTHER.value:        0.5,
        }
        risk_score = float(
            sum(
                probs[i] * severity_weights[ANOMALY_CLASSES[i]]
                for i in range(len(ANOMALY_CLASSES))
            )
        )

        return AnomalyResult(
            predicted_class=pred_class,
            confidence=round(confidence, 4),
            probabilities={cls: round(float(p), 4) for cls, p in
                           zip(ANOMALY_CLASSES, probs)},
            is_anomaly=is_anomaly,
            risk_score=round(risk_score, 4),
            window_samples=len(ecg_window),
        )
