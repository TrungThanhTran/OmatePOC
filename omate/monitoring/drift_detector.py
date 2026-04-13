"""
ECG signal distribution drift detector.

Monitors signal statistics daily. When a wearable firmware update or
electrode type change shifts the distribution, this module detects it
within 24 hours before it contaminates model outputs.

Uses scipy KS test (always available). Would use Evidently AI in
production for richer HTML reports and automatic scheduling.

Metrics tracked per window:
  mean, std, rms, skewness, kurtosis, amplitude_range, zero_crossing_rate

Production alert path:
  if drift_report.drift_detected:
      alert_clinical_team(drift_report.recommendation)
      trigger_model_recalibration()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SignalStats:
    """Statistical summary of one ECG signal window."""
    mean: float
    std: float
    rms: float
    skewness: float
    kurtosis: float
    amplitude_range: float
    zero_crossing_rate: float
    timestamp: str = ""


@dataclass
class DriftReport:
    """Result of a drift detection run."""
    drift_detected: bool
    drift_score: float              # 0.0 = stable, 1.0 = all features drifted
    drifted_features: list[str]
    p_values: dict[str, float]
    recommendation: str
    baseline_size: int
    current_size: int


def compute_signal_stats(signal: np.ndarray,
                         timestamp: str = "") -> SignalStats:
    """Compute seven statistical features for drift detection."""
    from scipy.stats import skew, kurtosis

    mean = float(np.mean(signal))
    std = float(np.std(signal))
    rms = float(np.sqrt(np.mean(signal ** 2)))
    sk = float(skew(signal))
    kurt = float(kurtosis(signal))
    amp_range = float(np.max(signal) - np.min(signal))
    zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
    zcr = float(len(zero_crossings) / max(len(signal), 1))

    return SignalStats(
        mean=round(mean, 6),
        std=round(std, 6),
        rms=round(rms, 6),
        skewness=round(sk, 6),
        kurtosis=round(kurt, 6),
        amplitude_range=round(amp_range, 6),
        zero_crossing_rate=round(zcr, 6),
        timestamp=timestamp,
    )


class DriftDetector:
    """
    Kolmogorov–Smirnov drift detector for ECG signal statistics.

    Build a reference baseline, then call check() for each new window.
    When more than 30% of features drift beyond p_threshold, raises alert.

    Usage:
        detector = DriftDetector()
        for signal in baseline_signals:          # first day of data
            detector.add_to_baseline(signal)
        for signal in production_signals:
            report = detector.check(signal)
            if report.drift_detected:
                alert_clinical_team(report.recommendation)
    """

    FEATURES = [
        "mean", "std", "rms", "skewness",
        "kurtosis", "amplitude_range", "zero_crossing_rate",
    ]

    def __init__(self, p_threshold: float = 0.05,
                 min_baseline_size: int = 10,
                 production_window: int = 20):
        self.p_threshold = p_threshold
        self.min_baseline_size = min_baseline_size
        self.production_window = production_window
        self._baseline: list[SignalStats] = []
        self._history: list[SignalStats] = []

    def add_to_baseline(self, signal: np.ndarray,
                        timestamp: str = "") -> None:
        """Add a signal to the reference baseline distribution."""
        self._baseline.append(compute_signal_stats(signal, timestamp))

    def check(self, signal: np.ndarray,
              timestamp: str = "") -> DriftReport:
        """
        Compare current signal against baseline distribution.

        Uses a two-sample KS test per feature. Flags drift when the
        fraction of drifted features exceeds 30%.

        Returns DriftReport — always safe to call even before baseline
        is ready (returns drift_detected=False with explanation).
        """
        if len(self._baseline) < self.min_baseline_size:
            return DriftReport(
                drift_detected=False,
                drift_score=0.0,
                drifted_features=[],
                p_values={},
                recommendation=(
                    f"Baseline building: {len(self._baseline)}"
                    f"/{self.min_baseline_size} samples collected."
                ),
                baseline_size=len(self._baseline),
                current_size=1,
            )

        from scipy.stats import ks_2samp

        current_stats = compute_signal_stats(signal, timestamp)
        self._history.append(current_stats)
        recent = self._history[-self.production_window:]

        p_values: dict[str, float] = {}
        drifted: list[str] = []

        for feat in self.FEATURES:
            baseline_vals = [getattr(s, feat) for s in self._baseline]
            current_vals = [getattr(s, feat) for s in recent]
            if len(current_vals) < 3:
                continue
            _, p_val = ks_2samp(baseline_vals, current_vals)
            p_values[feat] = round(p_val, 4)
            if p_val < self.p_threshold:
                drifted.append(feat)

        drift_score = round(len(drifted) / len(self.FEATURES), 3)
        drift_detected = drift_score > 0.30   # >30% features shifted

        if drift_detected:
            recommendation = (
                f"DRIFT ALERT: {', '.join(drifted)} shifted significantly "
                f"(KS p<{self.p_threshold}). Check wearable firmware or "
                "electrode type. Recalibrate signal model before next run."
            )
        else:
            recommendation = (
                "Signal distribution stable — no action required."
            )

        return DriftReport(
            drift_detected=drift_detected,
            drift_score=drift_score,
            drifted_features=drifted,
            p_values=p_values,
            recommendation=recommendation,
            baseline_size=len(self._baseline),
            current_size=len(recent),
        )

    def reset_baseline(self) -> None:
        """Clear baseline and history (use after model recalibration)."""
        self._baseline.clear()
        self._history.clear()

    @property
    def baseline_size(self) -> int:
        return len(self._baseline)

    @property
    def history_size(self) -> int:
        return len(self._history)
