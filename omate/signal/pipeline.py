"""
End-to-end signal intelligence pipeline.

Combines denoising + anomaly detection into a single callable.
"""

import numpy as np
from dataclasses import dataclass

from .denoising import run_denoising_pipeline, DenoisedSignal
from .anomaly import AnomalyDetector, AnomalyResult


@dataclass
class SignalEvent:
    """Structured event emitted when an anomaly is detected."""
    patient_id: str
    timestamp: str                  # ISO 8601
    anomaly_class: str
    risk_score: float
    confidence: float
    quality_score: float
    rr_mean_ms: float
    rr_std_ms: float
    heart_rate_bpm: float
    requires_escalation: bool       # True if risk_score > threshold
    probabilities: dict


ESCALATION_THRESHOLD = 0.90


def generate_synthetic_ecg(duration_s: float = 10.0, fs: int = 250,
                             heart_rate: int = 72,
                             noise_level: float = 0.05,
                             anomaly_type: str = "normal") -> np.ndarray:
    """
    Generate synthetic ECG signal for testing.

    Args:
        duration_s: signal length in seconds
        fs: sampling frequency
        heart_rate: beats per minute
        noise_level: Gaussian noise amplitude (fraction of QRS amplitude)
        anomaly_type: "normal", "afib", "st_elevation", "noisy"

    Returns:
        np.ndarray of shape (duration_s * fs,)
    """
    n_samples = int(duration_s * fs)
    t = np.linspace(0, duration_s, n_samples)
    signal = np.zeros(n_samples)

    rr_interval_s = 60.0 / heart_rate

    if anomaly_type == "afib":
        # AFib: irregular RR intervals
        beat_times = []
        current = 0.0
        while current < duration_s:
            jitter = np.random.uniform(-0.2, 0.2) * rr_interval_s
            current += rr_interval_s + jitter
            beat_times.append(current)
    else:
        # Regular rhythm
        beat_times = np.arange(0.1, duration_s, rr_interval_s)

    for beat_t in beat_times:
        if beat_t >= duration_s:
            break
        beat_idx = int(beat_t * fs)

        # P wave (small, before QRS)
        p_start = max(0, beat_idx - int(0.08 * fs))
        for i in range(p_start, min(n_samples, beat_idx - int(0.01 * fs))):
            phase = (i - p_start) / (beat_idx - int(0.01 * fs) - p_start + 1)
            signal[i] += 0.15 * np.sin(np.pi * phase)

        # QRS complex
        qrs_width = int(0.05 * fs)
        for i in range(max(0, beat_idx - qrs_width),
                       min(n_samples, beat_idx + qrs_width)):
            phase = (i - beat_idx) / qrs_width
            # Q wave
            if phase < -0.5:
                signal[i] -= 0.1 * np.sin(np.pi * (phase + 1) * 2)
            # R wave (main peak)
            elif -0.5 <= phase <= 0.5:
                signal[i] += np.exp(-8 * phase ** 2)
            # S wave
            else:
                signal[i] -= 0.2 * np.sin(np.pi * phase * 2)

        # T wave (after QRS)
        t_start = beat_idx + int(0.10 * fs)
        t_end = min(n_samples, beat_idx + int(0.35 * fs))
        for i in range(t_start, t_end):
            phase = (i - t_start) / (t_end - t_start + 1)
            t_amplitude = 0.4 if anomaly_type == "st_elevation" else 0.2
            signal[i] += t_amplitude * np.sin(np.pi * phase)

    # Add noise
    signal += np.random.normal(0, noise_level, n_samples)

    # Add baseline wander (low frequency drift)
    wander = 0.1 * np.sin(2 * np.pi * 0.2 * t)
    signal += wander

    return signal


def run_signal_pipeline(raw_ecg: np.ndarray, patient_id: str,
                          timestamp: str, fs: int = 250,
                          escalation_threshold: float = ESCALATION_THRESHOLD,
                          detector: AnomalyDetector | None = None,
                          classifier=None,
                          ) -> tuple[DenoisedSignal, AnomalyResult, SignalEvent | None]:
    """
    Full signal intelligence pipeline.

    Args:
        raw_ecg: raw ECG samples
        patient_id: patient identifier
        timestamp: ISO 8601 event timestamp
        fs: sampling frequency in Hz
        escalation_threshold: risk score above which to flag for escalation
        detector: pre-initialized AnomalyDetector (legacy, creates new if None)
        classifier: ClinicalRuleClassifier or HuggingFaceECGClassifier.
                    Takes priority over detector when provided.

    Returns:
        (denoised, anomaly_result, signal_event)
    """
    # Resolve which classifier to use
    clf = classifier or detector or AnomalyDetector()

    # Step 1 & 2: Denoise
    denoised = run_denoising_pipeline(raw_ecg, fs=fs)

    # Step 3: Anomaly detection
    # prefer predict_from_denoised (uses pre-computed RR intervals)
    if hasattr(clf, "predict_from_denoised"):
        anomaly = clf.predict_from_denoised(denoised)
    else:
        anomaly = clf.predict(denoised.denoised)

    # Build event only if anomaly detected or for logging
    rr = denoised.rr_intervals
    rr_mean = float(np.mean(rr)) if len(rr) > 0 else 0.0
    rr_std = float(np.std(rr)) if len(rr) > 0 else 0.0
    hr = 60_000 / rr_mean if rr_mean > 0 else 0.0

    event = SignalEvent(
        patient_id=patient_id,
        timestamp=timestamp,
        anomaly_class=anomaly.predicted_class,
        risk_score=anomaly.risk_score,
        confidence=anomaly.confidence,
        quality_score=denoised.quality_score,
        rr_mean_ms=round(rr_mean, 1),
        rr_std_ms=round(rr_std, 1),
        heart_rate_bpm=round(hr, 1),
        requires_escalation=anomaly.risk_score >= escalation_threshold,
        probabilities=anomaly.probabilities,
    )

    return denoised, anomaly, event
