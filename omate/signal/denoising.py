"""
Signal denoising pipeline.

Implements the three-step pipeline from the article:
  1. Bandpass filter (0.5–40Hz, AHA/AAMI EC38 standard)
  2. Daubechies db4 wavelet denoising (Donoho & Johnstone 1994)
  3. Pan-Tompkins R-peak detection (Pan & Tompkins 1985)
"""

import numpy as np
import pywt
from scipy.signal import butter, sosfiltfilt, find_peaks
from dataclasses import dataclass


@dataclass
class DenoisedSignal:
    raw: np.ndarray
    filtered: np.ndarray          # after bandpass
    denoised: np.ndarray          # after wavelet
    r_peaks: np.ndarray           # sample indices of R-peaks
    rr_intervals: np.ndarray      # ms between beats
    fs: int                        # sampling frequency (Hz)
    quality_score: float           # 0–1, higher = cleaner signal


def bandpass_ecg(signal: np.ndarray, fs: int = 250,
                  lowcut: float = 0.5, highcut: float = 40.0) -> np.ndarray:
    """
    Bandpass filter per AHA/AAMI EC38 clinical specification.

    Args:
        signal: raw ADC samples, shape (N,)
        fs: sampling frequency in Hz (typically 250 or 500)
        lowcut: lower cutoff — removes baseline wander (breathing ~0.15-0.3Hz)
        highcut: upper cutoff — removes EMI, muscle artifacts above 40Hz

    Returns:
        filtered signal, same shape as input
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(4, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, signal)


def wavelet_denoise(signal: np.ndarray, wavelet: str = "db4",
                     level: int = 4) -> np.ndarray:
    """
    Daubechies db4 wavelet denoising.

    db4 shape approximates QRS complex morphology, making it
    especially effective for cardiac signal denoising.

    Universal threshold: sigma * sqrt(2 * log(N))
    where sigma is estimated via MAD of finest-scale coefficients.

    Reference: Donoho & Johnstone (1994), Biometrika 81(3):425-455
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # Estimate noise from finest detail coefficients
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    # Soft thresholding — preserves smooth features better than hard
    denoised_coeffs = [
        pywt.threshold(c, threshold, mode="soft") for c in coeffs
    ]
    return pywt.waverec(denoised_coeffs, wavelet)[: len(signal)]


def detect_r_peaks(signal: np.ndarray, fs: int = 250) -> np.ndarray:
    """
    Simplified Pan-Tompkins R-peak detection.

    Full Pan-Tompkins involves: differentiation → squaring → moving window
    integration → adaptive thresholding. This implements the core idea
    with scipy.find_peaks for the POC.

    Reference: Pan & Tompkins (1985), IEEE Trans Biomed Eng 32(3):230-236
    """
    # Differentiate to amplify QRS complex slope
    diff_signal = np.diff(signal, prepend=signal[0])
    # Square to make all values positive, emphasize large slopes
    squared = diff_signal ** 2
    # Moving window integration (~150ms window)
    window = int(0.15 * fs)
    kernel = np.ones(window) / window
    integrated = np.convolve(squared, kernel, mode="same")
    # Find peaks with minimum distance ~0.3s between beats (max ~200 BPM)
    min_distance = int(0.3 * fs)
    height_threshold = np.mean(integrated) + 0.5 * np.std(integrated)
    peaks, _ = find_peaks(
        integrated, distance=min_distance, height=height_threshold
    )
    return peaks


def compute_rr_intervals(r_peaks: np.ndarray, fs: int = 250) -> np.ndarray:
    """Compute R-R intervals in milliseconds."""
    if len(r_peaks) < 2:
        return np.array([])
    return np.diff(r_peaks) / fs * 1000.0


def compute_signal_quality(signal: np.ndarray, r_peaks: np.ndarray,
                             fs: int = 250) -> float:
    """
    Heuristic signal quality score (0–1).

    Checks:
    - Heart rate is physiologically plausible (30–200 BPM)
    - RR interval coefficient of variation < 50% (excludes noise bursts)
    - Signal amplitude in expected range
    """
    if len(r_peaks) < 2:
        return 0.0

    rr = compute_rr_intervals(r_peaks, fs)
    mean_rr = np.mean(rr)
    cv_rr = np.std(rr) / mean_rr if mean_rr > 0 else 1.0

    # Heart rate from mean RR interval
    hr = 60_000 / mean_rr if mean_rr > 0 else 0
    hr_ok = 30 < hr < 200

    # Coefficient of variation
    cv_ok = cv_rr < 0.50

    # Amplitude — arbitrary units; real system would calibrate per device
    amp_ok = np.std(signal) > 0.01

    score = sum([hr_ok, cv_ok, amp_ok]) / 3.0
    return round(score, 3)


def run_denoising_pipeline(raw: np.ndarray, fs: int = 250) -> DenoisedSignal:
    """
    Full denoising pipeline: bandpass → wavelet → R-peak detection.

    Args:
        raw: raw ECG samples, shape (N,)
        fs: sampling frequency in Hz

    Returns:
        DenoisedSignal dataclass with all intermediate results
    """
    filtered = bandpass_ecg(raw, fs=fs)
    denoised = wavelet_denoise(filtered)
    r_peaks = detect_r_peaks(denoised, fs=fs)
    rr_intervals = compute_rr_intervals(r_peaks, fs=fs)
    quality = compute_signal_quality(denoised, r_peaks, fs=fs)

    return DenoisedSignal(
        raw=raw,
        filtered=filtered,
        denoised=denoised,
        r_peaks=r_peaks,
        rr_intervals=rr_intervals,
        fs=fs,
        quality_score=quality,
    )
