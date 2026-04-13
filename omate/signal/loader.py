"""
MIT-BIH Arrhythmia Database loader.

Loads real annotated ECG recordings from the MIT-BIH dataset
(PhysioNet, Moody & Mark 2001) for use with the Omate pipeline.

MIT-BIH specs:
  - 360 Hz sampling frequency
  - 2 leads: MLII (Lead II) and V1/V5 depending on record
  - 30-minute recordings
  - 11-bit resolution, ~200 ADC units/mV
  - Annotated by 2 cardiologists

Download first:
  make download-mitbih
  # or: pip install wfdb && python scripts/download_mitbih.py

Usage:
  from omate.signal.loader import load_mitbih_record, list_available_records
  signal, fs, meta = load_mitbih_record("202", start_s=60.0, duration_s=10.0)
"""
from __future__ import annotations

import os
import numpy as np
from dataclasses import dataclass, field
from typing import Any

# MIT-BIH native sampling rate
MITBIH_FS = 360

# Curated record catalogue with clinical descriptions
RECORD_CATALOGUE: dict[str, dict[str, str]] = {
    "100": {"label": "Normal sinus rhythm",
            "description": "Clean reference — regular R-R, clear P-QRS-T",
            "best_start_s": "10"},
    "101": {"label": "Normal sinus rhythm",
            "description": "Normal with occasional isolated beats",
            "best_start_s": "10"},
    "108": {"label": "ST depression + noise",
            "description": "Ischemic ST changes + significant baseline noise",
            "best_start_s": "60"},
    "200": {"label": "Premature ventricular contractions",
            "description": "Normal rhythm interrupted by PVC bursts",
            "best_start_s": "30"},
    "202": {"label": "Atrial fibrillation",
            "description": "Sustained AFib — irregular R-R, absent P waves",
            "best_start_s": "20"},
    "207": {"label": "Ventricular tachycardia + flutter",
            "description": "Runs of V-tach — wide QRS, very fast rate",
            "best_start_s": "5"},
    "208": {"label": "PVCs + ST changes",
            "description": "Frequent PVCs with bundle branch morphology",
            "best_start_s": "30"},
    "209": {"label": "Supraventricular tachycardia",
            "description": "Paroxysmal SVT episodes",
            "best_start_s": "40"},
    "214": {"label": "Left bundle branch block (LBBB)",
            "description": "Wide QRS, notched R-wave — classic LBBB morphology",
            "best_start_s": "10"},
    "217": {"label": "Ventricular bigeminy",
            "description": "Alternating normal and premature ventricular beats",
            "best_start_s": "15"},
    "231": {"label": "LBBB + right bundle branch block",
            "description": "Complete bundle branch block patterns",
            "best_start_s": "10"},
    "232": {"label": "Pacemaker rhythm",
            "description": "Pacemaker spikes + paced ventricular beats",
            "best_start_s": "10"},
}

# Map MIT-BIH annotation symbols to human-readable labels
ANNOTATION_MAP: dict[str, str] = {
    "N":  "Normal beat",
    "L":  "Left bundle branch block",
    "R":  "Right bundle branch block",
    "A":  "Atrial premature beat",
    "a":  "Aberrated atrial premature beat",
    "J":  "Nodal premature beat",
    "S":  "Supraventricular premature beat",
    "V":  "Premature ventricular contraction",
    "F":  "Fusion of ventricular and normal beat",
    "!":  "Ventricular flutter wave",
    "e":  "Atrial escape beat",
    "j":  "Nodal escape beat",
    "E":  "Ventricular escape beat",
    "/":  "Paced beat",
    "f":  "Fusion of paced and normal beat",
    "x":  "Non-conducted P-wave (blocked APC)",
    "Q":  "Unclassifiable beat",
    "|":  "Isolated QRS-like artifact",
    "+":  "Rhythm change annotation",
}


@dataclass
class MITBIHRecord:
    """Loaded MIT-BIH record with signal + annotations."""
    record_id: str
    signal: np.ndarray          # ECG samples for requested window
    fs: int                     # always 360 Hz
    lead_name: str              # e.g. "MLII", "V5"
    start_s: float              # start time in the 30-min recording
    duration_s: float           # length of the loaded window
    annotations: list[dict]     # beats within the window
    catalogue_info: dict        # description, label from catalogue
    adc_gain: float             # ADC units per mV
    units: str                  # usually "mV"

    @property
    def label(self) -> str:
        return self.catalogue_info.get("label", "Unknown")

    @property
    def description(self) -> str:
        return self.catalogue_info.get("description", "")

    @property
    def beat_count(self) -> int:
        return len(self.annotations)

    @property
    def dominant_rhythm(self) -> str:
        if not self.annotations:
            return "Unknown"
        counts: dict[str, int] = {}
        for ann in self.annotations:
            sym = ann.get("symbol", "?")
            counts[sym] = counts.get(sym, 0) + 1
        dominant = max(counts, key=counts.get)
        return ANNOTATION_MAP.get(dominant, dominant)

    def to_numpy(self) -> np.ndarray:
        return self.signal.copy()


def _default_data_dir() -> str:
    """Find data/mitbih relative to project root."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "data", "mitbih"))


def list_available_records(data_dir: str | None = None) -> list[str]:
    """Return record IDs that have been downloaded to data_dir."""
    directory = data_dir or _default_data_dir()
    if not os.path.isdir(directory):
        return []
    available = []
    for record_id in RECORD_CATALOGUE:
        hea = os.path.join(directory, f"{record_id}.hea")
        dat = os.path.join(directory, f"{record_id}.dat")
        if os.path.exists(hea) and os.path.exists(dat):
            available.append(record_id)
    return sorted(available)


def load_mitbih_record(
    record_id: str,
    data_dir: str | None = None,
    lead: int = 0,
    start_s: float | None = None,
    duration_s: float = 10.0,
    normalize: bool = False,
) -> MITBIHRecord:
    """
    Load a segment from a MIT-BIH record.

    Args:
        record_id:  Record number as string, e.g. "100", "202"
        data_dir:   Directory containing .hea/.dat/.atr files.
                    Defaults to data/mitbih/ in project root.
        lead:       Lead index (0 = MLII/primary, 1 = V-lead)
        start_s:    Start time within 30-min recording.
                    None → uses the catalogue's recommended start.
        duration_s: Length to load in seconds (default 10s)
        normalize:  If True, z-score normalise the signal

    Returns:
        MITBIHRecord with signal, annotations, and metadata.

    Raises:
        ImportError:   wfdb not installed
        FileNotFoundError:  record not found in data_dir
        ValueError:    invalid record_id or out-of-range start_s
    """
    try:
        import wfdb
    except ImportError:
        raise ImportError(
            "pip install wfdb\n"
            "Then run: make download-mitbih"
        )

    directory = data_dir or _default_data_dir()
    record_path = os.path.join(directory, record_id)

    if not os.path.exists(record_path + ".hea"):
        raise FileNotFoundError(
            f"Record {record_id} not found in {directory}.\n"
            f"Run: make download-mitbih"
        )

    # Determine start sample
    cat_info = RECORD_CATALOGUE.get(record_id, {})
    if start_s is None:
        start_s = float(cat_info.get("best_start_s", "10"))

    start_sample = int(start_s * MITBIH_FS)
    n_samples = int(duration_s * MITBIH_FS)

    # Load signal
    record = wfdb.rdrecord(
        record_path,
        sampfrom=start_sample,
        sampto=start_sample + n_samples,
        channels=[lead],
    )

    signal = record.p_signal[:, 0].astype(np.float32)
    lead_name = record.sig_name[0] if record.sig_name else f"Lead{lead}"
    adc_gain = record.adc_gain[0] if record.adc_gain else 200.0
    units = record.units[0] if record.units else "mV"

    if normalize:
        mu, sigma = signal.mean(), signal.std()
        if sigma > 1e-6:
            signal = ((signal - mu) / sigma).astype(np.float32)

    # Load annotations within the window
    annotations: list[dict[str, Any]] = []
    try:
        ann = wfdb.rdann(
            record_path,
            extension="atr",
            sampfrom=start_sample,
            sampto=start_sample + n_samples,
        )
        for i, sample in enumerate(ann.sample):
            sym = ann.symbol[i] if i < len(ann.symbol) else "?"
            rel_sample = sample - start_sample
            annotations.append({
                "symbol":       sym,
                "label":        ANNOTATION_MAP.get(sym, sym),
                "sample":       int(rel_sample),
                "time_s":       round(rel_sample / MITBIH_FS, 4),
            })
    except Exception:
        pass  # annotations optional

    return MITBIHRecord(
        record_id=record_id,
        signal=signal,
        fs=MITBIH_FS,
        lead_name=lead_name,
        start_s=start_s,
        duration_s=duration_s,
        annotations=annotations,
        catalogue_info=cat_info,
        adc_gain=adc_gain,
        units=units,
    )


def resample_to(signal: np.ndarray, src_fs: int,
                dst_fs: int) -> np.ndarray:
    """
    Resample signal from src_fs to dst_fs using scipy.

    MIT-BIH is 360 Hz; Omate pipeline default is 250 Hz.
    Resampling is optional — the pipeline accepts any fs via the fs= arg.
    """
    if src_fs == dst_fs:
        return signal
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(src_fs, dst_fs)
    up, down = dst_fs // g, src_fs // g
    return resample_poly(signal, up, down).astype(np.float32)
