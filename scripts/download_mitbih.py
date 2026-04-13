"""
Download MIT-BIH Arrhythmia Database for real ECG testing.

Usage:
    pip install wfdb
    python scripts/download_mitbih.py

Records downloaded to: data/mitbih/
"""

import os
import sys

try:
    import wfdb
except ImportError:
    print("Error: pip install wfdb")
    sys.exit(1)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "mitbih")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sample of interesting records: normal + various arrhythmias
RECORDS = [
    "100",   # Normal sinus rhythm
    "101",   # Normal
    "108",   # ST changes
    "200",   # Premature beats
    "202",   # Atrial fibrillation
    "217",   # Ventricular beats
    "232",   # Pacemaker
]

print(f"Downloading {len(RECORDS)} MIT-BIH records to {OUTPUT_DIR}/")
for record in RECORDS:
    print(f"  {record}...", end=" ", flush=True)
    try:
        wfdb.dl_files(
            "mitdb", OUTPUT_DIR,
            files=[f"{record}.hea", f"{record}.dat", f"{record}.atr"]
        )
        print("✓")
    except Exception as e:
        print(f"✗ ({e})")

print(f"\nDone. Use with:")
print(f"  python -m omate.demo_signal --data mitbih --record 100")
