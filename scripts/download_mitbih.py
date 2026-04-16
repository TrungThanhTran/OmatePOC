"""
Download MIT-BIH Arrhythmia Database (PhysioNet).

Downloads annotated ECG recordings covering 8 arrhythmia types.
Each record is a 30-minute, 2-lead ECG at 360 Hz annotated by 2 cardiologists.

Usage:
    make download-mitbih
    # or:
    pip install wfdb
    python scripts/download_mitbih.py

Output: data/mitbih/<record>.{hea,dat,atr}

After download, try:
    python -m omate.demo_signal --data mitbih --record 100
    python -m omate.demo_signal --data mitbih --record 202
    python -m omate.demo_full   --data mitbih --record 202
"""

import os
import sys

try:
    import wfdb
except ImportError:
    print("\nError: wfdb not installed.")
    print("  pip install wfdb\n")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# Records to download — covers all major arrhythmia types in MIT-BIH
RECORDS = [
    ("100", "Normal sinus rhythm",               "Clean reference baseline"),
    ("101", "Normal sinus rhythm + noise",        "Moderate motion artifact"),
    ("108", "ST depression",                      "Ischemic ST changes"),
    ("200", "Premature ventricular contractions", "PVC bursts"),
    ("202", "Atrial fibrillation",                "Sustained AFib — best demo record"),
    ("207", "Ventricular tachycardia",            "V-tach + flutter runs"),
    ("208", "PVCs + bundle branch block",         "Mixed morphology"),
    ("214", "Left bundle branch block (LBBB)",    "Wide QRS, notched R-wave"),
    ("217", "Ventricular bigeminy",               "Alternating normal + PVC"),
    ("231", "LBBB + RBBB",                        "Complete bundle branch blocks"),
    ("232", "Pacemaker rhythm",                   "Pacemaker spikes + paced beats"),
]

OUTPUT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "mitbih")
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_all() -> None:
    if HAS_RICH:
        _download_rich()
    else:
        _download_plain()


def _download_rich() -> None:
    console = Console()

    console.print(Panel.fit(
        "[bold cyan]MIT-BIH Arrhythmia Database — Download[/bold cyan]\n"
        f"Saving [bold]{len(RECORDS)}[/bold] records to [cyan]{OUTPUT_DIR}/[/cyan]\n"
        "[dim]Source: physionet.org/content/mitdb  (PhysioNet, open access)[/dim]",
        border_style="cyan",
    ))

    results: list[tuple[str, str, str, str]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading...", total=len(RECORDS))

        for record_id, label, note in RECORDS:
            progress.update(task, description=f"[cyan]{record_id}[/cyan] {label[:35]}")
            status, size = _download_record(record_id)
            results.append((record_id, label, note, status, size))
            progress.advance(task)

    # Summary table
    table = Table(title="Download Results", box=box.SIMPLE,
                  show_header=True, header_style="bold blue")
    table.add_column("Record", style="cyan", width=8)
    table.add_column("Condition", width=38)
    table.add_column("Note", width=32)
    table.add_column("Status", width=8)
    table.add_column("Size", justify="right", width=8)

    ok = fail = 0
    for record_id, label, note, status, size in results:
        if status == "ok":
            st = "[green]✓[/green]"
            ok += 1
        elif status == "skip":
            st = "[dim]skip[/dim]"
            ok += 1
        else:
            st = "[red]✗[/red]"
            fail += 1
        table.add_row(record_id, label, note, st, size)

    console.print(table)
    console.print(
        f"\n[green]✓[/green] {ok} records ready  "
        + (f"[red]✗ {fail} failed[/red]" if fail else "")
    )
    console.print("\n[bold]Try it:[/bold]")
    console.print("  [cyan]python -m omate.demo_signal --data mitbih --record 100[/cyan]  ← normal")
    console.print("  [cyan]python -m omate.demo_signal --data mitbih --record 202[/cyan]  ← AFib")
    console.print("  [cyan]python -m omate.demo_signal --data mitbih --list[/cyan]        ← all records\n")


def _download_plain() -> None:
    print(f"\nDownloading {len(RECORDS)} MIT-BIH records to {OUTPUT_DIR}/\n")
    for record_id, label, _ in RECORDS:
        print(f"  {record_id}  {label[:40]}", end="  ", flush=True)
        status, size = _download_record(record_id)
        print("✓" if status in ("ok", "skip") else "✗")
    print("\nDone. Run: python -m omate.demo_signal --data mitbih --record 202\n")


def _download_record(record_id: str) -> tuple[str, str]:
    """Download one record. Returns (status, size_str)."""
    files = [f"{record_id}.hea", f"{record_id}.dat", f"{record_id}.atr"]
    hea_path = os.path.join(OUTPUT_DIR, f"{record_id}.hea")

    # Skip if already downloaded
    if all(os.path.exists(os.path.join(OUTPUT_DIR, f)) for f in files):
        size = _dir_size_kb(record_id)
        return "skip", size

    try:
        wfdb.dl_files("mitdb", OUTPUT_DIR, files=files)
        size = _dir_size_kb(record_id)
        return "ok", size
    except Exception as e:
        return "fail", str(e)[:20]


def _dir_size_kb(record_id: str) -> str:
    total = 0
    for ext in (".hea", ".dat", ".atr"):
        p = os.path.join(OUTPUT_DIR, f"{record_id}{ext}")
        if os.path.exists(p):
            total += os.path.getsize(p)
    kb = total / 1024
    return f"{kb:.0f}KB" if kb < 1024 else f"{kb/1024:.1f}MB"


if __name__ == "__main__":
    download_all()
