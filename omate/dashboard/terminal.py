"""
Rich live terminal dashboard for Omate POC.

Displays real-time clinical AI monitoring across six panels:
  ┌────────────────────────────────────────────────────────┐
  │                    OMATE CLINICAL AI                    │
  ├──────────────────────┬──────────────────────┬──────────┤
  │  Patient Monitor     │  Signal Events       │  Triton  │
  ├──────────────────────┤                      │  Serving │
  │  Agent Decisions     │                      │          │
  ├──────────────────────┼──────────────────────┤  MLflow  │
  │  Signal Drift        │  Latest RAG Report   │  Metrics │
  └──────────────────────┴──────────────────────┴──────────┘

All panels update in-place without clearing the screen.

Run:  python -m omate.demo_dashboard
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# ---------------------------------------------------------------------------
# Mutable dashboard state
# ---------------------------------------------------------------------------

@dataclass
class DashboardState:
    """All data rendered by the dashboard. Mutate this, then call update()."""

    # Patient rows shown in the patient monitor table
    patients: list[dict[str, Any]] = field(default_factory=list)

    # Signal event feed (newest appended last)
    signal_events: list[dict[str, Any]] = field(default_factory=list)

    # Agent decision log
    agent_decisions: list[dict[str, Any]] = field(default_factory=list)

    # Triton serving metrics dict (from TritonClient.get_metrics())
    triton_metrics: dict[str, Any] = field(default_factory=dict)

    # MLflow summary dict (from MLflowTracker.get_summary())
    mlflow_summary: dict[str, Any] = field(default_factory=dict)

    # Drift detection reports
    drift_reports: list[dict[str, Any]] = field(default_factory=list)

    # Latest RAG report text + patient label
    last_report: str = ""
    last_report_patient: str = ""

    # Footer info
    last_updated: str = ""
    status_message: str = "Initializing..."
    llm_backend: str = "mock"


# ---------------------------------------------------------------------------
# Panel builders (each returns a Rich renderable)
# ---------------------------------------------------------------------------

def _header(state: DashboardState) -> Panel:
    backend_color = {
        "mock": "dim", "openai": "green",
        "anthropic": "bright_green", "ollama": "yellow",
    }.get(state.llm_backend, "white")

    t = Text()
    t.append("  OMATE CLINICAL AI ", style="bold cyan")
    t.append("│", style="dim")
    t.append("  Real-time ECG Monitoring  ", style="white")
    t.append("│", style="dim")
    t.append(f"  LLM: {state.llm_backend}  ", style=backend_color)
    t.append("│", style="dim")
    t.append(f"  {state.last_updated}  ", style="dim")
    t.append("│", style="dim")
    t.append(f"  {state.status_message}", style="dim italic")
    return Panel(t, style="cyan", height=3)


def _patient_table(state: DashboardState) -> Panel:
    tbl = Table(
        box=box.SIMPLE, show_header=True, header_style="bold blue",
        expand=True, padding=(0, 1),
    )
    tbl.add_column("Patient",  style="cyan",   width=16)
    tbl.add_column("Status",                   width=11)
    tbl.add_column("Anomaly",                  width=14)
    tbl.add_column("Risk",  justify="right",   width=6)
    tbl.add_column("HR",    justify="right",   width=6)
    tbl.add_column("QT Meds",                  width=14)

    for p in state.patients:
        risk = p.get("risk_score", 0.0)
        if risk >= 0.90:
            risk_str = f"[bold red]{risk:.2f}[/bold red]"
            status   = "[bold red]ESCALATE[/bold red]"
        elif risk >= 0.60:
            risk_str = f"[yellow]{risk:.2f}[/yellow]"
            status   = "[yellow]  WATCH [/yellow]"
        else:
            risk_str = f"[green]{risk:.2f}[/green]"
            status   = "[green]   OK   [/green]"

        tbl.add_row(
            p.get("name", p.get("patient_id", "?")),
            status,
            p.get("anomaly_class", "—"),
            risk_str,
            str(p.get("heart_rate_bpm", "—")),
            p.get("qt_meds", "—"),
        )

    if not state.patients:
        tbl.add_row("[dim]Loading...[/dim]", "", "", "", "", "")

    return Panel(tbl, title="[bold]Patient Monitor[/bold]",
                 border_style="blue")


def _events_feed(state: DashboardState) -> Panel:
    tbl = Table(
        box=box.SIMPLE, show_header=True, header_style="bold",
        expand=True, padding=(0, 1),
    )
    tbl.add_column("Time",       width=9)
    tbl.add_column("Patient",    width=12)
    tbl.add_column("Event",      width=15)
    tbl.add_column("Conf", justify="right", width=6)
    tbl.add_column("Risk", justify="right", width=6)

    anomaly_colors = {
        "ST Elevation": "bold red",
        "AFib":         "yellow",
        "LBBB":         "yellow",
        "Other":        "dim",
        "Normal":       "green",
    }

    for evt in reversed(state.signal_events[-8:]):
        cls   = evt.get("anomaly_class", "—")
        color = anomaly_colors.get(cls, "white")
        tbl.add_row(
            evt.get("time", ""),
            evt.get("patient_id", "")[-12:],
            f"[{color}]{cls}[/{color}]",
            f"{evt.get('confidence', 0):.2f}",
            f"{evt.get('risk_score', 0):.2f}",
        )

    if not state.signal_events:
        tbl.add_row("[dim]Waiting...[/dim]", "", "", "", "")

    return Panel(tbl, title="[bold]Signal Events[/bold]",
                 border_style="yellow")


def _triton_panel(state: DashboardState) -> Panel:
    m = state.triton_metrics
    lines: list[str] = []

    if not m:
        lines.append("[dim]Connecting to Triton...[/dim]")
    else:
        backend = m.get("backend", "—")
        if backend == "triton":
            lines.append(f"[green]● Server LIVE[/green]  {m.get('url', '')}")
        else:
            lines.append("[yellow]● mock_pytorch[/yellow]")
            lines.append("[dim]  Triton not running — using local model[/dim]")
            lines.append("[dim]  Start: docker run nvcr.io/nvidia/tritonserver[/dim]")

        for name, ms in m.get("models", {}).items():
            st = ms.get("state", "?")
            sc = "green" if st == "READY" else "red"
            lines.append("")
            lines.append(f"[bold]{name}[/bold]  v{ms.get('version', '?')}")
            lines.append(f"  State:      [{sc}]{st}[/{sc}]")
            lines.append(f"  Inferences: {ms.get('total_inferences', 0)}")
            avg = ms.get("avg_latency_ms", 0)
            p50 = ms.get("p50_latency_ms", 0)
            p99 = ms.get("p99_latency_ms", 0)
            lines.append(f"  Avg latency: [cyan]{avg:.1f}ms[/cyan]")
            lines.append(f"  p50/p99:     {p50:.1f}ms / {p99:.1f}ms")

    return Panel("\n".join(lines), title="[bold]Triton Serving[/bold]",
                 border_style="magenta")


def _mlflow_panel(state: DashboardState) -> Panel:
    s = state.mlflow_summary
    lines: list[str] = []

    if not s or s.get("total_runs", 0) == 0:
        lines.append("[dim]Waiting for first run...[/dim]")
    else:
        lines.append(f"Total runs:     [cyan]{s['total_runs']}[/cyan]")
        lines.append(f"Avg confidence: [cyan]{s.get('avg_confidence', 0):.3f}[/cyan]")
        lines.append(f"Avg risk score: [cyan]{s.get('avg_risk_score', 0):.3f}[/cyan]")
        lines.append(f"Sig latency:    [cyan]{s.get('avg_latency_signal_ms', 0):.0f}ms[/cyan]")
        lines.append(f"RAG latency:    [cyan]{s.get('avg_latency_rag_ms', 0):.0f}ms[/cyan]")
        lines.append("")
        lines.append(f"[green]Approved:   {s.get('approvals', 0)}[/green]")
        lines.append(f"[red]Escalated:  {s.get('escalations', 0)}[/red]")
        lines.append(f"[yellow]Rejected:   {s.get('rejections', 0)}[/yellow]")
        lines.append("")
        dist = s.get("anomaly_distribution", {})
        if dist:
            lines.append("[bold]Anomaly types:[/bold]")
            for cls, cnt in sorted(dist.items(), key=lambda x: -x[1]):
                lines.append(f"  {cls[:13]:<13}  {cnt}")
        drift = s.get("drift_alert")
        lines.append("")
        if drift:
            lines.append(f"[red]Drift: {drift}[/red]")
        else:
            lines.append("[green]Drift: none detected[/green]")

    return Panel("\n".join(lines), title="[bold]MLflow Metrics[/bold]",
                 border_style="green")


def _agent_panel(state: DashboardState) -> Panel:
    tbl = Table(
        box=box.SIMPLE, show_header=True, header_style="bold",
        expand=True, padding=(0, 1),
    )
    tbl.add_column("Patient",  width=18)
    tbl.add_column("Outcome",  width=17)
    tbl.add_column("Risk", justify="right", width=6)
    tbl.add_column("Steps", justify="right", width=6)

    outcome_colors = {
        "report_approved": "green",
        "escalated":       "bold red",
        "rejected":        "yellow",
        "completed":       "dim",
    }

    for d in reversed(state.agent_decisions[-6:]):
        outcome = d.get("outcome", "—")
        color = outcome_colors.get(outcome, "white")
        tbl.add_row(
            d.get("patient_id", "")[-18:],
            f"[{color}]{outcome}[/{color}]",
            f"{d.get('risk_score', 0):.2f}",
            str(d.get("steps", 0)),
        )

    if not state.agent_decisions:
        tbl.add_row("[dim]No decisions yet[/dim]", "", "", "")

    return Panel(tbl, title="[bold]Agent Decisions (HITL)[/bold]",
                 border_style="cyan")


def _drift_panel(state: DashboardState) -> Panel:
    lines: list[str] = []
    for r in state.drift_reports[-3:]:
        if r.get("drift_detected"):
            feats = r.get("drifted_features", [])
            lines.append(f"[bold red]DRIFT[/bold red]  score={r.get('drift_score', 0):.2f}")
            lines.append(f"  features: {', '.join(feats)}")
        else:
            lines.append(
                f"[green]✓ Stable[/green]  score={r.get('drift_score', 0):.2f}"
                f"  baseline={r.get('baseline_size', 0)}"
            )

    if not lines:
        lines.append("[dim]Building baseline... (need 10+ samples)[/dim]")

    return Panel("\n".join(lines), title="[bold]Signal Drift[/bold]",
                 border_style="yellow")


def _report_panel(state: DashboardState) -> Panel:
    if state.last_report:
        header = f"[bold]Patient:[/bold] {state.last_report_patient}\n\n"
        body = state.last_report[:700]
        if len(state.last_report) > 700:
            body += "\n[dim]… (truncated)[/dim]"
        content = header + body
    else:
        content = "[dim]Waiting for first RAG report...[/dim]"

    return Panel(content,
                 title="[bold]Latest Clinical RAG Report[/bold]",
                 border_style="dim")


# ---------------------------------------------------------------------------
# Layout builder
# ---------------------------------------------------------------------------

def build_layout(state: DashboardState) -> Layout:
    """Assemble all panels into the full-screen layout."""
    layout = Layout()

    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="report", size=12),
    )

    layout["body"].split_row(
        Layout(name="left",  ratio=3),
        Layout(name="right", ratio=2),
    )

    layout["left"].split_column(
        Layout(name="patients", ratio=3),
        Layout(name="events",   ratio=3),
        Layout(name="agents",   ratio=2),
    )

    layout["right"].split_column(
        Layout(name="triton", ratio=4),
        Layout(name="mlflow", ratio=5),
        Layout(name="drift",  ratio=2),
    )

    layout["header"].update(_header(state))
    layout["patients"].update(_patient_table(state))
    layout["events"].update(_events_feed(state))
    layout["agents"].update(_agent_panel(state))
    layout["triton"].update(_triton_panel(state))
    layout["mlflow"].update(_mlflow_panel(state))
    layout["drift"].update(_drift_panel(state))
    layout["report"].update(_report_panel(state))

    return layout


# ---------------------------------------------------------------------------
# Dashboard controller
# ---------------------------------------------------------------------------

class OmateDashboard:
    """
    Rich live terminal dashboard controller.

    Usage:
        state = DashboardState(llm_backend="mock")
        dash = OmateDashboard(state)
        with dash.live():
            # ... mutate state ...
            dash.update()

    The Live context takes over the full terminal screen.
    Exit cleanly with Ctrl-C or when the with-block ends.
    """

    def __init__(self, state: DashboardState,
                 refresh_per_second: int = 4):
        self.state = state
        self._fps = refresh_per_second
        self._console = Console()
        self._live: Live | None = None

    def live(self) -> Live:
        """Return the Live context manager. Use with `with dash.live():`."""
        layout = build_layout(self.state)
        self._live = Live(
            layout,
            refresh_per_second=self._fps,
            screen=True,
            console=self._console,
        )
        return self._live

    def update(self) -> None:
        """Rebuild all panels from current state. Call after each mutation."""
        if self._live is not None:
            self.state.last_updated = datetime.now(timezone.utc).strftime(
                "%H:%M:%S UTC"
            )
            self._live.update(build_layout(self.state))
