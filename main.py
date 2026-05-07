"""
ad-image-evaluator
------------------
Search Ads image policy compliance and UX evaluation.
Powered by Ollama Vision.

Usage:
    python main.py --image path/to/ad.jpg
    python main.py --image https://example.com/ad.png
"""

import argparse
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from evaluator import evaluate
from policies import VIOLATIONS, HIGH_RISK

load_dotenv()
console = Console()

SEVERITY_COLORS = {
    "HIGH":   "red",
    "MEDIUM": "yellow",
}

VERDICT_COLORS = {
    "PASS":  "bold green",
    "FAIL":  "bold red",
    "ERROR": "bold yellow",
}

VIOLATION_META = {v["id"]: v for v in VIOLATIONS}


def display_results(result: dict, source_label: str):
    console.print(f"\n[bold]Image:[/bold] {source_label}\n")

    violations = result.get("violations", [])
    detected = [v for v in violations if v.get("detected")]

    table = Table(title="Policy Violations", box=box.ROUNDED, show_lines=True)
    table.add_column("Violation", width=28)
    table.add_column("Severity", width=8)
    table.add_column("Detected", width=9)
    table.add_column("Evidence")

    for v in violations:
        meta = VIOLATION_META.get(v["id"], {})
        severity = meta.get("severity", "—")
        sev_color = SEVERITY_COLORS.get(severity, "white")
        detected_flag = "[red]✗ YES[/red]" if v.get("detected") else "[green]✓ NO[/green]"
        table.add_row(
            meta.get("name", v["id"]),
            f"[{sev_color}]{severity}[/{sev_color}]",
            detected_flag,
            v.get("evidence", ""),
        )

    console.print(table)

    ux_score = result.get("ux_score")
    if ux_score is not None and ux_score >= 7:
        ux_color = "green"
    elif ux_score is not None and ux_score >= 4:
        ux_color = "yellow"
    else:
        ux_color = "red"
    console.print(f"\n[bold]UX Score:[/bold] [{ux_color}]{ux_score}/10[/{ux_color}]")
    console.print(f"[dim]{result.get('ux_notes', '')}[/dim]\n")

    audience = result.get("audience", {})
    aud_table = Table(title="Audience Suitability", box=box.SIMPLE)
    aud_table.add_column("Age Group", width=16)
    aud_table.add_column("Appropriate", width=12)

    for key, label in [("under_13", "Under 13"), ("13_to_17", "13 to 17"), ("18_plus", "18+"), ("all_ages", "All Ages")]:
        val = audience.get(key)
        if val:
            flag = "[green]Yes[/green]"
        elif val is False:
            flag = "[red]No[/red]"
        else:
            flag = "[dim]—[/dim]"
        aud_table.add_row(label, flag)

    console.print(aud_table)
    console.print(f"[dim]Recommended targeting: {audience.get('recommended_targeting', '—')}[/dim]\n")

    verdict = result.get("overall_verdict", "ERROR")
    verdict_color = VERDICT_COLORS.get(verdict, "white")
    high_risk_hits = [v for v in detected if v["id"] in HIGH_RISK]
    suffix = f"  —  {len(high_risk_hits)} high-severity violation(s) detected" if high_risk_hits else ""

    console.print(Panel(
        f"[{verdict_color}]{verdict}[/{verdict_color}]{suffix}",
        title="Overall Verdict",
        expand=False,
    ))
    console.print()


def main():
    parser = argparse.ArgumentParser(description="Search Ads image policy evaluator.")
    parser.add_argument("--image", "-i", required=True, help="Path or URL of the ad image to evaluate")
    args = parser.parse_args()

    console.print("\n[bold blue]Ad Image Evaluator[/bold blue]  [dim]powered by Ollama Vision[/dim]\n")

    try:
        with console.status("[bold green]Evaluating image…[/bold green] this may take a minute on CPU"):
            result = evaluate(args.image)
    except (ValueError, OSError) as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)
    display_results(result, args.image)


if __name__ == "__main__":
    main()
