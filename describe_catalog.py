#!/usr/bin/env python
"""Script to read and describe the intake catalog."""

import h5py
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from qmc_repo.repo import Repo

console = Console()

# Connect to the repository
repo = Repo()

# List all of the catalog entries
table = Table(title="Catalog Entries", expand=True, show_lines=True)
table.add_column("Name", style="cyan", justify="right")
table.add_column("Description", style="magenta", justify="left")
table.add_column("Version", style="yellow", justify="right")
for entry in repo.entries:
    source = repo.get_source(entry)
    table.add_row(entry, source.description, source.metadata.get("version", "-"))
console.print(table)

# Use version 1.0 of the hydrogen source
source = repo.hydrogen_v1

# Display source info in a panel
source_info = (
    f"[bold]Description:[/bold] {source.description}\n"
    f"[bold]Container:[/bold] {source.container}\n"
    f"[bold]Version:[/bold] {source.metadata.get('version', 'unknown')}"
)
console.print(Panel(source_info, title="[bold cyan]Source: hydrogen[/bold cyan]", expand=False))
console.print()

# Load the data
df = source.read()

# Filter for pressure == 140 and temperature == 2400
high_pressure = df[(df["pressure"] == 140) & (df["temperature"] == 2400)]

console.print(f"[bold]Entries with pressure == 140 and temperature == 2400:[/bold]")
console.print(f"Found [green]{len(high_pressure)}[/green] entries out of [blue]{len(df)}[/blue] total")
console.print()

# Create a table for HDF5 attributes with selected interesting fields
table = Table(title="HDF5 File Attributes", expand=True, show_lines=True)
table.add_column("Config", style="cyan", justify="right")
table.add_column("P (GPa)", style="magenta", justify="right")
table.add_column("T (K)", style="magenta", justify="right")
table.add_column("State", style="yellow")
table.add_column("rs", style="green", justify="right")
table.add_column("Mol %", style="green", justify="right")
table.add_column("Method", style="blue")
table.add_column("Model", style="blue")
table.add_column("Potential Energy (eV)", style="red", justify="right")
table.add_column("Datasets", style="dim")

for row in high_pressure.itertuples():
    with repo.fs.open(row.uri, 'rb') as f:
        with h5py.File(f, 'r') as h5f:
            attrs = dict(h5f.attrs)
            datasets = list(h5f.keys())

            table.add_row(
                str(attrs.get("config_number", "-")),
                str(attrs.get("pressure", "-")),
                str(attrs.get("temperature", "-")),
                str(attrs.get("state", "-")),
                f"{attrs.get('rs', '-'):.2f}" if attrs.get("rs") else "-",
                f"{attrs.get('molecular_percentage', '-'):.1f}" if attrs.get("molecular_percentage") else "-",
                str(attrs.get("method", "-")),
                str(attrs.get("modelname", "-")),
                f"{attrs.get('potential_energy', '-'):.4f}" if attrs.get("potential_energy") else "-",
                ", ".join(datasets),
            )

console.print(table)