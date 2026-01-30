#!/usr/bin/env python
"""Script to read and describe the intake catalog."""
import os

import h5py
import intake
import s3fs
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Load environment variables from .env file
load_dotenv()

# Initialize s3fs to allow access to hdf5 files in S3
fs = s3fs.S3FileSystem(
    endpoint_url=os.getenv('S3_ENDPOINT_URL'),
    key=os.getenv('AWS_ACCESS_KEY_ID'),
    secret=os.getenv('AWS_SECRET_ACCESS_KEY')
)

console = Console()

# Load the catalog
catalog = intake.open_catalog('s3://phy240060/configurations/configurations.yaml',
                              storage_options={
                                  'anon': False,  # Set to True for public buckets
                                  'endpoint_url':  os.getenv('S3_ENDPOINT_URL')}
                              )

# Use version 1.0 of the hydrogen source
source = catalog.hydrogen_v1

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
    with fs.open(row.uri, 'rb') as f:
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