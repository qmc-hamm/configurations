"""Command-line interface for the configurations tool."""

import typer
from rich.console import Console
from rich import print as rprint
from typing import List, Optional
import numpy as np
import os
import re
import h5py
from pathlib import Path
from .configuration import Configuration
from .models import ConfigurationMeta

app = typer.Typer(
    name="configurations",
    help="Manage atomic configurations",
    add_completion=False
)
console = Console()

def parse_pressure_from_dir(dir_name: str) -> int:
    """Extract pressure value from directory name (e.g. 'P225' -> 225)."""
    match = re.match(r'P(\d+)', dir_name)
    if not match:
        raise ValueError(f"Invalid pressure directory name: {dir_name}")
    return int(match.group(1))

def parse_temperature_from_dir(dir_name: str) -> int:
    """Extract temperature value from directory name (e.g. 'T1000' -> 1000)."""
    match = re.match(r'T(\d+)', dir_name)
    if not match:
        raise ValueError(f"Invalid temperature directory name: {dir_name}")
    return int(match.group(1))

def process_xyz_file(file_path: Path, pressure: int, temperature: int):
    """Process a single xyz file and create a configuration."""
    try:
        # Create configuration meta with pressure and temperature
        meta = ConfigurationMeta(pressure=pressure, temperature=temperature)
        
        # Create configuration
        config = Configuration(file_path, meta)
        rprint(config)
        
    except Exception as e:
        rprint(f"[red]Error processing file {file_path}: {str(e)}[/red]")

@app.command()
def create(
    data_dir: str = typer.Argument(
        help="Path to the data directory containing P*/T* subdirectories (e.g. './data')"
    ),
    output_hdf5: str = typer.Argument(
        help="Path to the output HDF5 file (e.g. './output.hdf5')"
    )
):
    """Create configurations from xyz files in a directory structure.
    
    The directory structure should be:
    data_dir/
        P{pressure}/
            T{temperature}/
                *.xyz
    
    Example usage:
        configurations create ./data
    """
    try:
        data_path = Path(data_dir)
        if not data_path.exists():
            raise ValueError(f"Directory {data_dir} does not exist")
        
        output_path = Path(output_hdf5)
        rprint(f"[cyan]Creating HDF5 file at {output_path}...[/cyan]")
        with h5py.File(output_path, "w") as hdf5_file:
            for pressure_dir in data_path.glob('P*'):
                if not pressure_dir.is_dir():
                    continue
                try:
                    pressure = parse_pressure_from_dir(pressure_dir.name)

                    # Process each temperature directory
                    for temp_dir in pressure_dir.glob('T*'):
                        if not temp_dir.is_dir():
                            continue
                        try:
                            temperature = parse_temperature_from_dir(temp_dir.name)

                            # Process each xyz file
                            for xyz_file in temp_dir.glob('*.xyz'):
                                try:
                                    #Strip the name to reveal config number the name is of format P{pressure}T{temperature}_config_{config#}.xyz
                                    MDconfig_number = int(xyz_file.stem.split('_')[-1])  # Get the last part after '_'
                                    # Create metadata
                                    meta = ConfigurationMeta(
                                        pressure=pressure,
                                        temperature=temperature,
                                        config_number=-100+MDconfig_number,
                                        state=None,  # Add logic to determine state if needed
                                        MD_type="MD_classical"  # Replace with actual MD type if available
                                    )

                                    # Create configuration
                                    config = Configuration(xyz_file, meta)
                                    rprint(f"[green]Processing {xyz_file}...[/green]")

                                    # Save configuration to HDF5
                                    group_name = f"P{pressure}/T{temperature}"
                                    group = hdf5_file.require_group(group_name)

                                    # Save metadata
                                    group.attrs["pressure"] = meta.pressure
                                    group.attrs["temperature"] = meta.temperature
                                    group.attrs["state"] = meta.state.value if meta.state else "N/A"
                                    group.attrs["MD_type"] = meta.MD_type

                                    # Save XYZ file content
                                    with open(xyz_file, "r") as xyz:
                                        group.create_dataset("xyz_data", data=xyz.read())

                                except Exception as e:
                                    rprint(f"[red]Error processing file {xyz_file}: {str(e)}[/red]")

                        except ValueError as e:
                            rprint(f"[yellow]Skipping directory {temp_dir}: {str(e)}[/yellow]")
                            continue

                except ValueError as e:
                    rprint(f"[yellow]Skipping directory {pressure_dir}: {str(e)}[/yellow]")
                    continue

        rprint(f"[green]HDF5 file created successfully at {output_path}[/green]")

    except Exception as e:
        rprint(f"[red]Error creating HDF5 file: {str(e)}[/red]")
        raise typer.Exit(1)

@app.command()
def list():
    """List all configurations."""
    rprint("Listing all configurations...")

if __name__ == "__main__":
    app() 