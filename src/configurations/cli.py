"""Command-line interface for the configurations tool."""

import typer
from rich.console import Console
from rich import print as rprint
from typing import List, Optional
import numpy as np
import os
import re
from pathlib import Path

from .configuration import Configuration
from .models import ConfigurationMeta

app = typer.Typer(help="Manage atomic configurations")
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
    data_dir: str = typer.Argument(..., help="Path to the data directory containing P*/T* subdirectories")
):
    """Create configurations from xyz files in a directory structure.
    
    The directory structure should be:
    data_dir/
        P{pressure}/
            T{temperature}/
                *.xyz
    """
    try:
        data_path = Path(data_dir)
        if not data_path.exists():
            raise ValueError(f"Directory {data_dir} does not exist")
            
        # Process each pressure directory
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
                            process_xyz_file(xyz_file, pressure, temperature)
                            
                    except ValueError as e:
                        rprint(f"[yellow]Skipping directory {temp_dir}: {str(e)}[/yellow]")
                        continue
                        
            except ValueError as e:
                rprint(f"[yellow]Skipping directory {pressure_dir}: {str(e)}[/yellow]")
                continue
                
    except Exception as e:
        rprint(f"[red]Error creating configurations: {str(e)}[/red]")
        raise typer.Exit(1)

if __name__ == "__main__":
    app() 