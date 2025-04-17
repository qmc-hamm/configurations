"""Command-line interface for the configurations tool."""

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from typing import List, Optional
import numpy as np

from .core import ConfigurationManager

app = typer.Typer(help="Manage atomic configurations")
console = Console()
manager = ConfigurationManager()

@app.command()
def create(
    name: str = typer.Argument(..., help="Name of the configuration"),
    atoms: str = typer.Option(..., help="Comma-separated list of atomic symbols"),
    positions: str = typer.Option(..., help="Comma-separated list of positions (x1,y1,z1,x2,y2,z2,...)"),
    lattice_vectors: Optional[str] = typer.Option(None, help="Comma-separated list of lattice vectors (a1x,a1y,a1z,a2x,a2y,a2z,a3x,a3y,a3z)")
):
    """Create a new atomic configuration."""
    try:
        # Parse atoms
        atom_list = [a.strip() for a in atoms.split(",")]
        
        # Parse positions
        pos_values = [float(x) for x in positions.split(",")]
        if len(pos_values) % 3 != 0:
            raise ValueError("Number of position values must be divisible by 3")
        pos_array = np.array(pos_values).reshape(-1, 3)
        
        # Parse lattice vectors if provided
        lat_vecs = None
        if lattice_vectors:
            lat_values = [float(x) for x in lattice_vectors.split(",")]
            if len(lat_values) != 9:
                raise ValueError("Lattice vectors must have exactly 9 values")
            lat_vecs = np.array(lat_values).reshape(3, 3)
        
        # Create configuration
        config = manager.create(name, atom_list, pos_array, lat_vecs)
        rprint(f"[green]Created configuration '{name}' successfully![/green]")
        
    except Exception as e:
        rprint(f"[red]Error creating configuration: {str(e)}[/red]")
        raise typer.Exit(1)

@app.command()
def list():
    """List all available configurations."""
    configs = manager.list()
    if not configs:
        rprint("[yellow]No configurations found.[/yellow]")
        return
    
    table = Table(title="Available Configurations")
    table.add_column("Name", style="cyan")
    
    for config in configs:
        table.add_row(config)
    
    console.print(table)

@app.command()
def show(name: str = typer.Argument(..., help="Name of the configuration to show")):
    """Show details of a specific configuration."""
    try:
        config = manager.get(name)
        
        table = Table(title=f"Configuration: {name}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Atoms", ", ".join(config.atoms))
        table.add_row("Positions", str(config.positions))
        if config.lattice_vectors is not None:
            table.add_row("Lattice Vectors", str(config.lattice_vectors))
        
        console.print(table)
        
    except Exception as e:
        rprint(f"[red]Error showing configuration: {str(e)}[/red]")
        raise typer.Exit(1)

@app.command()
def delete(name: str = typer.Argument(..., help="Name of the configuration to delete")):
    """Delete a configuration."""
    try:
        manager.delete(name)
        rprint(f"[green]Deleted configuration '{name}' successfully![/green]")
    except Exception as e:
        rprint(f"[red]Error deleting configuration: {str(e)}[/red]")
        raise typer.Exit(1)

if __name__ == "__main__":
    app() 