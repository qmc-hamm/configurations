"""Command-line interface for the configurations tool."""

import typer
from rich.console import Console
from rich import print as rprint
from typing import Optional
import re
from pathlib import Path
from dotenv import load_dotenv
import boto3
import os
import s3fs
import h5py
from .configuration import Configuration
from .models import ConfigurationMeta, State

# Load environment variables from .env file
load_dotenv()

# Initialize S3 client
s3_client = boto3.client(
    's3',
    endpoint_url=os.getenv('S3_ENDPOINT_URL'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION', 'us-east-1')
)

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
    file: str = typer.Argument(
        help="Path to the xyz file (e.g. 'P150T2400_config_0.xyz')"
    ),
    output: str = typer.Argument(
        help="Directory where to output HDF5 file"
    ),
    state: Optional[State] = typer.Option(
        None,
        "--state",
        help="State of the configuration (solid or molten)"
    )
):
    """Create a configuration from a single xyz file.
    
    The filename should be in the format: P{pressure}T{temperature}_config_{number}.xyz
    
    Example usage:
        configurations create P150T2400_config_0.xyz ./output --state solid
    """
    try:
        xyz_path = Path(file)
        if not xyz_path.exists():
            raise ValueError(f"File {file} does not exist")
        
        output_path = Path(output)
        if not output_path.exists():
            output_path.mkdir(parents=True)
        
        rprint(f"[cyan]Creating HDF5 file at {output_path}...[/cyan]")

        # Extract pressure and temperature from filename
        filename = xyz_path.stem  # Get filename without extension
        pressure_match = re.search(r'P(\d+)', filename)
        temp_match = re.search(r'T(\d+)', filename)
        
        if not pressure_match or not temp_match:
            raise ValueError(f"Invalid filename format. Expected P{{pressure}}T{{temperature}}_config_{{number}}.xyz")
        
        pressure = int(pressure_match.group(1))
        temperature = int(temp_match.group(1))
        
        # Extract config number
        config_match = re.search(r'config_(\d+)', filename)
        if not config_match:
            raise ValueError(f"Invalid filename format. Could not find config number")
        MDconfig_number = int(config_match.group(1))

        # Create metadata
        meta = ConfigurationMeta(
            pressure=pressure,
            temperature=temperature,
            config_number=-100+MDconfig_number,
            state=state,
            MD_type="MD_classical"
        )

        # Create configuration
        config = Configuration(xyz_path, meta)
        rprint(f"[green]Processing {xyz_path}...[/green]")

        hdf5_file = output_path / f"P{pressure}T{temperature}_config_{MDconfig_number}.hdf5"

        # Delete HDF5 file if it exists
        if hdf5_file.exists():
            hdf5_file.unlink()
            rprint(f"[yellow]Deleted existing HDF5 file at {hdf5_file}[/yellow]")

        # Save configuration to HDF5
        config.save_to_hdf5(hdf5_file)
        rprint(f"[green]HDF5 file created successfully at {hdf5_file}[/green]")
        
        # Upload to S3
        bucket = os.getenv('BUCKET')
        prefix = os.getenv('PREFIX', '')
        if bucket:
            s3_key = f"{prefix}/{config.s3_key}" if prefix else config.s3_key
            try:
                s3_client.upload_file(hdf5_file, bucket, s3_key)
                rprint(f"[green]Successfully uploaded to S3: s3://{bucket}/{s3_key}[/green]")
            except Exception as e:
                rprint(f"[red]Failed to upload to S3: {str(e)}[/red]")
        else:
            rprint("[yellow]S3_BUCKET environment variable not set, skipping S3 upload[/yellow]")

    except Exception as e:
        rprint(f"[red]Error creating HDF5 file: {str(e)}[/red]")
        raise typer.Exit(1)

@app.command()
def list():
    """List all configurations."""
    rprint("Listing all configurations...")

@app.command()
def catalog():
    """List all HDF5 configurations in the S3 bucket and their group attributes."""
    try:
        bucket = os.getenv('BUCKET')
        prefix = os.getenv('PREFIX', '')
        
        if not bucket:
            rprint("[red]BUCKET environment variable not set[/red]")
            raise typer.Exit(1)
            
        rprint(f"[cyan]Listing HDF5 files and their attributes in s3://{bucket}/{prefix}...[/cyan]")
        
        # Initialize s3fs
        fs = s3fs.S3FileSystem(
            endpoint_url=os.getenv('S3_ENDPOINT_URL'),
            key=os.getenv('AWS_ACCESS_KEY_ID'),
            secret=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        
        # List objects in the bucket with the prefix
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith('.hdf5'):
                        try:
                            rprint(f"\n[bold green]{key}[/bold green]")
                            attributes = Configuration.read_hdf5_attributes(bucket, key, fs)
                            for group_name, group_attrs in attributes.items():
                                rprint(f"  [yellow]Group: {group_name}[/yellow]")
                                for attr_name, attr_value in group_attrs.items():
                                    rprint(f"    {attr_name}: {attr_value}")
                        except Exception as e:
                            rprint(f"[red]Error reading {key}: {str(e)}[/red]")
                        
    except Exception as e:
        rprint(f"[red]Error listing S3 files: {str(e)}[/red]")
        raise typer.Exit(1)

if __name__ == "__main__":
    app() 