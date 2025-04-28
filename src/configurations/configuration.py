from pathlib import Path
from rich.console import Console
from rich.panel import Panel
import h5py
from .models import ConfigurationMeta

class Configuration:
    def __init__(self, xyz_path: Path, meta: ConfigurationMeta):
        self.xyz_path = xyz_path
        self.meta = meta

    def __str__(self) -> str:
        console = Console()
        with console.capture() as capture:
            console.print(Panel.fit(
                f"XYZ File: {self.xyz_path.name}\n"
                f"Pressure: {self.meta.pressure or 'N/A'}\n"
                f"Temperature: {self.meta.temperature or 'N/A'}\n"
                f"State: {self.meta.state.value if self.meta.state else 'N/A'}",
                f"MD Type: {self.meta.MD_type or 'N/A'}",
                f"config_number: {self.meta.config_number or 'N/A'}",
                border_style="cyan",
                box="SIMPLE",
                title="Configuration Details"
            ))
        return capture.get() 
    
    
    def save_to_hdf5(self, hdf5_path: Path):
        """Save the configuration and metadata to an HDF5 file."""
        with h5py.File(hdf5_path, "a") as hdf5_file:
            group_name = f"{self.meta.MD_type}/{self.meta.pressure}/{self.meta.temperature}"
            group = hdf5_file.require_group(group_name)

            # Save metadata
            group.attrs["pressure"] = self.meta.pressure or "N/A"
            group.attrs["temperature"] = self.meta.temperature or "N/A"
            group.attrs["state"] = self.meta.state.value if self.meta.state else "N/A"
            group.attrs["MD_type"] = self.meta.MD_type or "N/A"
            group.attrs["config_number"]= self.meta.config_number or "N/A"

            # Save XYZ file content
            with open(self.xyz_path, "r") as xyz_file:
                group.create_dataset("xyz_data", data=xyz_file.read())