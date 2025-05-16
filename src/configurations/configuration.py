from pathlib import Path
from rich.console import Console
from rich.panel import Panel
import h5py
from .models import ConfigurationMeta
import s3fs

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
    
    @property
    def s3_key(self) -> str:
        return f"P{self.meta.pressure}/T{self.meta.temperature}/{self.xyz_path.name}"
    
    
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

    @staticmethod
    def read_hdf5_attributes(bucket: str, key: str, fs: s3fs.S3FileSystem) -> dict:
        """Read all group attributes from an HDF5 file in S3.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            fs: S3FileSystem instance for accessing S3
            
        Returns:
            A dictionary mapping group names to their attributes
            
        Raises:
            Exception: If there's an error reading the file
        """
        s3_path = f"s3://{bucket}/{key}"
        with fs.open(s3_path, 'rb') as f:
            with h5py.File(f, 'r') as h5f:
                attributes = {}
                for group_name, group in h5f.items():
                    attributes[group_name] = dict(group.attrs)
                return attributes