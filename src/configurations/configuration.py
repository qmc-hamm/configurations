from pathlib import Path
from rich.console import Console
from rich.panel import Panel
import h5py
from .models import ConfigurationMeta, State
import s3fs
from typing import Dict, Any, Optional
from ase import io

def parse_xyz_header(xyz_path: Path) -> ConfigurationMeta:
    """Parse metadata from XYZ file using ASE and create ConfigurationMeta object.
    
    Uses ASE (Atomic Simulation Environment) to read the XYZ file and extract
    metadata from atoms.info dictionary. This is cleaner and more reliable than
    parsing the raw text.
    
    Args:
        xyz_path: Path to the XYZ file
        
    Returns:
        ConfigurationMeta object with parsed metadata
        
    Raises:
        FileNotFoundError: If XYZ file doesn't exist
        ValueError: If XYZ file format is invalid
    """
    # Use ASE to read the XYZ file - it automatically parses metadata into atoms.info
    atoms = io.read(str(xyz_path))
    
    # Get the info dictionary from ASE (contains all metadata from XYZ header)
    info = atoms.info
    
    # Mapping from atoms.info field names to ConfigurationMeta field names
    field_mapping: Dict[str, str] = {
        'config': 'config_number',
        'phase': 'state',  # phase -> state
        'QMC-run-date': 'QMC_run_date',  # hyphen to underscore
        'QMC-quality': 'QMC_quality',  # hyphen to underscore
        'simulation-type': 'simulation_type',  # hyphen to underscore
        'fsc_dv_ev': 'fsc_potential_energy',  # dv -> potential
        'fsc_dt_ev': 'fsc_kinetic_energy',  # dt -> kinetic
    }
    
    # Build the metadata dictionary
    meta_dict: Dict[str, Any] = {}
    
    for info_key, info_value in info.items():
        # Map field name if needed
        meta_key = field_mapping.get(info_key, info_key)
        
        # Skip fields that don't exist in ConfigurationMeta
        if not meta_key in ConfigurationMeta.model_fields:
            continue
        
        # Type conversion based on field name
        try:
            if meta_key == 'state':
                # Convert phase string to State enum
                if isinstance(info_value, str):
                    phase_lower = info_value.lower()
                    if phase_lower == 'solid':
                        meta_dict[meta_key] = State.SOLID
                    elif phase_lower == 'liquid':
                        meta_dict[meta_key] = State.LIQUID
                    elif phase_lower == 'ambiguous':
                        meta_dict[meta_key] = State.AMBIGUOUS
                    else:
                        meta_dict[meta_key] = None
                else:
                    meta_dict[meta_key] = None
            elif meta_key in ['config_number', 'pressure', 'temperature', 'timestep', 'QMC_quality']:
                # Integer fields
                meta_dict[meta_key] = int(info_value) if info_value is not None else None
            elif meta_key in ['rs', 'molecular_percentage', 'energy', 'electron_kinetic_energy', 
                             'potential_energy', 'fsc_potential_energy', 'fsc_kinetic_energy']:
                # Float fields
                meta_dict[meta_key] = float(info_value) if info_value is not None else None
            else:
                # String fields (keep as string, or convert to string if needed)
                meta_dict[meta_key] = str(info_value) if info_value is not None else None
        except (ValueError, TypeError, AttributeError):
            # If conversion fails, skip this field
            continue
    
    # Create and return ConfigurationMeta object
    return ConfigurationMeta(**meta_dict)

class Configuration:
    def __init__(self, xyz_path: Path):
        self.xyz_path = xyz_path
        self.meta = parse_xyz_header(self.xyz_path)
        
        # Find related files in the same directory
        self.sofk_txt_path = self._find_related_file("_sofk.txt")
        self.gofr_txt_path = self._find_related_file("_gofr.txt")
        self.sk_path = self._find_related_file(".sk")
    @property
    def hdf5_filename(self) -> str:
        return f"P{self.meta.pressure}T{self.meta.temperature}_config_{self.meta.config_number}.hdf5"

    def _find_related_file(self, suffix: str) -> Optional[Path]:
        """Find a related file by appending or replacing suffix in the XYZ filename.
        
        Args:
            suffix: Suffix to look for (e.g., "_sofk.txt", "_gofr.txt", ".sk")
            
        Returns:
            Path to the file if found, None otherwise
        """
        xyz_dir = self.xyz_path.parent
        xyz_stem = self.xyz_path.stem  # filename without extension
        
        # For .sk file, replace .xyz extension with .sk
        if suffix == ".sk":
            file_path = xyz_dir / f"{xyz_stem}.sk"
        else:
            # For _sofk.txt and _gofr.txt, append suffix
            file_path = xyz_dir / f"{xyz_stem}{suffix}"
        
        return file_path if file_path.exists() else None

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
        return f"P{self.meta.pressure}/T{self.meta.temperature}/{self.hdf5_filename}"
    
    def save_to_hdf5(self, hdf5_path: Path):
        """Save the configuration and all related files to an HDF5 file.
        
        Saves:
        - All metadata as root-level attributes
        - XYZ file content as 'xyz_data' dataset
        - SOFK file content as 'sofk_data' dataset (if exists)
        - GOFR file content as 'gofr_data' dataset (if exists)
        - SK file content as 'electronic_sk_data' dataset (if exists)
        
        Args:
            hdf5_path: Path to the output HDF5 file
        """
        # Use 'w' mode to create new file (overwrite if exists)
        with h5py.File(hdf5_path, "w") as hdf5_file:
            # Save all metadata as root-level attributes
            # Convert metadata to dictionary and save all fields
            try:
                meta_dict = self.meta.model_dump()  # Pydantic v2
            except AttributeError:
                meta_dict = self.meta.dict()  # Pydantic v1 fallback
            
            for key, value in meta_dict.items():
                if value is not None:
                    # Convert State enum to string (already handled by model_dump/dict, but check anyway)
                    if isinstance(value, State):
                        hdf5_file.attrs[key] = value.value
                    # Convert lists/other types to string if needed
                    elif isinstance(value, (list, dict)):
                        hdf5_file.attrs[key] = str(value)
                    else:
                        hdf5_file.attrs[key] = value
            
            # Save XYZ file content
            with open(self.xyz_path, "r") as xyz_file:
                xyz_content = xyz_file.read()
                hdf5_file.create_dataset("xyz_data", data=xyz_content.encode('utf-8'))
            
            # Save SOFK file content (if exists)
            if self.sofk_txt_path:
                with open(self.sofk_txt_path, "r") as sofk_file:
                    sofk_content = sofk_file.read()
                    hdf5_file.create_dataset("sofk_data", data=sofk_content.encode('utf-8'))
            
            # Save GOFR file content (if exists)
            if self.gofr_txt_path:
                with open(self.gofr_txt_path, "r") as gofr_file:
                    gofr_content = gofr_file.read()
                    hdf5_file.create_dataset("gofr_data", data=gofr_content.encode('utf-8'))
            
            # Save SK file content (if exists)
            if self.sk_path:
                with open(self.sk_path, "r") as sk_file:
                    sk_content = sk_file.read()
                    hdf5_file.create_dataset("electronic_sk_data", data=sk_content.encode('utf-8'))

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
                return dict(h5f.attrs)