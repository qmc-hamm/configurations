"""Core functionality for managing atomic configurations."""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class Configuration:
    """Represents an atomic configuration."""
    
    def __init__(self, name: str, atoms: List[str], positions: np.ndarray, 
                 lattice_vectors: Optional[np.ndarray] = None):
        """Initialize a configuration.
        
        Args:
            name: Name of the configuration
            atoms: List of atomic symbols
            positions: Array of atomic positions (N x 3)
            lattice_vectors: Optional lattice vectors (3 x 3)
        """
        self.name = name
        self.atoms = atoms
        self.positions = positions
        self.lattice_vectors = lattice_vectors
        
    def save(self, filename: str) -> None:
        """Save configuration to HDF5 file."""
        with h5py.File(filename, 'w') as f:
            f.create_dataset('name', data=self.name)
            f.create_dataset('atoms', data=np.array(self.atoms, dtype='S'))
            f.create_dataset('positions', data=self.positions)
            if self.lattice_vectors is not None:
                f.create_dataset('lattice_vectors', data=self.lattice_vectors)
    
    @classmethod
    def load(cls, filename: str) -> 'Configuration':
        """Load configuration from HDF5 file."""
        with h5py.File(filename, 'r') as f:
            name = f['name'][()].decode('utf-8')
            atoms = [a.decode('utf-8') for a in f['atoms'][()]]
            positions = f['positions'][()]
            lattice_vectors = f['lattice_vectors'][()] if 'lattice_vectors' in f else None
            return cls(name, atoms, positions, lattice_vectors)

class ConfigurationManager:
    """Manages a collection of configurations."""
    
    def __init__(self, storage_dir: str = "configurations"):
        """Initialize the configuration manager.
        
        Args:
            storage_dir: Directory to store configuration files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
    
    def create(self, name: str, atoms: List[str], positions: np.ndarray,
              lattice_vectors: Optional[np.ndarray] = None) -> Configuration:
        """Create a new configuration."""
        config = Configuration(name, atoms, positions, lattice_vectors)
        config.save(str(self.storage_dir / f"{name}.h5"))
        return config
    
    def get(self, name: str) -> Configuration:
        """Get a configuration by name."""
        return Configuration.load(str(self.storage_dir / f"{name}.h5"))
    
    def list(self) -> List[str]:
        """List all available configurations."""
        return [f.stem for f in self.storage_dir.glob("*.h5")]
    
    def delete(self, name: str) -> None:
        """Delete a configuration."""
        (self.storage_dir / f"{name}.h5").unlink() 