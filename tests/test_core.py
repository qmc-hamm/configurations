"""Tests for the core configuration functionality."""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import shutil

from configurations.core import Configuration, ConfigurationManager

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_config():
    """Create a sample configuration."""
    atoms = ["H", "O", "H"]
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0]
    ])
    lattice_vectors = np.eye(3)
    return Configuration("water", atoms, positions, lattice_vectors)

def test_configuration_save_load(temp_dir, sample_config):
    """Test saving and loading a configuration."""
    # Save configuration
    filename = str(Path(temp_dir) / "test.h5")
    sample_config.save(filename)
    
    # Load configuration
    loaded_config = Configuration.load(filename)
    
    # Check properties
    assert loaded_config.name == sample_config.name
    assert loaded_config.atoms == sample_config.atoms
    assert np.allclose(loaded_config.positions, sample_config.positions)
    assert np.allclose(loaded_config.lattice_vectors, sample_config.lattice_vectors)

def test_configuration_manager(temp_dir):
    """Test the configuration manager."""
    manager = ConfigurationManager(temp_dir)
    
    # Create a configuration
    atoms = ["H", "H"]
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    config = manager.create("h2", atoms, positions)
    
    # Check if it exists
    assert "h2" in manager.list()
    
    # Get the configuration
    loaded_config = manager.get("h2")
    assert loaded_config.name == "h2"
    assert loaded_config.atoms == atoms
    assert np.allclose(loaded_config.positions, positions)
    
    # Delete the configuration
    manager.delete("h2")
    assert "h2" not in manager.list() 