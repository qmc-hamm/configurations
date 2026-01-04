from enum import Enum
from typing import Optional
from pydantic import BaseModel

class State(str, Enum):
    SOLID = "solid"
    LIQUID = "liquid"
    AMBIGUOUS = "ambiguous"

class ConfigurationMeta(BaseModel):
    config_number: Optional[int] = None
    uuid: Optional[str] = None
    
    # Thermodynamic conditions
    pressure: Optional[int] = None  # GPa
    temperature: Optional[int] = None  # K
    state: Optional[State] = None  # phase: solid/liquid/molten/ambiguous
    
    # Material properties
    rs: Optional[float] = None  # Wigner-Seitz radius
    molecular_percentage: Optional[float] = None
    
    # Simulation parameters
    MD_type: Optional[str] = None  # e.g., "MD_classical"
    method: Optional[str] = None  # e.g., "NPT", "NVT"
    code: Optional[str] = None  # e.g., "LAMMPS"
    modelname: Optional[str] = None  # e.g., "M18"
    simulation_type: Optional[str] = None  # e.g., "classical"
    timestep: Optional[int] = None
    
    # Metadata
    config_gen_date: Optional[str] = None  # Date string
    author: Optional[str] = None
    qmc_machine: Optional[str] = None  # e.g., "Aurora"
    QMC_run_date: Optional[str] = None  # Date string format: YYYY_MM_DD
    QMC_quality: Optional[int] = None  # Quality score (0-10)
    
    # Energy properties (in eV)
    energy: Optional[float] = None  # Total energy
    electron_kinetic_energy: Optional[float] = None
    potential_energy: Optional[float] = None
    fsc_potential_energy: Optional[float] = None  # Finite-size correction for potential energy
    fsc_kinetic_energy: Optional[float] = None  # Finite-size correction for kinetic energy
    