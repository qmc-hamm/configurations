#!/usr/bin/env python3
"""Script to create an example HDF5 file from a configuration."""

import sys
from pathlib import Path
sys.path.insert(0, 'src')

from configurations.configuration import parse_xyz_header, Configuration

# Create example HDF5 file
xyz_path = Path('/projects/illinois/grants/qmchamm/shared/shubhang/aurora_backup/run_2025_05_25/runs/LLPT_261_configs/P150T2000config60/P150T2000config60.xyz')
output_path = Path('/projects/illinois/grants/qmchamm/shared/shubhang/aurora_backup/database_work/configurations/example.hdf5')

print("Creating example HDF5 file...")
print(f"Input XYZ: {xyz_path}")
print(f"Output HDF5: {output_path}")

# Parse metadata from XYZ file
print("\n1. Parsing metadata from XYZ file...")
meta = parse_xyz_header(xyz_path)
print(f"   ✓ Parsed metadata:")
print(f"     - pressure: {meta.pressure}")
print(f"     - temperature: {meta.temperature}")
print(f"     - config_number: {meta.config_number}")
print(f"     - state: {meta.state}")

# Create Configuration object
print("\n2. Creating Configuration object...")
config = Configuration(xyz_path, meta)
print(f"   ✓ Configuration created")
print(f"     - xyz: {config.xyz_path.name}")
print(f"     - sofk_txt: {config.sofk_txt_path.name if config.sofk_txt_path else 'NOT FOUND'}")
print(f"     - gofr_txt: {config.gofr_txt_path.name if config.gofr_txt_path else 'NOT FOUND'}")
print(f"     - sk: {config.sk_path.name if config.sk_path else 'NOT FOUND'}")

# Save to HDF5
print("\n3. Saving to HDF5...")
config.save_to_hdf5(output_path)
print(f"   ✓ HDF5 file created at: {output_path}")

# Verify the HDF5 file
print("\n4. Verifying HDF5 file...")
import h5py
with h5py.File(output_path, 'r') as f:
    print(f"   Attributes ({len(f.attrs)}):")
    for key in sorted(f.attrs.keys())[:10]:  # Show first 10
        value = f.attrs[key]
        if isinstance(value, bytes):
            value = value.decode('utf-8') if len(str(value)) < 50 else f"{str(value)[:47]}..."
        print(f"     {key}: {value}")
    if len(f.attrs) > 10:
        print(f"     ... and {len(f.attrs) - 10} more attributes")
    
    print(f"\n   Datasets ({len(f.keys())}):")
    for key in sorted(f.keys()):
        dataset = f[key]
        print(f"     {key}: shape={dataset.shape}, dtype={dataset.dtype}")

print(f"\n✓ Example HDF5 file created successfully!")
print(f"  File: {output_path}")
print(f"  Size: {output_path.stat().st_size / 1024:.2f} KB")

