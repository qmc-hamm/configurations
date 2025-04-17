# Configurations

A Python tool for managing atomic configurations in HDF5 format, designed for use with QMC-HAMM project.

## Installation

This project uses `uv` as the dependency manager and `hatch` as the build system. To install:

```bash
# Install uv if you haven't already
pip install uv

# Install the package in development mode
uv pip install -e .
```

## Usage

The `configurations` tool provides a command-line interface for creating atomic configurations from xyz files in a structured directory format.

### Creating Configurations

The `create` command processes xyz files from a directory structure that follows this pattern:
```
data_dir/
    P{pressure}/
        T{temperature}/
            *.xyz
```

Where:
- `P{pressure}` directories contain pressure values (e.g., `P225` for 225 GPa)
- `T{temperature}` directories contain temperature values (e.g., `T1000` for 1000 K)
- Each temperature directory contains xyz files with atomic configurations

To create configurations:

```bash
# Create configurations from a directory structure
configurations create /path/to/data_dir
```

The tool will:
1. Process each pressure directory (P*)
2. For each pressure, process temperature directories (T*)
3. Create configurations from all xyz files found
4. Automatically extract pressure and temperature values from directory names
5. Print information about each configuration as it's created

Example output:
```
Configuration: config1.xyz
Pressure: 225 GPa
Temperature: 1000 K
Atoms: H2O
```

## Development

To run tests:

```bash
hatch run test
```

## License

MIT License 