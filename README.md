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

The `configurations` tool provides a command-line interface for managing atomic configurations:

```bash
# List available commands
configurations --help

# Create a new configuration
configurations create --name my_config --atoms "H2O"

# List all configurations
configurations list

# View details of a specific configuration
configurations show my_config
```

## Development

To run tests:

```bash
hatch run test
```

## License

MIT License 