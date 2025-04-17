from pathlib import Path
from rich.console import Console
from rich.panel import Panel
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
                title="Configuration Details"
            ))
        return capture.get() 