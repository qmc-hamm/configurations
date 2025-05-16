from enum import Enum
from typing import Optional
from pydantic import BaseModel

class State(str, Enum):
    SOLID = "solid"
    MOLTEN = "molten"

class ConfigurationMeta(BaseModel):
    config_number: Optional[int] = None
    pressure: Optional[int] = None
    temperature: Optional[int] = None
    state: Optional[State] = None
    MD_type: Optional[str] = None
    