from enum import Enum
from typing import Optional
from pydantic import BaseModel

class State(str, Enum):
    SOLID = "solid"
    LIQUID = "liquid"

class ConfigurationMeta(BaseModel):
    pressure: Optional[int] = None
    temperature: Optional[int] = None
    state: Optional[State] = None 