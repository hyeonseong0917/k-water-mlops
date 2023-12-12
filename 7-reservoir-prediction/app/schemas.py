from pydantic import BaseModel, Field
from typing import List, Tuple, Dict, Optional

## ge : greater than or equal / le : less than or equal
class Reservoir(BaseModel):
    data: Dict[str, float]

class Flow(BaseModel):
    data: List[float]