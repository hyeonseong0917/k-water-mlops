from pydantic import BaseModel, Field, validator
from typing import List, Tuple, Dict, Optional

## ge : greater than or equal / le : less than or equal
class Reservoir(BaseModel):
    data: Dict[str, float]

class Flow(BaseModel):
    data: List[float]
    # validator를 사용하여 값을 할당하기 전에 반올림 로직 적용
    @validator('data', pre=True, each_item=True)
    def round_data(self, value):
        return round(value, 4)  # 원하는 반올림 자릿수로 수정