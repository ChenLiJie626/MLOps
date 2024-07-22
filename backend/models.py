
from typing import Any, Optional, Union
from pydantic import BaseModel


class TrainApiData(BaseModel):
    model_name: str
    hyperparams: dict[str, Any]
    epochs: int


class PredictApiData(BaseModel):
    input_image: Any
    model_name: str


class DeleteApiData(BaseModel):
    model_name: str
    model_version: Optional[Union[list[int], int]]  # list | int in python 10

class User(BaseModel):
    server_address: str # server address
    name: str
    role: str
    gpu: str