from pydantic import BaseModel


class Data(BaseModel):
    text: str = ...
