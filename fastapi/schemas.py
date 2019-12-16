from typing import List

from pydantic import BaseModel
from enum import Enum


class TextSample(BaseModel):
    text: str


class RequestBody(BaseModel):
    samples: List[TextSample]

    def to_array(self):
        return [sample.text for sample in self.samples]


class ResponseValues(str, Enum):
    hockey = "rec.sport.hockey"
    space = "sci.space"
    politics = "talk.politics.misc"


class ResponseBody(BaseModel):
    predictions: List[str]
    probabilities: List[float]


class LabelResponseBody(BaseModel):
    label: str
    probabilities: List[float]
