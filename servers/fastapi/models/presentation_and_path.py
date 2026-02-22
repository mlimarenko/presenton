import uuid
from typing import Optional

from pydantic import BaseModel

from models.openai_usage_cost import OpenAIUsageCostSummary


class PresentationAndPath(BaseModel):
    presentation_id: uuid.UUID
    path: str


class PresentationPathAndEditPath(PresentationAndPath):
    edit_path: str
    openai_usage: Optional[OpenAIUsageCostSummary] = None
