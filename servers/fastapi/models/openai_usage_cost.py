from typing import List, Literal

from pydantic import BaseModel, Field


class OpenAIUsageTokenBreakdown(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    cached_input_text_tokens: int = 0
    cached_input_image_tokens: int = 0
    reasoning_output_tokens: int = 0

    input_text_tokens: int = 0
    input_image_tokens: int = 0
    output_text_tokens: int = 0
    output_image_tokens: int = 0


class OpenAICostBreakdown(BaseModel):
    currency: Literal["USD"] = "USD"

    input_usd: float = 0.0
    cached_input_usd: float = 0.0
    output_usd: float = 0.0

    input_text_usd: float = 0.0
    cached_input_text_usd: float = 0.0
    output_text_usd: float = 0.0
    input_image_usd: float = 0.0
    cached_input_image_usd: float = 0.0
    output_image_usd: float = 0.0

    total_usd: float = 0.0


class OpenAIModelUsageCost(BaseModel):
    model: str
    requests: int = 0
    image_requests: int = 0
    images_generated: int = 0

    usage: OpenAIUsageTokenBreakdown = Field(
        default_factory=OpenAIUsageTokenBreakdown
    )
    cost: OpenAICostBreakdown = Field(default_factory=OpenAICostBreakdown)

    priced: bool = True
    pricing_source: str = "default"
    warnings: List[str] = Field(default_factory=list)


class OpenAIUsageCostSummary(BaseModel):
    enabled: bool
    provider: Literal["openai"] = "openai"

    requests: int = 0
    image_requests: int = 0
    images_generated: int = 0

    usage: OpenAIUsageTokenBreakdown = Field(
        default_factory=OpenAIUsageTokenBreakdown
    )
    cost: OpenAICostBreakdown = Field(default_factory=OpenAICostBreakdown)

    models: List[OpenAIModelUsageCost] = Field(default_factory=list)
    unpriced_models: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
