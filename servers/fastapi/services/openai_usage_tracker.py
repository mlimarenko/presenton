import json
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Dict, Optional

from models.openai_usage_cost import (
    OpenAICostBreakdown,
    OpenAIModelUsageCost,
    OpenAIUsageCostSummary,
    OpenAIUsageTokenBreakdown,
)
from utils.get_env import get_openai_pricing_json_env

_ONE_MILLION = 1_000_000


@dataclass
class _ModelPricing:
    input_usd_per_1m: Optional[float] = None
    cached_input_usd_per_1m: Optional[float] = None
    output_usd_per_1m: Optional[float] = None

    input_text_usd_per_1m: Optional[float] = None
    cached_input_text_usd_per_1m: Optional[float] = None
    output_text_usd_per_1m: Optional[float] = None

    input_image_usd_per_1m: Optional[float] = None
    cached_input_image_usd_per_1m: Optional[float] = None
    output_image_usd_per_1m: Optional[float] = None

    @staticmethod
    def from_dict(data: dict) -> "_ModelPricing":
        return _ModelPricing(
            input_usd_per_1m=_to_float_or_none(data.get("input_usd_per_1m")),
            cached_input_usd_per_1m=_to_float_or_none(
                data.get("cached_input_usd_per_1m")
            ),
            output_usd_per_1m=_to_float_or_none(data.get("output_usd_per_1m")),
            input_text_usd_per_1m=_to_float_or_none(data.get("input_text_usd_per_1m")),
            cached_input_text_usd_per_1m=_to_float_or_none(
                data.get("cached_input_text_usd_per_1m")
            ),
            output_text_usd_per_1m=_to_float_or_none(
                data.get("output_text_usd_per_1m")
            ),
            input_image_usd_per_1m=_to_float_or_none(
                data.get("input_image_usd_per_1m")
            ),
            cached_input_image_usd_per_1m=_to_float_or_none(
                data.get("cached_input_image_usd_per_1m")
            ),
            output_image_usd_per_1m=_to_float_or_none(
                data.get("output_image_usd_per_1m")
            ),
        )

    def has_generic_pricing(self) -> bool:
        return (
            self.input_usd_per_1m is not None
            and self.cached_input_usd_per_1m is not None
            and self.output_usd_per_1m is not None
        )

    def has_modal_pricing(self) -> bool:
        return (
            self.input_text_usd_per_1m is not None
            and self.output_text_usd_per_1m is not None
            and self.input_image_usd_per_1m is not None
            and self.output_image_usd_per_1m is not None
        )


DEFAULT_OPENAI_PRICING: Dict[str, _ModelPricing] = {
    # Reference: OpenAI API pricing pages (token pricing)
    "gpt-5.2": _ModelPricing(
        input_usd_per_1m=2.0,
        cached_input_usd_per_1m=0.5,
        output_usd_per_1m=8.0,
    ),
    "gpt-image-1.5": _ModelPricing(
        input_text_usd_per_1m=5.0,
        cached_input_text_usd_per_1m=1.25,
        output_text_usd_per_1m=10.0,
        input_image_usd_per_1m=10.0,
        cached_input_image_usd_per_1m=2.5,
        output_image_usd_per_1m=40.0,
    ),
    # Backward-compatible alias
    "gpt-image-1": _ModelPricing(
        input_text_usd_per_1m=5.0,
        cached_input_text_usd_per_1m=1.25,
        output_text_usd_per_1m=10.0,
        input_image_usd_per_1m=10.0,
        cached_input_image_usd_per_1m=2.5,
        output_image_usd_per_1m=40.0,
    ),
}


_TRACKER_CONTEXT: ContextVar["OpenAIUsageTracker | None"] = ContextVar(
    "openai_usage_tracker", default=None
)


def _to_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except Exception:
        return 0


def _to_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _read(obj: Any, *path: str) -> Any:
    curr = obj
    for key in path:
        if curr is None:
            return None
        if isinstance(curr, dict):
            curr = curr.get(key)
        else:
            curr = getattr(curr, key, None)
    return curr


def _add_usage(target: OpenAIUsageTokenBreakdown, usage: OpenAIUsageTokenBreakdown):
    target.input_tokens += usage.input_tokens
    target.output_tokens += usage.output_tokens
    target.total_tokens += usage.total_tokens
    target.cached_input_tokens += usage.cached_input_tokens
    target.cached_input_text_tokens += usage.cached_input_text_tokens
    target.cached_input_image_tokens += usage.cached_input_image_tokens
    target.reasoning_output_tokens += usage.reasoning_output_tokens
    target.input_text_tokens += usage.input_text_tokens
    target.input_image_tokens += usage.input_image_tokens
    target.output_text_tokens += usage.output_text_tokens
    target.output_image_tokens += usage.output_image_tokens


class OpenAIUsageTracker:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._model_usage: Dict[str, OpenAIModelUsageCost] = {}
        self._pricing_warnings: list[str] = []
        self._pricing_source = "default"

    def activate(self):
        return _OpenAIUsageTrackerContext(self)

    def track_chat_completion_usage(self, model: str, usage: Any):
        if not self.enabled or usage is None:
            return

        usage_data = OpenAIUsageTokenBreakdown(
            input_tokens=_to_int(_read(usage, "prompt_tokens")),
            output_tokens=_to_int(_read(usage, "completion_tokens")),
            total_tokens=_to_int(_read(usage, "total_tokens")),
            cached_input_tokens=_to_int(
                _read(usage, "prompt_tokens_details", "cached_tokens")
            ),
            reasoning_output_tokens=_to_int(
                _read(usage, "completion_tokens_details", "reasoning_tokens")
            ),
        )
        self._add(model=model, usage=usage_data, is_image_request=False, images_count=0)

    def track_response_usage(self, model: str, usage: Any):
        if not self.enabled or usage is None:
            return

        usage_data = OpenAIUsageTokenBreakdown(
            input_tokens=_to_int(_read(usage, "input_tokens")),
            output_tokens=_to_int(_read(usage, "output_tokens")),
            total_tokens=_to_int(_read(usage, "total_tokens")),
            cached_input_tokens=_to_int(
                _read(usage, "input_tokens_details", "cached_tokens")
            ),
            reasoning_output_tokens=_to_int(
                _read(usage, "output_tokens_details", "reasoning_tokens")
            ),
        )
        self._add(model=model, usage=usage_data, is_image_request=False, images_count=0)

    def track_image_usage(self, model: str, usage: Any, images_count: int):
        if not self.enabled or usage is None:
            return

        output_text_tokens = _to_int(_read(usage, "output_tokens_details", "text_tokens"))
        output_image_tokens = _to_int(
            _read(usage, "output_tokens_details", "image_tokens")
        )
        output_tokens = _to_int(_read(usage, "output_tokens"))
        if output_image_tokens == 0 and output_tokens > 0:
            # images.generate usage currently reports output_tokens without per-modality breakdown.
            output_image_tokens = max(output_tokens - output_text_tokens, 0)

        cached_input_tokens = _to_int(
            _read(usage, "input_tokens_details", "cached_tokens")
        )
        cached_input_text_tokens = _to_int(
            _read(usage, "input_tokens_details", "cached_text_tokens")
        )
        cached_input_image_tokens = _to_int(
            _read(usage, "input_tokens_details", "cached_image_tokens")
        )
        if (
            cached_input_tokens > 0
            and cached_input_text_tokens == 0
            and cached_input_image_tokens == 0
        ):
            input_text_tokens = _to_int(_read(usage, "input_tokens_details", "text_tokens"))
            input_image_tokens = _to_int(
                _read(usage, "input_tokens_details", "image_tokens")
            )
            modal_total = max(input_text_tokens + input_image_tokens, 0)
            if modal_total > 0:
                cached_input_text_tokens = (
                    cached_input_tokens * input_text_tokens // modal_total
                )
                cached_input_image_tokens = max(
                    cached_input_tokens - cached_input_text_tokens,
                    0,
                )

        usage_data = OpenAIUsageTokenBreakdown(
            input_tokens=_to_int(_read(usage, "input_tokens")),
            output_tokens=output_tokens,
            total_tokens=_to_int(_read(usage, "total_tokens")),
            cached_input_tokens=cached_input_tokens,
            cached_input_text_tokens=cached_input_text_tokens,
            cached_input_image_tokens=cached_input_image_tokens,
            input_text_tokens=_to_int(
                _read(usage, "input_tokens_details", "text_tokens")
            ),
            input_image_tokens=_to_int(
                _read(usage, "input_tokens_details", "image_tokens")
            ),
            output_text_tokens=output_text_tokens,
            output_image_tokens=output_image_tokens,
        )
        self._add(
            model=model,
            usage=usage_data,
            is_image_request=True,
            images_count=images_count,
        )

    def _add(
        self,
        model: str,
        usage: OpenAIUsageTokenBreakdown,
        is_image_request: bool,
        images_count: int,
    ):
        model_key = model or "unknown"
        if model_key not in self._model_usage:
            self._model_usage[model_key] = OpenAIModelUsageCost(
                model=model_key,
                pricing_source=self._pricing_source,
            )

        model_usage = self._model_usage[model_key]
        model_usage.requests += 1
        if is_image_request:
            model_usage.image_requests += 1
            model_usage.images_generated += max(images_count, 0)
        _add_usage(model_usage.usage, usage)

    def _resolve_pricing(self) -> Dict[str, _ModelPricing]:
        self._pricing_source = "default"
        self._pricing_warnings = []

        merged = dict(DEFAULT_OPENAI_PRICING)
        pricing_json = get_openai_pricing_json_env()
        if not pricing_json:
            return merged

        try:
            overrides = json.loads(pricing_json)
            if not isinstance(overrides, dict):
                raise ValueError("OPENAI_PRICING_JSON must be a JSON object")
            for model_prefix, pricing in overrides.items():
                if isinstance(model_prefix, str) and isinstance(pricing, dict):
                    merged[model_prefix] = _ModelPricing.from_dict(pricing)
            self._pricing_source = "env_override"
        except Exception as e:
            self._pricing_warnings.append(
                f"Failed to parse OPENAI_PRICING_JSON: {str(e)}"
            )
        return merged

    def _find_pricing_for_model(
        self, model: str, catalog: Dict[str, _ModelPricing]
    ) -> Optional[_ModelPricing]:
        if model in catalog:
            return catalog[model]
        for prefix in sorted(catalog.keys(), key=lambda item: len(item), reverse=True):
            if model.startswith(prefix):
                return catalog[prefix]
        return None

    def _calculate_cost(
        self, usage: OpenAIUsageTokenBreakdown, pricing: _ModelPricing
    ) -> OpenAICostBreakdown:
        cost = OpenAICostBreakdown()

        has_modal_tokens = any(
            [
                usage.input_text_tokens,
                usage.input_image_tokens,
                usage.output_text_tokens,
                usage.output_image_tokens,
            ]
        )

        if has_modal_tokens and pricing.has_modal_pricing():
            uncached_input_text_tokens = max(
                usage.input_text_tokens - usage.cached_input_text_tokens,
                0,
            )
            uncached_input_image_tokens = max(
                usage.input_image_tokens - usage.cached_input_image_tokens,
                0,
            )

            cost.input_text_usd = (
                uncached_input_text_tokens
                * pricing.input_text_usd_per_1m
                / _ONE_MILLION
            )
            if pricing.cached_input_text_usd_per_1m is not None:
                cost.cached_input_text_usd = (
                    usage.cached_input_text_tokens
                    * pricing.cached_input_text_usd_per_1m
                    / _ONE_MILLION
                )
            else:
                cost.cached_input_text_usd = (
                    usage.cached_input_text_tokens
                    * pricing.input_text_usd_per_1m
                    / _ONE_MILLION
                )

            cost.input_image_usd = (
                uncached_input_image_tokens
                * pricing.input_image_usd_per_1m
                / _ONE_MILLION
            )
            if pricing.cached_input_image_usd_per_1m is not None:
                cost.cached_input_image_usd = (
                    usage.cached_input_image_tokens
                    * pricing.cached_input_image_usd_per_1m
                    / _ONE_MILLION
                )
            else:
                cost.cached_input_image_usd = (
                    usage.cached_input_image_tokens
                    * pricing.input_image_usd_per_1m
                    / _ONE_MILLION
                )

            cost.output_text_usd = (
                usage.output_text_tokens
                * pricing.output_text_usd_per_1m
                / _ONE_MILLION
            )
            cost.output_image_usd = (
                usage.output_image_tokens
                * pricing.output_image_usd_per_1m
                / _ONE_MILLION
            )
        elif pricing.has_generic_pricing():
            uncached_input_tokens = max(
                usage.input_tokens - usage.cached_input_tokens,
                0,
            )
            cost.input_usd = (
                uncached_input_tokens * pricing.input_usd_per_1m / _ONE_MILLION
            )
            cost.cached_input_usd = (
                usage.cached_input_tokens
                * pricing.cached_input_usd_per_1m
                / _ONE_MILLION
            )
            cost.output_usd = (
                usage.output_tokens * pricing.output_usd_per_1m / _ONE_MILLION
            )

        cost.total_usd = round(
            cost.input_usd
            + cost.cached_input_usd
            + cost.output_usd
            + cost.input_text_usd
            + cost.cached_input_text_usd
            + cost.output_text_usd
            + cost.input_image_usd
            + cost.cached_input_image_usd
            + cost.output_image_usd,
            8,
        )
        return cost

    def build_summary(self) -> OpenAIUsageCostSummary:
        if not self.enabled:
            return OpenAIUsageCostSummary(enabled=False)

        pricing_catalog = self._resolve_pricing()
        model_summaries: list[OpenAIModelUsageCost] = []
        unpriced_models: list[str] = []

        total_usage = OpenAIUsageTokenBreakdown()
        total_cost = OpenAICostBreakdown()
        requests_total = 0
        image_requests_total = 0
        images_generated_total = 0

        for model in sorted(self._model_usage.keys()):
            model_summary = self._model_usage[model]
            model_summary.pricing_source = self._pricing_source
            model_summary.priced = True
            model_summary.warnings = []
            model_summary.cost = OpenAICostBreakdown()

            pricing = self._find_pricing_for_model(model, pricing_catalog)
            if pricing is None:
                model_summary.priced = False
                model_summary.warnings.append(
                    "Pricing for this model is not configured"
                )
                unpriced_models.append(model)
            else:
                model_summary.cost = self._calculate_cost(model_summary.usage, pricing)

            model_summaries.append(model_summary)
            _add_usage(total_usage, model_summary.usage)
            requests_total += model_summary.requests
            image_requests_total += model_summary.image_requests
            images_generated_total += model_summary.images_generated

            total_cost.input_usd += model_summary.cost.input_usd
            total_cost.cached_input_usd += model_summary.cost.cached_input_usd
            total_cost.output_usd += model_summary.cost.output_usd
            total_cost.input_text_usd += model_summary.cost.input_text_usd
            total_cost.cached_input_text_usd += model_summary.cost.cached_input_text_usd
            total_cost.output_text_usd += model_summary.cost.output_text_usd
            total_cost.input_image_usd += model_summary.cost.input_image_usd
            total_cost.cached_input_image_usd += (
                model_summary.cost.cached_input_image_usd
            )
            total_cost.output_image_usd += model_summary.cost.output_image_usd
            total_cost.total_usd += model_summary.cost.total_usd

        total_cost.total_usd = round(total_cost.total_usd, 8)

        warnings = [*self._pricing_warnings]
        if unpriced_models:
            warnings.append(
                "Some models are unpriced. Total cost excludes those model calls."
            )

        return OpenAIUsageCostSummary(
            enabled=True,
            requests=requests_total,
            image_requests=image_requests_total,
            images_generated=images_generated_total,
            usage=total_usage,
            cost=total_cost,
            models=model_summaries,
            unpriced_models=unpriced_models,
            warnings=warnings,
        )


class _OpenAIUsageTrackerContext:
    def __init__(self, tracker: OpenAIUsageTracker):
        self.tracker = tracker
        self._token: Optional[Token] = None

    def __enter__(self):
        self._token = _TRACKER_CONTEXT.set(self.tracker)
        return self.tracker

    def __exit__(self, exc_type, exc, tb):
        if self._token is not None:
            _TRACKER_CONTEXT.reset(self._token)


def get_openai_usage_tracker() -> Optional[OpenAIUsageTracker]:
    return _TRACKER_CONTEXT.get()


def track_openai_chat_completion_usage(model: str, usage: Any):
    tracker = get_openai_usage_tracker()
    if tracker:
        tracker.track_chat_completion_usage(model=model, usage=usage)


def track_openai_response_usage(model: str, usage: Any):
    tracker = get_openai_usage_tracker()
    if tracker:
        tracker.track_response_usage(model=model, usage=usage)


def track_openai_image_usage(model: str, usage: Any, images_count: int):
    tracker = get_openai_usage_tracker()
    if tracker:
        tracker.track_image_usage(model=model, usage=usage, images_count=images_count)
