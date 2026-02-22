from types import SimpleNamespace

from services.openai_usage_tracker import (
    OpenAIUsageTracker,
    track_openai_chat_completion_usage,
    track_openai_image_usage,
)


def _completion_usage(
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    cached_tokens: int = 0,
    reasoning_tokens: int = 0,
):
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
    )


def _image_usage(
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    input_text_tokens: int,
    input_image_tokens: int,
    output_text_tokens: int = 0,
    output_image_tokens: int = 0,
    cached_tokens: int = 0,
    cached_text_tokens: int = 0,
    cached_image_tokens: int = 0,
):
    input_details = dict(
        text_tokens=input_text_tokens,
        image_tokens=input_image_tokens,
    )
    if cached_tokens:
        input_details["cached_tokens"] = cached_tokens
    if cached_text_tokens:
        input_details["cached_text_tokens"] = cached_text_tokens
    if cached_image_tokens:
        input_details["cached_image_tokens"] = cached_image_tokens

    output_details = None
    if output_text_tokens or output_image_tokens:
        output_details = SimpleNamespace(
            text_tokens=output_text_tokens,
            image_tokens=output_image_tokens,
        )

    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        input_tokens_details=SimpleNamespace(**input_details),
        output_tokens_details=output_details,
    )


def test_openai_usage_tracker_disabled():
    tracker = OpenAIUsageTracker(enabled=False)
    with tracker.activate():
        track_openai_chat_completion_usage(
            model="gpt-5.2",
            usage=_completion_usage(100, 50, 150),
        )

    summary = tracker.build_summary()
    assert summary.enabled is False
    assert summary.requests == 0
    assert summary.cost.total_usd == 0.0


def test_openai_usage_tracker_chat_pricing_and_cached_tokens():
    tracker = OpenAIUsageTracker(enabled=True)
    with tracker.activate():
        track_openai_chat_completion_usage(
            model="gpt-5.2",
            usage=_completion_usage(
                prompt_tokens=1000,
                completion_tokens=500,
                total_tokens=1500,
                cached_tokens=200,
                reasoning_tokens=25,
            ),
        )

    summary = tracker.build_summary()
    assert summary.enabled is True
    assert summary.requests == 1
    assert summary.usage.input_tokens == 1000
    assert summary.usage.output_tokens == 500
    assert summary.usage.cached_input_tokens == 200
    assert summary.usage.reasoning_output_tokens == 25

    # gpt-5.2 defaults:
    # uncached_input: 800 * 2.0 / 1e6 = 0.0016
    # cached_input:   200 * 0.5 / 1e6 = 0.0001
    # output:         500 * 8.0 / 1e6 = 0.004
    assert summary.cost.total_usd == 0.0057
    assert summary.unpriced_models == []


def test_openai_usage_tracker_image_modal_pricing():
    tracker = OpenAIUsageTracker(enabled=True)
    with tracker.activate():
        track_openai_image_usage(
            model="gpt-image-1.5",
            usage=_image_usage(
                input_tokens=400,
                output_tokens=250,
                total_tokens=650,
                input_text_tokens=100,
                input_image_tokens=300,
            ),
            images_count=1,
        )

    summary = tracker.build_summary()
    assert summary.requests == 1
    assert summary.image_requests == 1
    assert summary.images_generated == 1

    assert summary.usage.input_text_tokens == 100
    assert summary.usage.input_image_tokens == 300
    assert summary.usage.output_text_tokens == 0
    assert summary.usage.output_image_tokens == 250

    # gpt-image-1.5 defaults:
    # input_text:  100 * 5   / 1e6 = 0.0005
    # input_image: 300 * 10  / 1e6 = 0.003
    # output_img:  250 * 40  / 1e6 = 0.01
    assert summary.cost.total_usd == 0.0135


def test_openai_usage_tracker_image_modal_cached_pricing():
    tracker = OpenAIUsageTracker(enabled=True)
    with tracker.activate():
        track_openai_image_usage(
            model="gpt-image-1.5",
            usage=_image_usage(
                input_tokens=400,
                output_tokens=100,
                total_tokens=500,
                input_text_tokens=100,
                input_image_tokens=300,
                cached_tokens=40,
                cached_text_tokens=10,
                cached_image_tokens=30,
            ),
            images_count=1,
        )

    summary = tracker.build_summary()

    # text uncached: 90 * 5 / 1e6 = 0.00045
    # text cached:   10 * 1.25 / 1e6 = 0.0000125
    # image uncached:270 * 10 / 1e6 = 0.0027
    # image cached:   30 * 2.5 / 1e6 = 0.000075
    # output image:  100 * 40 / 1e6 = 0.004
    assert summary.cost.total_usd == 0.0072375
    assert summary.usage.cached_input_tokens == 40
    assert summary.usage.cached_input_text_tokens == 10
    assert summary.usage.cached_input_image_tokens == 30


def test_openai_usage_tracker_unpriced_model(monkeypatch):
    monkeypatch.delenv("OPENAI_PRICING_JSON", raising=False)

    tracker = OpenAIUsageTracker(enabled=True)
    with tracker.activate():
        track_openai_chat_completion_usage(
            model="unknown-model",
            usage=_completion_usage(100, 50, 150),
        )

    summary = tracker.build_summary()
    assert summary.requests == 1
    assert summary.cost.total_usd == 0.0
    assert summary.unpriced_models == ["unknown-model"]
    assert len(summary.warnings) > 0


def test_openai_usage_tracker_pricing_override(monkeypatch):
    monkeypatch.setenv(
        "OPENAI_PRICING_JSON",
        '{"gpt-5.2":{"input_usd_per_1m":10,"cached_input_usd_per_1m":1,"output_usd_per_1m":20}}',
    )

    tracker = OpenAIUsageTracker(enabled=True)
    with tracker.activate():
        track_openai_chat_completion_usage(
            model="gpt-5.2",
            usage=_completion_usage(
                prompt_tokens=1000,
                completion_tokens=500,
                total_tokens=1500,
                cached_tokens=100,
            ),
        )

    summary = tracker.build_summary()
    # uncached_input: 900 * 10 / 1e6 = 0.009
    # cached_input:   100 * 1  / 1e6 = 0.0001
    # output:         500 * 20 / 1e6 = 0.01
    assert summary.cost.total_usd == 0.0191
    assert summary.models[0].pricing_source == "env_override"
