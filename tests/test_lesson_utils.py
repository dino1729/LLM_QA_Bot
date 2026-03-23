from unittest.mock import MagicMock, patch

from helper_functions.lesson_utils import (
    _get_fallback_lesson,
    generate_lesson_response,
    parse_lesson_to_dict,
)


def test_lesson_generation_retries_with_tier_default_after_model_failure():
    """Explicit model failures should retry with the tier-default model."""
    prompt = """Topic: How does the Weber-Fechner Law explain our perception of price differences?

YOU MUST START YOUR RESPONSE WITH "[KEY INSIGHT]" - this is mandatory."""

    failing_client = MagicMock()
    failing_client.chat_completion.side_effect = Exception("410 Gone")

    fallback_client = MagicMock()
    fallback_client.chat_completion.return_value = """[KEY INSIGHT]
People perceive change proportionally, not absolutely, which means the same delta can feel massive or negligible depending on the baseline.

[HISTORICAL]
Nineteenth-century psychophysicists Ernst Weber and Gustav Fechner showed that perceived sensory change scales with relative difference, not raw magnitude. Their work became the foundation for modern thinking about thresholds, pricing perception, and human-centered measurement.

[APPLICATION]
In engineering and product leadership, evaluate system changes relative to the user's current baseline. Small absolute improvements can create major adoption gains when they cross a meaningful perception threshold."""

    with patch(
        "helper_functions.lesson_utils.get_client",
        side_effect=[failing_client, fallback_client],
    ) as mock_get_client:
        result = generate_lesson_response(
            prompt,
            "litellm",
            model_tier="fast",
            model_name="retired-model",
        )

    parsed = parse_lesson_to_dict(result, "Weber-Fechner")

    assert mock_get_client.call_count == 2
    assert mock_get_client.call_args_list[0].kwargs["model_name"] == "retired-model"
    assert mock_get_client.call_args_list[1].kwargs["model_name"] is None
    assert parsed["historical"]
    assert parsed["application"]


def test_fallback_lesson_is_structured_and_parseable():
    """Fallback lesson should preserve structured sections for rendering."""
    fallback = _get_fallback_lesson(
        "How does the Weber-Fechner Law explain our perception of price differences?"
    )
    parsed = parse_lesson_to_dict(fallback, "Weber-Fechner")

    assert "[KEY INSIGHT]" in fallback
    assert "[HISTORICAL]" in fallback
    assert "[APPLICATION]" in fallback
    assert parsed["key_insight"]
    assert parsed["historical"]
    assert parsed["application"]
