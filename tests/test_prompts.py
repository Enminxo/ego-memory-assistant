import pytest

from egomem.prompts import PROMPT_VARIANTS, describer_prompt


def test_default_prompt_keeps_the_example_objects():
    prompt = describer_prompt("00:00", "00:15", 8)
    assert "(keys, wallet, phone, glasses, bags)" in prompt
    assert prompt == describer_prompt("00:00", "00:15", 8, variant="default")


def test_no_examples_variant_drops_the_list_and_keeps_the_rest():
    default = describer_prompt("00:00", "00:15", 8, max_words=60)
    variant = describer_prompt("00:00", "00:15", 8, max_words=60, variant="no_examples")
    for word in ("keys", "wallet", "glasses", "bags"):
        assert word not in variant
    assert "Objects I pick up or put down, and exactly where I leave them." in variant
    assert "do not list things I did not do" in variant
    # everything else is shared: frames, time span, priorities 2-5, questions, word limit
    for line in default.splitlines():
        if "Personal objects" in line or line.startswith("At most"):
            continue
        assert line in variant
    assert "At most 60 words." in variant


def test_unknown_variant_is_rejected():
    assert set(PROMPT_VARIANTS) == {"default", "no_examples"}
    with pytest.raises(ValueError):
        describer_prompt("00:00", "00:15", 8, variant="nope")
