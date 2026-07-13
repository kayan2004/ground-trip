from app.core.llm_allowlist import BYOK_ALLOWLIST, is_allowlisted


def test_is_allowlisted_true_for_known_pairs():
    for entry in BYOK_ALLOWLIST:
        assert is_allowlisted(entry.provider, entry.model)


def test_is_allowlisted_false_for_unknown_pair():
    assert not is_allowlisted("gemini", "gemini-3.1-pro-preview")
    assert not is_allowlisted("openai", "gpt-5.4")
    assert not is_allowlisted("made-up-provider", "made-up-model")
