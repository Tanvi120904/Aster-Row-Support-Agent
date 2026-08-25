from app.agent import SupportAgent


def test_agent_module_contains_system_instruction():
    import app.agent

    assert "Do not invent policy facts" in app.agent.SYSTEM_INSTRUCTION
    assert "Do not invent order information" in app.agent.SYSTEM_INSTRUCTION


def test_support_agent_class_exposes_answer_method():
    assert hasattr(SupportAgent, "answer")


def test_function_call_extractor_returns_none_without_candidates():
    response = type("Response", (), {"candidates": []})()

    assert SupportAgent._extract_function_call(response) is None


def test_function_call_extractor_returns_none_without_function_call():
    part = type(
        "Part",
        (),
        {
            "function_call": None,
        },
    )()

    content = type(
        "Content",
        (),
        {
            "parts": [part],
        },
    )()

    candidate = type(
        "Candidate",
        (),
        {
            "content": content,
        },
    )()

    response = type(
        "Response",
        (),
        {
            "candidates": [candidate],
        },
    )()

    assert SupportAgent._extract_function_call(response) is None


def test_empty_user_message_is_rejected_without_calling_gemini():
    agent = object.__new__(SupportAgent)

    try:
        agent.answer("   ")
    except ValueError as exc:
        assert "must not be empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError")