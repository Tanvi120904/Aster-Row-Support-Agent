from app.session import ConversationSession, SessionStore


def test_session_stores_user_and_assistant_messages():
    session = ConversationSession("s1")

    session.add_user("Where is ORD-1007?")
    session.add_assistant("Your order has shipped.")

    assert len(session.messages) == 2
    assert session.messages[0].role == "user"
    assert session.messages[1].role == "assistant"


def test_session_context_preserves_order():
    session = ConversationSession("s1")

    session.add_user("Where is ORD-1007?")
    session.add_assistant("It has shipped.")
    session.add_user("When will it arrive?")

    context = session.recent_context()

    assert "Where is ORD-1007?" in context
    assert "It has shipped." in context
    assert "When will it arrive?" in context


def test_session_is_bounded():
    session = ConversationSession(
        "s1",
        max_messages=3,
    )

    session.add_user("one")
    session.add_assistant("two")
    session.add_user("three")
    session.add_assistant("four")

    assert len(session.messages) == 3
    assert session.messages[0].content == "two"


def test_session_store_reuses_same_session():
    store = SessionStore()

    first = store.get("abc")
    second = store.get("abc")

    assert first is second


def test_session_store_keeps_sessions_separate():
    store = SessionStore()

    first = store.get("abc")
    second = store.get("xyz")

    first.add_user("hello")

    assert second.messages == []


def test_session_store_can_clear_session():
    store = SessionStore()

    session = store.get("abc")
    session.add_user("hello")

    store.clear("abc")

    new_session = store.get("abc")

    assert new_session.messages == []