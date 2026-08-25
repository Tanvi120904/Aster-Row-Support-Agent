from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Message:
    role: str
    content: str


@dataclass
class ConversationSession:
    session_id: str
    max_messages: int = 8
    messages: list[Message] = field(default_factory=list)

    def add_user(self, content: str) -> None:
        self._add("user", content)

    def add_assistant(self, content: str) -> None:
        self._add("assistant", content)

    def _add(self, role: str, content: str) -> None:
        if not content.strip():
            return

        self.messages.append(
            Message(
                role=role,
                content=content,
            )
        )

        self.messages = self.messages[-self.max_messages :]

    def recent_context(self) -> str:
        if not self.messages:
            return ""

        return "\n".join(
            f"{message.role.upper()}: {message.content}"
            for message in self.messages
        )


class SessionStore:
    def __init__(self, max_messages: int = 8):
        self._sessions: dict[str, ConversationSession] = {}
        self._max_messages = max_messages

    def get(self, session_id: str) -> ConversationSession:
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationSession(
                session_id=session_id,
                max_messages=self._max_messages,
            )

        return self._sessions[session_id]

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)