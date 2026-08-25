from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from app.agent import create_agent
from app.session import SessionStore


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="../templates",
    )

    agent = create_agent()
    sessions = SessionStore()

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/api/chat")
    def chat():
        payload = request.get_json(silent=True) or {}

        message = payload.get("message", "")
        session_id = payload.get("session_id")

        if not isinstance(message, str) or not message.strip():
            return jsonify(
                {
                    "error": "message is required",
                }
            ), 400

        if not isinstance(session_id, str) or not session_id.strip():
            return jsonify(
                {
                    "error": "session_id is required",
                }
            ), 400

        try:
            session = sessions.get(session_id)

            response = agent.answer(
                message,
                session=session,
            )

            return jsonify(
                {
                    "answer": response.text,
                    "sources": response.sources,
                    "tool": response.tool_name,
                    "tool_arguments": response.tool_arguments,
                    "handoff": response.handoff,
                }
            )

        except Exception as exc:
            app.logger.exception("Agent request failed")

            if getattr(exc, "code", None) == 429:
                return jsonify(
                    {
                        "error": (
                            "The AI service is temporarily unavailable because "
                            "the Gemini API quota has been reached. Please try "
                            "again later."
                        )
                    }
                ), 429

            return jsonify(
                {
                    "error": "The agent could not complete the request.",
                }
            ), 500

    return app