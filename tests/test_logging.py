from pathlib import Path

from app.logging_utils import log_event


def test_log_event_writes_json_lines(tmp_path: Path):
    path = tmp_path / "agent.jsonl"

    log_event(
        {
            "event": "test",
            "value": 123,
        },
        path=path,
    )

    content = path.read_text(encoding="utf-8").strip()

    assert '"event": "test"' in content
    assert '"value": 123' in content
