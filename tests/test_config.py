"""
Phase 1 tests: the config module points at real files, without needing
any API key or network access. These should pass on any machine that has
just cloned the repo and run `pip install -r requirements.txt`.
"""

from app.config import settings


def test_knowledge_base_dir_exists():
    assert settings.knowledge_base_dir.is_dir()


def test_knowledge_base_has_markdown_files():
    md_files = list(settings.knowledge_base_dir.glob("*.md"))
    # The assignment ships 14 policy/product documents.
    assert len(md_files) == 14


def test_orders_file_exists():
    assert settings.orders_path.is_file()


def test_no_gemini_key_required_for_config_to_load():
    # Config must be constructible even with no API key configured —
    # only Phase 9+ (actual generation) should require one.
    assert isinstance(settings.has_gemini_key(), bool)
