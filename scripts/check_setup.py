"""
Phase 1 smoke test.

Run this after `pip install -r requirements.txt` to confirm the environment
is ready before we start ingesting the knowledge base in Phase 2.

It deliberately does NOT require a Gemini API key to pass — retrieval,
chunking, and the order tool don't need one. It just tells you clearly
what is and isn't configured yet.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this script directly (python scripts/check_setup.py) by
# putting the repository root on sys.path so `import app` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings


def check(label: str, ok: bool, detail: str = "") -> bool:
    status = "OK" if ok else "FAIL"
    line = f"[{status}] {label}"
    if detail:
        line += f" — {detail}"
    print(line)
    return ok


def main() -> int:
    results = []

    # 1. Supplied assignment content is present and untouched.
    results.append(
        check(
            "knowledge-base/ directory found",
            settings.knowledge_base_dir.is_dir(),
            str(settings.knowledge_base_dir),
        )
    )
    md_files = (
        list(settings.knowledge_base_dir.glob("*.md"))
        if settings.knowledge_base_dir.is_dir()
        else []
    )
    results.append(check("knowledge-base contains .md files", len(md_files) > 0, f"{len(md_files)} files"))

    results.append(
        check("data/orders.json found", settings.orders_path.is_file(), str(settings.orders_path))
    )

    # 2. Vector index library.
    try:
        import faiss  # noqa: F401

        results.append(check("faiss importable", True, f"version {faiss.__version__}"))
    except ImportError as exc:
        results.append(check("faiss importable", False, str(exc)))

    # 3. Local embedding library. This is the one most likely to be slow/large
    #    to install — report clearly if it's missing rather than crashing.
    try:
        import sentence_transformers  # noqa: F401

        results.append(check("sentence-transformers importable", True))
    except ImportError as exc:
        results.append(
            check(
                "sentence-transformers importable",
                False,
                f"{exc} — run: pip install -r requirements.txt",
            )
        )

    # 4. Gemini client library + key. The key is optional at this phase.
    try:
        from google import genai  # noqa: F401

        results.append(check("google-genai importable", True))
    except ImportError as exc:
        results.append(check("google-genai importable", False, str(exc)))

    if settings.has_gemini_key():
        results.append(check("GEMINI_API_KEY configured", True))
    else:
        # Not a failure yet — just informational until Phase 9.
        print("[INFO] GEMINI_API_KEY not set yet — fine for now, required starting Phase 9.")

    # 5. Writable directories for derived data.
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    results.append(check("index/ and logs/ directories writable", True))

    print()
    if all(results):
        print("Environment looks ready for Phase 2.")
        return 0
    else:
        print("Fix the FAIL items above before continuing.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
