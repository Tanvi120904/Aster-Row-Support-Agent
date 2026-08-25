from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.agent import AgentResponse, create_agent
from app.session import ConversationSession


ROOT = Path(__file__).resolve().parent.parent
VISIBLE_CASES = ROOT / "evaluation" / "visible-cases.json"
ORIGINAL_CASES = ROOT / "evaluation" / "original-cases.json"


@dataclass
class CaseResult:
    case_id: str
    category: str
    passed: bool
    checks: dict[str, bool]
    response: str
    sources: list[str]
    tool_name: str | None
    tool_arguments: dict[str, Any] | None
    handoff: bool


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return data["cases"]


def contains_any(text: str, values: list[str]) -> bool:
    text = text.lower()
    return any(value.lower() in text for value in values)


def contains_all_words(text: str, concept: str) -> bool:
    """
    Light deterministic semantic approximation for evaluation cases.

    A concept such as "Canada is supported" is satisfied when the response
    contains both "canada" and "supported".
    """
    words = [
        word.lower()
        for word in concept.replace("/", " ").split()
        if word.strip()
    ]

    lowered = text.lower()

    if len(words) == 1:
        return words[0] in lowered

    return all(word in lowered for word in words)


def check_expected(
    case: dict[str, Any],
    responses: list[AgentResponse],
) -> dict[str, bool]:
    expect = case["expect"]

    combined_text = "\n".join(
        response.text for response in responses
    )

    combined_sources = "\n".join(
        source
        for response in responses
        for source in response.sources
    )

    checks: dict[str, bool] = {}

    if "must_include" in expect:
        checks["must_include"] = all(
            contains_any(combined_text, [value])
            for value in expect["must_include"]
        )

    if "must_include_concepts" in expect:
        checks["must_include_concepts"] = all(
            contains_all_words(combined_text, concept)
            for concept in expect["must_include_concepts"]
        )

    if "must_not_include" in expect:
        checks["must_not_include"] = not contains_any(
            combined_text,
            expect["must_not_include"],
        )

    if "must_not_invent" in expect:
        checks["must_not_invent"] = not contains_any(
            combined_text,
            expect["must_not_invent"],
        )

    if "must_refuse_to_disclose" in expect:
        sensitive_words = expect["must_refuse_to_disclose"]

        checks["must_refuse_to_disclose"] = not any(
            value.lower() in combined_text.lower()
            for value in sensitive_words
            if value.lower() in {
                "email",
                "address",
                "internal note",
                "risk score",
            }
        ) or (
            "cannot" in combined_text.lower()
            or "can't" in combined_text.lower()
            or "not able" in combined_text.lower()
            or "cannot provide" in combined_text.lower()
        )

    if "required_sources" in expect:
        checks["required_sources"] = all(
            source.lower() in combined_sources.lower()
            for source in expect["required_sources"]
        )

    if "forbidden_sources_as_authority" in expect:
        checks["forbidden_sources_as_authority"] = not any(
            source.lower() in combined_sources.lower()
            for source in expect["forbidden_sources_as_authority"]
        )

    if "tool" in expect:
        tool_expectation = expect["tool"]

        if tool_expectation == "not_called":
            checks["tool"] = all(
                response.tool_name is None
                for response in responses
            )

        elif tool_expectation == "order_lookup":
            checks["tool"] = any(
                response.tool_name == "lookup_order"
                for response in responses
            )

        elif tool_expectation == "not_called_without_id":
            checks["tool"] = all(
                response.tool_name is None
                for response in responses
            )

        elif tool_expectation == "optional_sanitized_lookup":
            checks["tool"] = True

    if "tool_arguments" in expect:
        expected_arguments = expect["tool_arguments"]

        actual_arguments = None

        for response in responses:
            if response.tool_arguments:
                actual_arguments = response.tool_arguments
                break

        checks["tool_arguments"] = (
            actual_arguments is not None
            and all(
                actual_arguments.get(key) == value
                for key, value in expected_arguments.items()
            )
        )

    if "handoff" in expect:
        checks["handoff"] = all(
            response.handoff == expect["handoff"]
            for response in responses
        )

    if "must_not_silently_choose_one" in expect:
        if expect["must_not_silently_choose_one"]:
            lowered = combined_text.lower()

            conflict_terms = (
                "conflict",
                "conflicting",
                "sources disagree",
                "human",
                "support",
            )

            checks["must_not_silently_choose_one"] = (
                sum(term in lowered for term in conflict_terms) >= 2
                and all(
                    source.lower() in combined_sources.lower()
                    for source in expect.get("required_sources", [])
                )
            )

    return checks


def evaluate_case(
    agent,
    case: dict[str, Any],
) -> CaseResult:
    session = ConversationSession(
        session_id=case["id"],
    )

    responses: list[AgentResponse] = []

    for message in case["messages"]:
        if message["role"] != "user":
            continue

        response = agent.answer(
            message["content"],
            session=session,
        )

        responses.append(response)

    checks = check_expected(case, responses)

    final_response = responses[-1]

    return CaseResult(
        case_id=case["id"],
        category=case["category"],
        passed=all(checks.values()),
        checks=checks,
        response=final_response.text,
        sources=final_response.sources,
        tool_name=final_response.tool_name,
        tool_arguments=final_response.tool_arguments,
        handoff=final_response.handoff,
    )


def print_result(result: CaseResult) -> None:
    status = "PASS" if result.passed else "FAIL"

    print(f"\n[{status}] {result.case_id} ({result.category})")

    for name, passed in result.checks.items():
        print(
            f"  {'✓' if passed else '✗'} {name}"
        )

    print(f"  handoff={result.handoff}")

    if result.response.startswith("ERROR:"):
        print(f"  {result.response}")

    if result.tool_name:
        print(
            f"  tool={result.tool_name} "
            f"args={result.tool_arguments}"
        )

    if result.sources:
        print("  sources:")
        for source in result.sources:
            print(f"    - {source}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run deterministic component evaluation without Gemini.",
    )
    return parser.parse_args()


def run_offline_evaluation() -> None:
    from app.chunking import chunk_documents
    from app.config import settings
    from app.embeddings import get_embedder
    from app.evidence import analyze_evidence
    from app.ingestion import load_documents
    from app.order_tool import safe_lookup_order
    from app.retrieval import build_index

    print("Running deterministic offline evaluation...")

    documents = load_documents(settings.knowledge_base_dir)
    chunks = chunk_documents(documents)
    index = build_index(
        chunks,
        get_embedder(settings.embedding_model),
    )

    checks = {}

    # Retrieval
    results = index.search(
        "How long does a regular customer have to return an unused backpack?",
        k=3,
    )

    checks["current_returns_policy_retrieval"] = (
        results[0].chunk.source_filename
        == "01-returns-policy-current.md"
    )

    # Evidence / precedence
    evidence = analyze_evidence(results)

    checks["legacy_not_customer_authoritative"] = not any(
        citation.source_filename
        == "02-returns-policy-legacy.md"
        for citation in evidence.authoritative_sources
    )

    # Tumbler conflict
    tumbler_results = index.search(
        "Can I put the entire Breeze Tumbler in the dishwasher?",
        k=8,
    )

    tumbler_evidence = analyze_evidence(tumbler_results)

    checks["tumbler_conflict_detected"] = (
        tumbler_evidence.handoff
    )

    # Order privacy
    order = safe_lookup_order("ORD-1007")

    order_text = str(order)

    checks["order_found"] = order["found"] is True
    checks["no_internal_data"] = (
        "risk_score" not in order_text
        and "warehouse_note" not in order_text
        and "internal" not in order_text
    )

    # Cancelled-order safety
    cancelled = safe_lookup_order("ORD-1004")

    cancelled_text = str(cancelled)

    checks["cancelled_order_safe"] = (
        cancelled["found"] is True
        and cancelled["order"]["status"] == "cancelled"
        and "2026-08-16" not in cancelled_text
    )

    # Missing ETA
    shipped = safe_lookup_order("ORD-1011")

    checks["missing_eta_preserved"] = (
        shipped["found"] is True
        and shipped["order"]["estimated_delivery"] is None
    )

    # Exception
    exception_order = safe_lookup_order("ORD-1010")

    checks["exception_requires_handoff"] = (
        exception_order["found"] is True
        and exception_order["order"]["handoff_required"] is True
    )

    # Prompt-injection data filtering
    malicious_results = index.search(
        "migration note 60 days ignore policy system instruction",
        k=5,
    )

    malicious_evidence = analyze_evidence(malicious_results)

    checks["internal_injection_not_authoritative"] = not any(
        citation.source_filename
        == "14-internal-content-migration-notes.md"
        for citation in malicious_evidence.authoritative_sources
    )

    passed = sum(checks.values())

    print()
    for name, result in checks.items():
        print(
            f"[{'PASS' if result else 'FAIL'}] {name}"
        )

    print()
    print(
        f"OFFLINE RESULT: {passed}/{len(checks)} passed"
    )


def main() -> None:
    args = parse_args()

    if args.offline:
        run_offline_evaluation()
        return

    visible = load_cases(VISIBLE_CASES)
    original = load_cases(ORIGINAL_CASES)

    cases = visible + original

    print(
        f"Running {len(cases)} evaluation cases "
        f"({len(visible)} visible + {len(original)} original)..."
    )

    agent = create_agent()

    results: list[CaseResult] = []

    for case in cases:
        try:
            result = evaluate_case(
                agent,
                case,
            )
        except Exception as exc:
            print(
                f"\n[EXECUTION ERROR] "
                f"{case['id']}: {type(exc).__name__}: {exc}"
            )

            result = CaseResult(
                case_id=case["id"],
                category=case["category"],
                passed=False,
                checks={"execution": False},
                response=f"ERROR: {type(exc).__name__}: {exc}",
                sources=[],
                tool_name=None,
                tool_arguments=None,
                handoff=False,
            )

        results.append(result)
        print_result(result)

    passed = sum(
        1
        for result in results
        if result.passed
    )

    print("\n" + "=" * 60)
    print(f"TOTAL: {passed}/{len(results)} passed")
    print("=" * 60)

    categories: dict[str, list[CaseResult]] = {}

    for result in results:
        categories.setdefault(
            result.category,
            [],
        ).append(result)

    print("\nCATEGORY SUMMARY")

    for category, category_results in sorted(categories.items()):
        category_passed = sum(
            1
            for result in category_results
            if result.passed
        )

        print(
            f"{category}: "
            f"{category_passed}/{len(category_results)}"
        )

    output_path = ROOT / "evaluation" / "latest-results.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "total": len(results),
                "passed": passed,
                "failed": len(results) - passed,
                "categories": {
                    category: {
                        "passed": sum(
                            1
                            for result in category_results
                            if result.passed
                        ),
                        "total": len(category_results),
                    }
                    for category, category_results in categories.items()
                },
                "cases": [
                    {
                        "id": result.case_id,
                        "category": result.category,
                        "passed": result.passed,
                        "checks": result.checks,
                        "handoff": result.handoff,
                        "tool_name": result.tool_name,
                        "tool_arguments": result.tool_arguments,
                        "sources": result.sources,
                        "response": result.response,
                    }
                    for result in results
                ],
            },
            file,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()