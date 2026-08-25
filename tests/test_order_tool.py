from pathlib import Path

import pytest

from app.order_tool import (
    InvalidOrderIdError,
    OrderNotFoundError,
    lookup_order,
    normalize_order_id,
    safe_lookup_order,
)
from app.config import settings


ORDERS_PATH = Path(settings.orders_path)


def test_normalize_order_id_accepts_lowercase_and_surrounding_punctuation():
    assert normalize_order_id(" ord-1007. ") == "ORD-1007"


def test_normalize_order_id_rejects_substantially_different_identifier():
    with pytest.raises(InvalidOrderIdError):
        normalize_order_id("ORD-9999X")


def test_lookup_returns_requested_customer_safe_fields():
    result = lookup_order(
        "ord-1003",
        fields=[
            "status",
            "carrier",
            "tracking_number",
            "estimated_delivery",
        ],
        data_path=ORDERS_PATH,
    )

    assert result["found"] is True

    order = result["order"]

    assert order["order_id"] == "ORD-1003"
    assert order["status"] == "shipped"
    assert order["carrier"] == "USPS"
    assert order["tracking_number"] == "94001118995600001003"
    assert order["estimated_delivery"] == "2026-08-18"


def test_lookup_never_exposes_customer_private_fields():
    result = lookup_order(
        "ORD-1007",
        fields=[
            "order_id",
            "status",
            "customer.name",
            "customer.email",
            "customer.shipping_address",
            "internal",
            "internal.risk_score",
            "internal.warehouse_note",
        ],
        data_path=ORDERS_PATH,
    )

    order = result["order"]

    serialized = str(order)

    assert "Ava Morgan" not in serialized
    assert "ava.morgan@example.test" not in serialized
    assert "Toronto" not in serialized
    assert "82" not in serialized
    assert "Never expose this note" not in serialized

    assert "customer" not in order
    assert "internal" not in order


def test_cancelled_order_does_not_expose_stale_delivery_fields():
    result = lookup_order(
        "ORD-1004",
        fields=[
            "status",
            "carrier",
            "tracking_number",
            "estimated_delivery",
            "customer_safe_message",
        ],
        data_path=ORDERS_PATH,
    )

    order = result["order"]

    assert order["status"] == "cancelled"

    assert "carrier" not in order
    assert "tracking_number" not in order
    assert "estimated_delivery" not in order

    assert "will not be shipped" in order["customer_safe_message"]


def test_shipped_order_with_missing_eta_keeps_eta_unavailable():
    result = lookup_order(
        "ORD-1011",
        fields=[
            "status",
            "carrier",
            "tracking_number",
            "estimated_delivery",
            "customer_safe_message",
        ],
        data_path=ORDERS_PATH,
    )

    order = result["order"]

    assert order["status"] == "shipped"
    assert order["carrier"] == "Canada Post"
    assert order["tracking_number"] == "AR1011CA00001"
    assert order["estimated_delivery"] is None

    assert "estimate is not currently available" in order["customer_safe_message"]


def test_exception_order_requires_handoff():
    result = lookup_order(
        "ORD-1010",
        fields=[
            "status",
            "customer_safe_message",
        ],
        data_path=ORDERS_PATH,
    )

    order = result["order"]

    assert order["status"] == "exception"
    assert order["handoff_required"] is True
    assert "support review" in order["customer_safe_message"]


def test_returned_order_does_not_look_like_an_active_delivery():
    result = lookup_order(
        "ORD-1008",
        fields=[
            "status",
            "carrier",
            "tracking_number",
            "estimated_delivery",
            "customer_safe_message",
        ],
        data_path=ORDERS_PATH,
    )

    order = result["order"]

    assert order["status"] == "returned"

    assert "carrier" not in order
    assert "tracking_number" not in order
    assert "estimated_delivery" not in order


def test_items_are_reduced_to_customer_safe_fields():
    result = lookup_order(
        "ORD-1007",
        fields=["items"],
        data_path=ORDERS_PATH,
    )

    items = result["order"]["items"]

    assert len(items) == 1

    assert items[0] == {
        "name": "Atlas Weekender",
        "quantity": 1,
        "final_sale": False,
    }

    assert "sku" not in items[0]


def test_unknown_order_returns_not_found():
    with pytest.raises(OrderNotFoundError):
        lookup_order(
            "ORD-9999",
            data_path=ORDERS_PATH,
        )


def test_safe_lookup_returns_structured_not_found_result():
    result = safe_lookup_order(
        "ORD-9999",
        data_path=ORDERS_PATH,
    )

    assert result["found"] is False
    assert result["error_type"] == "order_not_found"


def test_safe_lookup_returns_structured_invalid_id_result():
    result = safe_lookup_order(
        "this-is-not-an-order",
        data_path=ORDERS_PATH,
    )

    assert result["found"] is False
    assert result["error_type"] == "invalid_order_id"


def test_tool_output_contains_no_internal_section():
    result = lookup_order(
        "ORD-1005",
        data_path=ORDERS_PATH,
    )

    assert "internal" not in result["order"]
    assert "risk_score" not in result["order"]
    assert "warehouse_note" not in result["order"]
    assert "support_tags" not in result["order"]

def test_private_field_requests_are_ignored_even_with_nested_paths():
    result = lookup_order(
        "ORD-1007",
        fields=[
            "customer",
            "customer.name",
            "customer.email",
            "customer.shipping_address",
            "internal",
            "internal.risk_score",
            "internal.warehouse_note",
            "internal.support_tags",
        ],
        data_path=ORDERS_PATH,
    )

    order = result["order"]

    assert "customer" not in order
    assert "internal" not in order
    assert "risk_score" not in order
    assert "warehouse_note" not in order
    assert "support_tags" not in order


def test_order_id_near_miss_is_not_guessed():
    result = safe_lookup_order(
        "ORD-1007X",
        data_path=ORDERS_PATH,
    )

    assert result["found"] is False
    assert result["error_type"] == "invalid_order_id"


def test_order_lookup_does_not_return_raw_customer_object():
    result = lookup_order(
        "ORD-1001",
        data_path=ORDERS_PATH,
    )

    order = result["order"]

    assert "customer" not in order

    result_text = str(result)

    assert "Maya Reed" not in result_text
    assert "maya.reed@example.test" not in result_text
    assert "18 Cedar Lane" not in result_text


def test_order_lookup_does_not_return_raw_internal_object():
    result = lookup_order(
        "ORD-1005",
        data_path=ORDERS_PATH,
    )

    order = result["order"]

    assert "internal" not in order

    result_text = str(result)

    assert "risk_score" not in result_text
    assert "warehouse_note" not in result_text
    assert "support-tags" not in result_text
    assert "$100 coupon" not in result_text


def test_cancelled_order_status_takes_precedence_over_stale_logistics():
    result = lookup_order(
        "ORD-1004",
        data_path=ORDERS_PATH,
    )

    order = result["order"]

    assert order["status"] == "cancelled"
    assert order["customer_safe_message"] == (
        "The order was cancelled and will not be shipped."
    )

    assert "carrier" not in order
    assert "tracking_number" not in order
    assert "estimated_delivery" not in order


def test_returned_order_status_takes_precedence_over_old_delivery_fields():
    result = lookup_order(
        "ORD-1008",
        data_path=ORDERS_PATH,
    )

    order = result["order"]

    assert order["status"] == "returned"
    assert "carrier" not in order
    assert "tracking_number" not in order
    assert "estimated_delivery" not in order


def test_shipped_order_without_eta_is_not_given_a_computed_eta():
    result = lookup_order(
        "ORD-1011",
        data_path=ORDERS_PATH,
    )

    order = result["order"]

    assert order["status"] == "shipped"
    assert order["estimated_delivery"] is None
    assert "estimate is not currently available" in (
        order["customer_safe_message"]
    )


def test_exception_order_is_explicitly_marked_for_handoff():
    result = lookup_order(
        "ORD-1010",
        data_path=ORDERS_PATH,
    )

    order = result["order"]

    assert order["status"] == "exception"
    assert order["handoff_required"] is True


def test_tool_never_claims_an_action_was_performed():
    result = lookup_order(
        "ORD-1007",
        data_path=ORDERS_PATH,
    )

    result_text = str(result).lower()

    forbidden_action_claims = (
        "cancelled successfully",
        "refund completed",
        "replacement created",
        "address changed",
        "escalation created",
    )

    for phrase in forbidden_action_claims:
        assert phrase not in result_text