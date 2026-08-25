"""
Deterministic order lookup tool.

This module is the application's security boundary around orders.json.

The model should NEVER receive the raw order record. The tool loads the
requested order, validates the ID, and returns only an explicit customer-safe
allowlist of fields.

No LLM, network call, or prompt is involved.
"""

from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any


# Fields explicitly allowed by the assignment's data dictionary.
SAFE_FIELDS = {
    "order_id",
    "membership_tier",
    "items",
    "placed_at",
    "status",
    "status_updated_at",
    "shipped_at",
    "delivered_at",
    "carrier",
    "tracking_number",
    "estimated_delivery",
    "customer_safe_message",
}
# These names are explicitly forbidden even if a caller tries to request them.
#
# The application never permits arbitrary nested field access.
FORBIDDEN_FIELD_PREFIXES = (
    "customer",
    "internal",
)

# Fields that can become misleading when an order has been cancelled or
# returned. We omit them rather than risk presenting stale operational data.
STALE_LOGISTICS_FIELDS = {
    "carrier",
    "tracking_number",
    "estimated_delivery",
}


class OrderLookupError(Exception):
    """Base error for safe, deterministic order lookup failures."""


class InvalidOrderIdError(OrderLookupError):
    """Raised when the supplied order ID has an invalid shape."""


class OrderNotFoundError(OrderLookupError):
    """Raised when the order ID is valid but does not exist."""


def _load_dataset(path: Path) -> dict[str, Any]:
    """Load the mock order dataset from disk."""

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as exc:
        raise OrderLookupError(f"Order dataset not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise OrderLookupError(f"Order dataset is invalid JSON: {path}") from exc

    if not isinstance(data, dict) or not isinstance(data.get("orders"), list):
        raise OrderLookupError("Order dataset does not contain an orders list")

    return data


def normalize_order_id(order_id: str) -> str:
    """
    Normalize harmless input differences.

    Accepted examples:
      ORD-1007
      ord-1007
      " ORD-1007 "
      "ORD-1007."

    We do NOT guess substantially different IDs.
    """

    if not isinstance(order_id, str):
        raise InvalidOrderIdError("order_id must be a string")

    normalized = order_id.strip().upper()

    # Remove ordinary punctuation surrounding the identifier.
    normalized = normalized.strip(" \t\r\n.,!?;:()[]{}<>\"'`")

    if not re.fullmatch(r"ORD-\d{4}", normalized):
        raise InvalidOrderIdError(
            "Invalid order ID format. Expected something like ORD-1007."
        )

    return normalized


def _safe_item(item: dict[str, Any]) -> dict[str, Any]:
    """Return only customer-safe item fields."""

    return {
        "name": item.get("name"),
        "quantity": item.get("quantity"),
        "final_sale": item.get("final_sale"),
    }


def _safe_order(
    order: dict[str, Any],
    requested_fields: set[str] | None,
) -> dict[str, Any]:
    """
    Apply the customer-safe field allowlist.

    The caller can request a subset of safe fields, but requesting unsafe
    fields never expands what the tool can expose.
    """

    allowed = SAFE_FIELDS if requested_fields is None else (
        requested_fields & SAFE_FIELDS
    )

    # Never allow explicitly private/nested fields through even if a future
    # SAFE_FIELDS change accidentally includes a prefix-like name.
    if requested_fields:
        allowed = {
            field
            for field in allowed
            if not _is_forbidden_requested_field(field)
        }

    # These fields are essential for interpreting any result safely.
    allowed |= {
        "order_id",
        "status",
        "customer_safe_message",
    }

    result: dict[str, Any] = {}

    status = order.get("status")

    for field in sorted(allowed):
        # Defense in depth.
        if _is_forbidden_requested_field(field):
            continue

        # Items require nested allowlisting.
        if field == "items":
            result["items"] = [
                _safe_item(item)
                for item in order.get("items", [])
                if isinstance(item, dict)
            ]
            continue

        # Cancelled and returned orders may contain stale logistics fields.
        if status in {"cancelled", "returned"}:
            if field in STALE_LOGISTICS_FIELDS:
                continue

        result[field] = order.get(field)

    result["handoff_required"] = status == "exception"

    return result

def _is_forbidden_requested_field(field: str) -> bool:
    """
    Return True when a caller attempts to request private/internal data.

    We reject the request semantically rather than merely ignoring a few
    currently-known field names. This protects newly-added nested private
    fields too.
    """

    normalized = field.strip().lower()

    return normalized == "customer" or normalized.startswith(
        "customer."
    ) or normalized == "internal" or normalized.startswith(
        "internal."
    )


def lookup_order(
    order_id: str,
    *,
    fields: list[str] | None = None,
    data_path: Path | None = None,
) -> dict[str, Any]:
    """
    Look up one order and return a customer-safe result.

    Args:
        order_id:
            User-supplied order identifier.
        fields:
            Optional subset of customer-safe fields requested by the caller.
            Unsafe fields are ignored by the allowlist.
        data_path:
            Optional path override used for tests.

    Returns:
        A dictionary containing either a safe order result or a structured
        error.
    """

    normalized_id = normalize_order_id(order_id)

    if data_path is None:
        data_path = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "orders.json"
        )

    dataset = _load_dataset(data_path)

    matching_order = next(
        (
            order
            for order in dataset["orders"]
            if isinstance(order, dict)
            and order.get("order_id") == normalized_id
        ),
        None,
    )

    if matching_order is None:
        raise OrderNotFoundError(
            f"No order found for {normalized_id}."
        )

    requested_fields: set[str] | None = None

    if fields is not None:
        requested_fields = set(fields)

    safe_data = _safe_order(
        matching_order,
        requested_fields,
    )

    return {
        "found": True,
        "order": safe_data,
    }


def safe_lookup_order(
    order_id: str,
    *,
    fields: list[str] | None = None,
    data_path: Path | None = None,
) -> dict[str, Any]:
    """
    Exception-safe wrapper intended for future agent/tool calling.

    Instead of exposing Python exceptions to the model layer, return a
    structured success/error object.
    """

    try:
        return lookup_order(
            order_id,
            fields=fields,
            data_path=data_path,
        )
    except InvalidOrderIdError as exc:
        return {
            "found": False,
            "error_type": "invalid_order_id",
            "message": str(exc),
        }
    except OrderNotFoundError as exc:
        return {
            "found": False,
            "error_type": "order_not_found",
            "message": str(exc),
        }
    except OrderLookupError as exc:
        return {
            "found": False,
            "error_type": "lookup_error",
            "message": str(exc),
        }