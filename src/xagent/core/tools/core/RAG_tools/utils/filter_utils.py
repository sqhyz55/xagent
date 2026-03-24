"""Filter parsing utilities for backend-agnostic filter expressions.

This module provides utilities to convert API-facing filter dictionaries into
abstract filter expressions that can be translated to backend-specific syntax.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..storage.contracts import FilterCondition, FilterExpression, FilterOperator


def parse_legacy_filters(filters: Optional[Dict[str, Any]]) -> Optional[FilterExpression]:
    """Convert Dict-based filters to an abstract FilterExpression.

    Supported input formats:
    - Simple equality:
      {"field": "value"}
    - Operator form:
      {"field": {"operator": "gte", "value": 5}}

    Multiple fields are combined as an AND expression (tuple convention).

    Args:
        filters: Filter dictionary from API layer.

    Returns:
        Parsed FilterExpression, or None if filters is None/empty.

    Raises:
        ValueError: If an unsupported operator is provided.
    """
    if not filters:
        return None

    op_map: Dict[str, FilterOperator] = {
        "eq": FilterOperator.EQ,
        "ne": FilterOperator.NE,
        "gt": FilterOperator.GT,
        "gte": FilterOperator.GTE,
        "lt": FilterOperator.LT,
        "lte": FilterOperator.LTE,
        "in": FilterOperator.IN,
        "contains": FilterOperator.CONTAINS,
    }

    conditions: list[FilterCondition] = []
    for field, spec in filters.items():
        if isinstance(spec, dict) and "operator" in spec and "value" in spec:
            op_str = str(spec["operator"]).lower()
            if op_str not in op_map:
                raise ValueError(
                    f"Unknown filter operator: {op_str}. Supported operators: {sorted(op_map.keys())}"
                )
            conditions.append(
                FilterCondition(field=field, operator=op_map[op_str], value=spec["value"])
            )
        else:
            conditions.append(
                FilterCondition(field=field, operator=FilterOperator.EQ, value=spec)
            )

    if len(conditions) == 1:
        return conditions[0]
    return tuple(conditions)
