"""Exact empirical upper-tail sample counts for decimal quantiles."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from operator import index


def upper_tail_count(n: int, quantile: float = 0.95) -> int:
    """Return ceil((1-q)*n), interpreting q as its displayed decimal value.

    The supported range is 0 <= q < 1. At q=0 the tail is the full sample;
    an empty sample has zero tail rows. Integer arithmetic prevents binary
    subtraction/ceiling from adding a row at exact boundaries (e.g. n=20,
    q=0.95). No fractional weighting at the tail boundary is applied.
    """
    if isinstance(n, bool):
        raise TypeError("sample count must be an integer, not a boolean")
    try:
        count = index(n)
    except TypeError as exc:
        raise TypeError("sample count must be an integer") from exc
    if count < 0:
        raise ValueError("sample count must be non-negative")
    try:
        q = Decimal(str(quantile))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("quantile must be a finite decimal in [0, 1)") from exc
    if not q.is_finite() or not Decimal(0) <= q < Decimal(1):
        raise ValueError("quantile must be a finite decimal in [0, 1)")
    numerator, denominator = q.as_integer_ratio()
    return ((denominator - numerator) * count + denominator - 1) // denominator
