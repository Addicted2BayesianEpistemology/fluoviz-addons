"""Small checks of the numbers of a request (server side)."""
from __future__ import annotations

from fastapi import HTTPException


def num_query(v, what: str, lo: float, hi: float) -> float:
    """``v`` as a number between ``lo`` and ``hi``, or a plain 400."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise HTTPException(400, f"'{what}' must be a number.") from None
    if not lo <= x <= hi:
        raise HTTPException(400, f"'{what}' must be between {lo:g} and {hi:g}.")
    return x
