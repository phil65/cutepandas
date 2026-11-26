"""Per-kind display formatters (Qt-free, applied to one chunk at a time)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any

from cutepandas.frames.source import Kind


if TYPE_CHECKING:
    import numpy as np


Formatter = Callable[[Any], str]


@dataclass(frozen=True, slots=True)
class FormatOptions:
    """How cell values are turned into display text."""

    float_precision: int = 6
    """Significant digits for floats (`{:.6g}`, whole numbers keep a `.0`)."""
    thousands_separator: bool = False
    """Group digits of integers and floats (`1,234,567`)."""
    null_text: str = ""
    """Text shown for missing values."""
    max_text_length: int = 200
    """Longer texts are cut and end with an ellipsis (the tooltip shows all)."""


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def formatter_for(kind: Kind, options: FormatOptions) -> Formatter:
    """Return a function that formats one non-null value of the given kind."""
    limit = options.max_text_length
    sep = "," if options.thousands_separator else ""
    match kind:
        case Kind.FLOAT:
            spec = f"{sep}.{options.float_precision}g"

            def fmt_float(value: Any) -> str:
                if not isinstance(value, float):
                    return str(value)  # complex
                if math.isnan(value):
                    return "NaN"
                text = format(value, spec)
                # keep floats recognisable: 3.0 -> "3.0", not "3"
                return text if any(c in text for c in ".eni") else text + ".0"

            return fmt_float
        case Kind.INTEGER if sep:
            return lambda value: format(value, sep)
        case Kind.INTEGER | Kind.DECIMAL:
            return str
        case Kind.STRING | Kind.CATEGORICAL | Kind.OBJECT | Kind.NESTED | Kind.BINARY:
            return lambda value: _truncate(str(value), limit)
        case _:
            return str


def format_chunk(
    values: list[Any], nulls: np.ndarray, formatter: Formatter, null_text: str
) -> list[str]:
    """Format a chunk of values; missing values become `null_text`."""
    if nulls.any():
        return [
            null_text if is_null else formatter(value)
            for value, is_null in zip(values, nulls.tolist())
        ]
    return [formatter(value) for value in values]
