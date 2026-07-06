"""Identification of record-setting experimental results.

This is not physics — it encodes the paper's editorial definition of a
"record" — but it must be shared verbatim by every consumer that labels
points as records (the paper's triple-product-vs-year plot, the FEB
website's progress charts), so it lives in lib/ rather than in the
notebook script.
"""

import pandas as pd


def is_concept_record(
    df: pd.DataFrame,
    *,
    concept_col: str = "Concept Displayname",
    date_col: str = "Date",
    value_col: str = "nTtauEstar_max",
    present_year: int | None = None,
) -> pd.Series:
    """Return a boolean Series marking rows that set a record for their concept.

    A row is a record iff no row of the same concept with an earlier-or-equal
    date has a strictly greater value. Ties on value are all records. Rows
    dated after ``present_year`` (projected results) are never records; pass
    ``present_year=None`` to skip that filter.
    """

    def _row_is_record(row) -> bool:
        if present_year is not None and row[date_col].year > present_year:
            return False
        beaten_by = (
            (df[concept_col] == row[concept_col])
            & (df[date_col] <= row[date_col])
            & (df[value_col] > row[value_col])
        )
        return not beaten_by.any()

    return df.apply(_row_is_record, axis=1)
