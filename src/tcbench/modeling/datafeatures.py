from __future__ import annotations

import polars as pl
import numpy as np

from typing import Tuple, List, Dict, Iterable
from numpy.typing import NDArray

from tcbench.modeling.columns import (
    COL_APP,
    COL_BYTES,
    COL_PACKETS,
    COL_ROW_ID,
)
from tcbench.modeling.enums import (
    MODELING_FEATURE
)

DEFAULT_EXTRA_COLUMNS = (
    COL_BYTES,
    COL_PACKETS,
    COL_ROW_ID,
)


def packet_series_colnames(df:pl.DataFrame) -> List[str]:
    return [
        col
        for col in df.columns
        if col.startswith("pkts_") and df.schema.get(col).is_nested()
    ]


def expr_packet_series_pad(
    colname: str,
    expected_len: int,
    return_dtype: pl.DataType,
    pad_value: int = 0,
) -> pl.Expr:
    return (
        pl.when(
            pl.col(colname).list.len() < expected_len
        )
        .then(
            pl.col(colname).map_elements(
                function=lambda data: pl.Series(
                    np.pad(
                        data, 
                        pad_width=(0, expected_len-min(len(data), expected_len)), 
                        constant_values=pad_value,
                    )
                ),
                return_dtype=return_dtype
            )
        )
    )

def expr_packet_series_cut(
    colname: str,
    expected_len: int,
) -> pl.Expr:
    return pl.col(colname).list.head(expected_len)


def packet_series_pad(
    df: pl.DataFrame,
    expected_len: int,
    pad_value: int = 0,
) -> pl.DataFrame:
    cols = packet_series_colnames(df)
    return df.with_columns(
        **{
            col: expr_packet_series_pad(
                col,
                expected_len,
                df.schema.get(col),
                pad_value,
            )
            for col in cols
        },
        is_padded=(
            pl.col(cols[0]).list.len() < expected_len
        )
    )

def packet_series_cut(
    df: pl.DataFrame,
    expected_len: int | Dict[str, int],
) -> pl.DataFrame:
    cols = packet_series_colnames(df)
    return df.with_columns(
        **{
            col: expr_packet_series_cut(col, expected_len)
            for col in cols
        },
        is_cut=(
            pl.col(cols[0]).list.len() < expected_len
        )
    )


def features_dataprep(
    df: pl.DataFrame,
    features: Iterable[MODELING_FEATURE],
    *,
    series_len: int = -1, 
    series_pad: int = -1,
    y_colname: str = COL_APP,
    extra_colnames: Iterable[str] | None = DEFAULT_EXTRA_COLUMNS,
) -> Tuple[NDArray, NDArray, pl.DataFrame]:

    if extra_colnames is None:
        extra_colnames = []

    # converting enumeration to string
    _features  = list(map(str, features))

    cols_series = [
        col
        for col in packet_series_colnames(df)
        if col in _features
    ]

    df_feat = df.select(
        *_features,
        y_colname,
        *extra_colnames
    )

    if series_pad > 0:
        # enforce padding (where needed)
        df_feat = df_feat.with_columns(**{
              col: expr_packet_series_pad(
                  col,
                  series_len,
                  df_feat.schema.get(col),
                  series_pad,
              )
              for col in cols_series
        })

    def _struct_field_name(col, idx):
        return f"{col}_{idx}"

    if series_len > 0:
        df_feat = (df_feat
            # discard rows if series are too short
            .filter(
                pl.col(cols_series[0]).list.len() >= series_len
            )
            .with_columns(**{
                col: (
                    # cut series ...and packet them into struct
                    expr_packet_series_cut(col, series_len)
                    .list
                    .to_struct(
                        fields=[
                            f"{col}_{idx}"
                            for idx in range(1, series_len+1)
                        ]
                    )
                )
                for col in cols_series
            })
            # unnest structs (so each series value is a separate column)
            .unnest(*cols_series)
        )

    y = df_feat[y_colname].to_numpy()
    X = df_feat.drop(y_colname, *extra_colnames).to_numpy()
    return X, y, df_feat 
