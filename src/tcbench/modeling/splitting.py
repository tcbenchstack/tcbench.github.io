from __future__ import annotations

from typing import Tuple

import polars as pl

from tcbench.core import to_lazy

from tcbench.modeling.columns import (
    COL_APP,
    COL_ROW_ID,
    SPLIT_NAME_TEST,
    SPLIT_NAME_TRAIN,
    COL_SPLIT_NAME,
    COL_SPLIT_INDEX,
    COL_SPLIT_TRAIN_INDICES,
    COL_SPLIT_TEST_INDICES,
)

def _verify_split_indices_table(
    df: pl.DataFrame, 
    y_colname:str, 
    index_colname:str = "row_id",
    expected_test_size: float = 0.1
) -> bool:
    df_split_sizes = (df
        .group_by(y_colname, COL_SPLIT_NAME)
        .agg(
            pl.col(index_colname).count()
        )
        .pivot(
            index=y_colname,
            on=COL_SPLIT_NAME,
            values=index_colname,
        )
        .with_columns(
            total=(
                pl.sum_horizontal(SPLIT_NAME_TEST, SPLIT_NAME_TRAIN)
            )
        )
        .with_columns(
            test_size=(
                pl.col(SPLIT_NAME_TEST) / pl.col("total")
            ),
            train_size=(
                pl.col(SPLIT_NAME_TRAIN) / pl.col("total")
            )
        )
        .sort("total", descending=True)
    )
    
    df_tmp = df_split_sizes.filter(pl.col("test_size") < expected_test_size)
    assert df_tmp.is_empty(), \
        "some classes have less test sample than expected" + "\n" + str(df_tmp)
    
    df_tmp = df_split_sizes.filter(pl.col("train_size") == 0)
    assert df_tmp.is_empty(), \
        "some classes have no train samples" + "\n" + str(df_tmp)

    return True

def _get_split_indices(
    df: pl.LazyFrame, 
    *,
    y_colname: str = COL_APP,
    index_colname: str = COL_ROW_ID,
    test_size: float = 0.1,
    seed: int = 1, 
) -> pl.DataFrame:
    df_split_indices = (df
        .select(
            y_colname,
            index_colname,
        )
        .group_by(y_colname)
        .map_groups(
            lambda df_group: (
                df_group
                # new index, specific for each label
                .with_row_index("_inner_idx")
                # shuffle original index
                .with_columns(
                    pl.col(index_colname).shuffle(seed=seed)    
                )
                .with_columns(**{
                    # split=test for the first test_size samples
                    # split=train otherwise
                    COL_SPLIT_NAME: (
                        pl.when(
                            pl.col("_inner_idx") 
                            < pl.lit(len(df_group) * test_size).ceil()
                        )
                        .then(pl.lit(SPLIT_NAME_TEST))
                        .otherwise(pl.lit(SPLIT_NAME_TRAIN))
                    )
                })
                .drop("_inner_idx")
            )
        )
    )    

    _verify_split_indices_table(
        df_split_indices, 
        y_colname,
        expected_test_size=test_size,
        index_colname=index_colname
    )

    return df_split_indices

def _split_indices_to_list(
    df: pl.LazyFrame, 
    split_index: int, 
    index_colname:str = COL_ROW_ID,
) -> pl.DataFrame:
    return (
        df
        .group_by(COL_SPLIT_NAME)
        .agg(pl.col(index_colname))
        .with_columns(**{
            COL_SPLIT_INDEX: pl.lit(split_index)
        })
        .pivot(
            on=COL_SPLIT_NAME,
            index=COL_SPLIT_INDEX,
            values=index_colname,
        )
        .rename({
            SPLIT_NAME_TRAIN: COL_SPLIT_TRAIN_INDICES,
            SPLIT_NAME_TEST: COL_SPLIT_TEST_INDICES,
        })
        # guarantee that columns order is deterministic
        .select(
            COL_SPLIT_INDEX,
            COL_SPLIT_TRAIN_INDICES,
            COL_SPLIT_TEST_INDICES
        )
    )


def _split_indices_from_list(
    df_splits: pl.LazyFrame, 
    split_index: int, 
    *,
    index_colname: str = COL_ROW_ID,
) -> pl.LazyFrame:
    """Extract the train and test indices from a specific split
    are compose a LazyFrame with two columns reporting the 
    row index and the related split type
    """
    expr = pl.col(COL_SPLIT_INDEX) == pl.lit(split_index)
    expr = pl.col(COL_SPLIT_INDEX) == pl.lit(split_index)
    return pl.concat((
        df_splits
            .filter(expr)
            .select(
                (
                    pl.col(COL_SPLIT_TRAIN_INDICES)
                    .list
                    .explode()
                    .alias(index_colname)
                ),
                pl.lit(SPLIT_NAME_TRAIN).alias(COL_SPLIT_NAME)
            ),
        df_splits
            .filter(expr)
            .select(
                (
                    pl.col(COL_SPLIT_TEST_INDICES)
                    .list
                    .explode()
                    .alias(index_colname)
                ),
                pl.lit(SPLIT_NAME_TEST).alias(COL_SPLIT_NAME)
            )
    ))


def add_split_column(
    df: pl.DataFrame, 
    y_colname: str = "app",
    index_colname: str = "row_id", 
    test_size: float = 0.1,
    seed: int = 1, 
) -> pl.DataFrame:
    df_split_indices = _get_split_indices(
        df, 
        y_colname=y_colname, 
        index_colname=index_colname, 
        test_size=test_size, 
        seed=seed
    )
    return df.join(
        df_split_indices, 
        on=[y_colname, index_colname]
    )

def split_monte_carlo(
    df: pl.DataFrame,
    y_colname: str,
    index_colname: str = "row_id",
    num_splits: int = 1,
    seed: int = 1,
    test_size: float = 0.1,
) -> pl.DataFrame:
    return pl.concat([
        _split_indices_to_list(
            _get_split_indices(
                df, 
                y_colname=y_colname, 
                index_colname=index_colname, 
                test_size=test_size, seed=seed+split_index
            ),
            split_index=split_index
        )
        for split_index in range(1, num_splits+1)
    ])


def get_train_test_splits(
    df: pl.DataFrame | pl.LazyFrame, 
    df_splits: pl.DataFrame | pl.LazyFrame, 
    *,
    split_index: int = 1,
    index_colname: str = COL_ROW_ID,
) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
    df = to_lazy(df)
    df_splits = to_lazy(df_splits)

    df_split_indices = _split_indices_from_list(
        df_splits, 
        split_index, 
        index_colname=index_colname
    )
    df_tmp = df.join(
        df_split_indices,
        on=[index_colname]
    )
    return (
        df_tmp.filter(pl.col(COL_SPLIT_NAME) == SPLIT_NAME_TRAIN),
        df_tmp.filter(pl.col(COL_SPLIT_NAME) == SPLIT_NAME_TEST)
    )
