from __future__ import annotations

import polars as pl
import numpy as np

from numpy.typing import NDArray
from typing import Iterable, List

def _confusion_matrix_add_totals(
    conf_mtx: pl.DataFrame,
) -> pl.DataFrame:
    # add horizontal sum
    conf_mtx = conf_mtx.with_columns(
        _total_=(
            conf_mtx
            .drop("y_true")
            .sum_horizontal()
        )
    )
    # compute vertical sum
    total_row = (
        conf_mtx
        .drop("y_true")
        .sum()
        .with_columns(y_true=pl.lit("_total_"))
        .select(conf_mtx.columns)
    )
    # combine everything
    return pl.concat((
        conf_mtx,
        total_row
    ))


def _confusion_matrix_reorder_rows_columns(
    conf_mtx: pl.DataFrame,
    *,
    expected_labels: List[str] | None,
    order: str = "lexicographic",
    descending: bool = False,
) -> pl.DataFrame:
    if order not in ("lexicographic", "samples"):
        raise ValueError(
            f"""Unrecognized {order=} can be "lexicographic" or "samples" """
        )

    # remove all totals (if any)
    conf_mtx = (
        conf_mtx
        .drop("_total_", strict=False)
        .filter(pl.col("y_true") != "_total_")
    )
    # find true labels
    true_labels = (
        conf_mtx
        .select("y_true")
        .to_numpy()
        .squeeze()
        .tolist()
    )
    # find predicted labels
    pred_labels = conf_mtx.drop("y_true").columns

    if len(pred_labels) < len(true_labels):
        raise RuntimeError(
            "The number of predicted classes cannot be lower than "
            "the number of true classes: \n"
            f"{true_labels=}\n"
            f"{pred_labels=}"
        )

    # inject empty prediction columns (if needed)
    # for predicted classes expected but not found
    if expected_labels is not None:
        conf_mtx = (
            conf_mtx.with_columns(
                **{
                    col: pl.lit(0).cast(pl.Int64())
                    for col in expected_labels
                    if col not in pred_labels
                }
            )
            .select("y_true", *expected_labels)
        )
        pred_labels = expected_labels

    if order == "lexicographic":
        true_labels = sorted(true_labels, reverse=descending)
    elif order == "samples":
        true_labels = (
            pl.DataFrame({
                "y_true": conf_mtx["y_true"],
                "samples": conf_mtx.drop("y_true").sum_horizontal()
            })
            .sort("samples", descending=descending)
            .to_series()
            .to_list()
        )

    # Note: prediction might regard only a subset of classes,
    # so the confusion matrix might have less rows than columns.
    # In this case, the columns are rearranged so place first
    # the true labels based on the predefined order, then
    # the remainder in lexicographic order
    sorted_labels = true_labels
    if len(true_labels) < len(pred_labels):
        columns = true_labels[:]
        for col in sorted(pred_labels, reverse=descending):
            if col not in columns:
                columns.append(col)
        sorted_labels = columns

    label_to_idx = dict(zip(
        sorted_labels, range(len(sorted_labels))
    ))

    # reorder columns
    conf_mtx = conf_mtx.select(
        "y_true",
        *sorted_labels,
    )

    # reorder rows
    conf_mtx = (
        # inject a dummy column to define rows order
        conf_mtx.with_columns(
            y_true_order=(
                pl.col("y_true")
                .map_elements(
                    function=lambda text: label_to_idx.get(text, -1),
                    return_dtype=pl.UInt32
                )
            )
        )
        # ...and impose row order
        .sort("y_true_order", "y_true", descending=False)
        .drop("y_true_order")
        # # ...and add total by row
        # .with_columns(
        #     _total_=pl.sum_horizontal(expected_labels)
        # )        
        # .fill_null(0)
    )
    return _confusion_matrix_add_totals(conf_mtx)


def _confusion_matrix_normalize_rows(
    conf_mtx: pl.DataFrame,
) -> pl.DataFrame:
    if (
        "_total_" not in conf_mtx.columns 
        or len(conf_mtx.filter(pl.col("y_true") == "_total_")) == 0
    ):
        conf_mtx = _confusion_matrix_add_totals(conf_mtx)
    # remove columns totals
    conf_mtx = conf_mtx.filter(pl.col("y_true") != "_total_")
    return (
        conf_mtx.select(
            "y_true", 
            *[
                pl.col(col).truediv(pl.col("_total_"))
                for col in conf_mtx.drop("y_true", "_total_").columns
            ]
        )
    )


def confusion_matrix(
    y_true: NDArray, 
    y_pred: NDArray, 
    *,
    expected_labels: List[str] | None = None, 
    order: str | None = "lexicographic",
    descending: bool = False,
    normalize: bool = False,
) -> pl.DataFrame:
    # compute base confusion matrix
    conf_mtx = (
        pl.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred,
        })	
        .group_by("y_true", "y_pred")
        .len()
        .pivot(
            index="y_true",
            on="y_pred",
            values="len"
        )
        .fill_null(0)
    )

    if expected_labels is None:
        expected_labels = conf_mtx.drop("y_true").columns
    if order is None:
        order = "lexicographic"

    conf_mtx = _confusion_matrix_reorder_rows_columns(
        conf_mtx, 
        expected_labels=expected_labels,
        order=order,
        descending=descending,
    )

    if normalize:
        conf_mtx = _confusion_matrix_normalize_rows(conf_mtx)

    return conf_mtx


def classification_report_from_confusion_matrix(
    conf_mtx: pl.DataFrame,
    *,
    order: str = "lexicographic",
    descending: bool = False,
) -> pl.DataFrame:

    # Note: the confusion matrix might have 
    # empty columns, so the labels
    # are only the y_true values
    labels = (
        conf_mtx
        .select("y_true")
        .filter(
            pl.col("y_true") != "_total_"
        )
        .to_numpy()
        .squeeze()
        .tolist()
    )

    diag_counts = (
        conf_mtx
        .filter(pl.col("y_true") != "_total_")
        .select(*labels)
        .to_numpy()
        [np.diag_indices(len(labels))]
    )
    true_counts = (
        conf_mtx
            .filter(pl.col("y_true") != "_total_")
            ["_total_"]
            .to_numpy()
    )
    pred_counts = (
        conf_mtx
        .filter(pl.col("y_true") == "_total_")
        .select(*labels)
        .to_numpy()
        .squeeze()
    )

    recall = diag_counts / true_counts
    precision = diag_counts / pred_counts
    f1_score = 2 * (precision * recall) / (precision + recall)
    accuracy = diag_counts.sum() / true_counts.sum()

    schema={
        "label": pl.String,
        "precision": pl.Float32,
        "recall": pl.Float32,
        "f1-score": pl.Float32,
        "support": pl.Float32
    }

    class_rep = (
        pl.DataFrame(
            {
                "label": labels,
                "precision": precision,
                "recall": recall,
                "f1-score": f1_score,
                "support": true_counts,
            }, 
            schema=schema
        )
    )
    if order == "lexicographic":
        class_rep = class_rep.sort("label", descending=descending)
    elif order == "samples":
        class_rep = class_rep.sort("support", descending=descending)

    extra_metrics = (
        pl.DataFrame(
            {
                "label": [
                    "accuracy",
                    "macro avg",
                    "weighted avg",
                ],
                "precision": [
                    accuracy,
                    precision.mean(),
                    np.dot(precision, true_counts) / true_counts.sum(),
                ],
                "recall": [
                    accuracy,  
                    recall.mean(),
                    np.dot(recall, true_counts) / true_counts.sum(),
                ],
                "f1-score": [
                    accuracy,
                    f1_score.mean(),
                    np.dot(f1_score, true_counts) / true_counts.sum(),
                ],
                "support": [
                    accuracy,
                    true_counts.sum(),
                    true_counts.sum(),
                ]
            }, 
            schema=schema
        )
    )

    return pl.concat((class_rep, extra_metrics))


def classification_report(
    y_true: NDArray,
    y_pred: NDArray,
    *,
    expected_labels: List[str] | None = None,
    order: str = "lexicographic",
    descending: bool = False,
) -> pl.DataFrame:
    conf_mtx = confusion_matrix(
        y_true, 
        y_pred, 
        expected_labels=expected_labels,
        order=order,
        descending=descending,
        normalize=False,
    )
    return classification_report_from_confusion_matrix(
        conf_mtx, 
        order=order, 
        descending=descending
    )

def average_confusion_matrix(
    data: List[pl.DataFrame],
    expected_labels: List[str] | None = None, 
    order: str | None = "lexicographic",
    descending: bool = False,
    normalize: bool = False,
) -> pl.DataFrame:

    def _reorder_rows_columns(
            df: pl.DataFrame
    ) -> tuple[pl.DataFrame, list[str]]:
        _df = (
            df
            .drop("_total_")
            .filter(pl.col("y_true") != "_total_")
            .sort(by="y_true", descending=False)
        )
        _rows = (
            _df["y_true"]
            .to_numpy()
            .squeeze()
            .tolist()
        )
        _df = _df.select(_rows)
        return _df, _rows

    def _verify_labels(expected_labels, found_labels):
        if expected_labels != found_labels:
            raise RuntimeError(
                f"Different rows for {idx=}\n"
                f"{expected_labels=}\n"
                f"{found_labels=}" 
            )

    confmtx, found_labels = _reorder_rows_columns(data[0])
    mtx = confmtx.to_numpy()

    if expected_labels is None:
        expected_labels = found_labels
    else:
        _verify_labels(expected_labels, found_labels)

    for idx, confmtx in enumerate(data[1:], start=2):
        confmtx, found_labels = _reorder_rows_columns(confmtx)
        _verify_labels(expected_labels, found_labels)
        mtx += confmtx.to_numpy()

    confmtx_avg = pl.DataFrame({
        col_name: mtx[:, idx].tolist() + [mtx[:, idx].sum()]
        for idx, col_name in enumerate(expected_labels)
    })
    confmtx_avg = (
        confmtx_avg
        .with_columns(
            y_true=pl.Series(expected_labels + ["_total_"]),
           _total_=confmtx_avg.sum_horizontal(), 
        )
        .select(
            ["y_true"] + expected_labels + ["_total_"]
        )
    )

    return confmtx_avg


def y_columns_from_confusion_matrix(conf_mtx: pl.DataFrame) -> pl.DataFrame:
    df_tmp = (
        conf_mtx
        # remove totals
        .drop("_total_", strict=False)
        .filter(pl.col("y_true") != "_total_")
        # disaggregate matrix in row/col pairs
        .unpivot(index="y_true")
        # remove pairs without an occurrence count
        .filter(pl.col("value") > 0)
        .rename({
            "variable": "y_pred",
            "value": "count"
        })
    )

    return (
        df_tmp
        # form lists repeating each pair value
        .select(
            pl.col("y_true", "y_pred")
            .repeat_by(pl.col("count"))
        )
        .explode("y_true", "y_pred")
    )
