import pytest

import polars as pl
import numpy as np
import pandas as pd

import polars.testing
from tcbench.modeling import reporting


@pytest.mark.parametrize("test_input,expected,options", [
    (
        pl.DataFrame(dict(
            y_true=["a", "c", "b", "_total_"],
            a=[10, 0, 1, 11],
            c=[0, 20, 1, 21],
            b=[1, 5, 15, 21],
            _total_=[11, 25, 17, 53],
        )),
        pl.DataFrame(dict(
            y_true=["a", "b", "c", "_total_"],
            a=[10, 1, 0, 11],
            b=[1, 15, 5, 21],
            c=[0, 1, 20, 21],
            _total_=[11, 17, 25, 53],
        )),
        dict(
            order="lexicographic", 
            descending=False,
            expected_labels=None,
        ),
    ),
    (
        pl.DataFrame(dict(
            y_true=["a", "c", "b", "_total_"],
            a=[10, 0, 1, 11],
            c=[0, 20, 1, 21],
            b=[1, 5, 15, 21],
            _total_=[11, 25, 17, 53],
        )),
        pl.DataFrame(dict(
            y_true=["c", "b", "a", "_total_"],
            c=[20, 1, 0, 21],
            b=[5, 15, 1, 21],
            a=[0, 1, 10, 11],
            _total_=[25, 17, 11, 53],
        )),
        dict(
            order="lexicographic", 
            descending=True,
            expected_labels=None,
        ),
    ),
    (
        pl.DataFrame(dict(
            y_true=["a", "c", "b", "_total_"],
            a=[10, 0, 1, 11],
            c=[0, 20, 1, 21],
            b=[1, 5, 0, 6],
            _total_=[11, 25, 2, 38],
        )),
        pl.DataFrame(dict(
            y_true=["b", "a", "c", "_total_"],
            b=[0, 1, 5, 6],
            a=[1, 10, 0, 11],
            c=[1, 0, 20, 21],
            _total_=[2, 11, 25, 38],
        )),
        dict(
            order="samples", 
            descending=False,
            expected_labels=None,
        ),
    ),
    (
        pl.DataFrame(dict(
            y_true=["a", "c", "b", "_total_"],
            a=[10, 0, 1, 11],
            c=[0, 20, 1, 21],
            b=[1, 5, 0, 6],
            _total_=[11, 25, 2, 38],
        )),
        pl.DataFrame(dict(
            y_true=["c", "a", "b", "_total_"],
            c=[20, 0, 1, 21],
            a=[0, 10, 1, 11],
            b=[5, 1, 0, 6],
            _total_=[25, 11, 2, 38],
        )),
        dict(
            order="samples", 
            descending=True,
            expected_labels=None,
        ),
    ),
    (
        pl.DataFrame(dict(
            y_true=["a", "c", "b", "_total_"],
            a=[10, 0, 1, 11],
            c=[0, 20, 1, 21],
            b=[1, 5, 15, 21],
            _total_=[11, 25, 17, 53],
        )),
        pl.DataFrame(dict(
            y_true=["a", "b", "c", "_total_"],
            a=[10, 1, 0, 11],
            b=[1, 15, 5, 21],
            c=[0, 1, 20, 21],
            d=[0, 0, 0, 0],
            _total_=[11, 17, 25, 53],
        )),
        dict(
            order="lexicographic", 
            descending=False,
            expected_labels=["a", "b", "c", "d"]
        ),
    ),
])
def test__confusion_matrix_reorder_rows_columns(test_input, expected, options):
    res = reporting._confusion_matrix_reorder_rows_columns(
        test_input, **options
    )
    polars.testing.assert_frame_equal(
        expected, res, check_exact=True
    )
    

@pytest.mark.parametrize("test_input,expected", [
    (
        pl.DataFrame(dict(
            y_true=["a", "c", "b"],
            a=[10, 0, 1],
            c=[0, 20, 1],
            b=[1, 5, 15],
        )),
        pl.DataFrame(dict(
            y_true=["a", "c", "b", "_total_"],
            a=[10, 0, 1, 11],
            c=[0, 20, 1, 21],
            b=[1, 5, 15, 21],
            _total_=[11, 25, 17, 53],
        )),
 )])
def test_confusion_matrix_add_totals(test_input, expected):
    res = reporting._confusion_matrix_add_totals(test_input)

    polars.testing.assert_frame_equal(
        expected.select("_total_"), 
        res.select("_total_")
    )
    polars.testing.assert_frame_equal(
        expected.filter(pl.col("y_true") == "_total_"), 
        res.filter(pl.col("y_true") == "_total_"),
        check_exact=True,
    )


@pytest.mark.parametrize("test_input,expected", [
    (
        pl.DataFrame(dict(
            y_true=["a", "c", "b"],
            a=[10, 0, 1],
            c=[0, 20, 1],
            b=[1, 5, 15],
        )),
        pl.DataFrame(dict(
            y_true=["a", "c", "b"],
            a=[10/11, 0.0, 1/17],
            c=[0.0, 20/25, 1/17],
            b=[1/11, 5/25, 15/17],
        )),
 )])
def test__confusion_matrix_normalize_rows(test_input, expected):
    res = reporting._confusion_matrix_normalize_rows(test_input)
    polars.testing.assert_frame_equal(
        expected, res, check_exact=False
    )


@pytest.mark.parametrize("input_data,expected", [
    (
        dict(
            y_true=np.array(list("aaa bbb ccc".replace(" ", ""))),
            y_pred=np.array(list("abb bca ccc".replace(" ", ""))),
        ),
        pl.DataFrame(dict(
            y_true=list("abc") + ["_total_"],
            a=np.array([1, 1, 0, 2], dtype=np.uint32),
            b=np.array([2, 1, 0, 3], dtype=np.uint32),
            c=np.array([0, 1, 3, 4], dtype=np.uint32),
            _total_=np.array([3, 3, 3, 9], dtype=np.uint32),
        ))
    ),
    (
        dict(
            y_true=np.array(list("aaabbbccc")),
            y_pred=np.array(list("abbbcaccc")),
            order="lexicographic",
            descending=True,
        ),
        pl.DataFrame(dict(
            y_true=list("cba") + ["_total_"],
            c=np.array([3, 1, 0, 4], dtype=np.uint32),
            b=np.array([0, 1, 2, 3], dtype=np.uint32),
            a=np.array([0, 1, 1, 2], dtype=np.uint32),
            _total_=np.array([3, 3, 3, 9], dtype=np.uint32),
        ))
    ),
    (
        dict(
            y_true=np.array(list("bbbb aaaaaa cc".replace(" ", ""))),
            y_pred=np.array(list("bcbc abbbcc ac".replace(" ", ""))),
            order="samples",
            descending=True,
        ),
        pl.DataFrame(dict(
            y_true=list("abc") + ["_total_"],
            a=np.array([1, 0, 1, 2], dtype=np.uint32),
            b=np.array([3, 2, 0, 5], dtype=np.uint32),
            c=np.array([2, 2, 1, 5], dtype=np.uint32),
            _total_=np.array([6, 4, 2, 12], dtype=np.uint32),
        ))
    ),
    (
        dict(
            y_true=np.array(list("bbbb aaaaaa cc".replace(" ", ""))),
            y_pred=np.array(list("bcbc abbbcc ac".replace(" ", ""))),
            order="samples",
            descending=False,
        ),
        pl.DataFrame(dict(
            y_true=list("cba") + ["_total_"],
            c=np.array([1, 2, 2, 5], dtype=np.uint32),
            b=np.array([0, 2, 3, 5], dtype=np.uint32),
            a=np.array([1, 0, 1, 2], dtype=np.uint32),
            _total_=np.array([2, 4, 6, 12], dtype=np.uint32),
        ))
    ),
    (
        dict(
            y_true=np.array(list("aaabbbccc")),
            y_pred=np.array(list("abbbcaccc")),
            expected_labels=["a", "b", "c", "d", "e"],
        ),
        pl.DataFrame(dict(
            y_true=list("abc") + ["_total_"],
            a=np.array([1, 1, 0, 2], dtype=np.uint32),
            b=np.array([2, 1, 0, 3], dtype=np.uint32),
            c=np.array([0, 1, 3, 4], dtype=np.uint32),
            d=np.zeros(4, dtype=np.int64),
            e=np.zeros(4, dtype=np.int64),
            _total_=np.array([3, 3, 3, 9], dtype=np.int64),
        ))
    ),
    (
        dict(
            y_true=np.array(list("aaabbbccc")),
            y_pred=np.array(list("abbbcaccc")),
            order="lexicographic",
            descending=True,
            expected_labels=["a", "b", "c", "d", "e"],
        ),
        pl.DataFrame(dict(
            y_true=list("cba") + ["_total_"],
            c=np.array([3, 1, 0, 4], dtype=np.uint32),
            b=np.array([0, 1, 2, 3], dtype=np.uint32),
            a=np.array([0, 1, 1, 2], dtype=np.uint32),
            e=np.zeros(4, dtype=np.int64),
            d=np.zeros(4, dtype=np.int64),
            _total_=np.array([3, 3, 3, 9], dtype=np.int64),
        ))
    ),
])
def test_confusion_matrix(input_data, expected):
    from sklearn import metrics
    res = reporting.confusion_matrix(**input_data)
    polars.testing.assert_frame_equal(res, expected)

    # re-verify the confusion matrix based on sklearn

    # in case extra columns are injected, we need
    # to remove them from the counting because
    # sklearn always return a squared matrix
    # while the computation only inject columns
    labels = res.drop("y_true", "_total_").columns
    true_labels = np.unique(input_data["y_true"])
    if (
        "expected_labels" in input_data 
        and len(true_labels) < len(labels)
    ):
        labels = [ 
            lab
            for lab in labels
            if lab in true_labels
        ]

    mtx_expected = metrics.confusion_matrix(
        y_true=input_data["y_true"],
        y_pred=input_data["y_pred"],
        labels=labels,
    )
    mtx_found = (
        res
        .filter(pl.col("y_true") != "_total_")
        .drop("y_true", "_total_")
        .to_numpy()
        [:len(labels), :len(labels)]
    )
    assert (mtx_expected == mtx_found).all()


@pytest.mark.parametrize("test_input,expected", [
    (
        pl.DataFrame(dict(
            y_true=list("cba") + ["_total_"],
            c=np.array([3, 1, 0, 4]),
            b=np.array([0, 1, 2, 3]),
            a=np.array([0, 1, 1, 2]),
            e=np.zeros(4, dtype=int),
            d=np.zeros(4, dtype=int),
            _total_=np.array([3, 3, 3, 9])
        )),
        pl.DataFrame(dict(
            y_true=list("ccc b b b aa a".replace(" ", "")),
            y_pred=list("ccc c b a bb a".replace(" ", "")),
        ))
    ),
    (
        pl.DataFrame(dict(
            y_true=["a", "c", "b"],
            a=[10, 0, 1],
            c=[0, 20, 1],
            b=[1, 5, 15],
        )),
        pl.DataFrame(dict(
            y_true=list("a"*10 + "a" + "c"*20 + "c"*5 + "b" + "b" + "b"*15),
            y_pred=list("a"*10 + "b" + "c"*20 + "b"*5 + "a" + "c" + "b"*15),
        ))
    )
])
def test_y_columns_from_confusion_matrix(test_input, expected):
    res = reporting.y_columns_from_confusion_matrix(test_input)

    polars.testing.assert_frame_equal(
        res.sort(by=["y_true", "y_pred"]),
        expected.sort(by=["y_true", "y_pred"])
    )


@pytest.mark.parametrize("test_input, other_input_params, expected_order", [
    (
        pl.DataFrame(dict(
            y_true=list("cba") + ["_total_"],
            c=np.array([3, 1, 0, 4]),
            b=np.array([0, 1, 2, 3]),
            a=np.array([0, 1, 1, 2]),
            e=np.zeros(4, dtype=int),
            d=np.zeros(4, dtype=int),
            _total_=np.array([3, 3, 3, 9])
        )),
        dict(),
        list("abc"),
    ),
    (
        pl.DataFrame(dict(
            y_true=["a", "c", "b", "_total_"],
            a=[10, 0, 1, 11],
            c=[0, 20, 1, 21],
            b=[1, 5, 15, 21],
            _total_=[11, 25, 17, 53],
        )),
        dict(),
        list("abc")
    ),
    (
        pl.DataFrame(dict(
            y_true=["a", "c", "b", "_total_"],
            a=[10, 0, 1, 11],
            c=[0, 20, 1, 21],
            b=[1, 5, 15, 21],
            _total_=[11, 25, 17, 53],
        )),
        dict(
            order="samples",
            descending=False,
        ),
        list("abc")
    ),
    (
        pl.DataFrame(dict(
            y_true=["a", "c", "b", "_total_"],
            a=[10, 0, 1, 11],
            c=[0, 20, 1, 21],
            b=[1, 5, 15, 21],
            _total_=[11, 25, 17, 53],
        )),
        dict(
            order="samples",
            descending=True,
        ),
        list("cba")
    )
])
def test_classification_report_from_confusion_matrix(
    test_input, other_input_params, expected_order
):
    res = reporting.classification_report_from_confusion_matrix(
        test_input,
        **other_input_params,
    )

    labels = res.select(pl.col("label")).to_numpy().squeeze()[:-3].tolist()
    assert labels == expected_order

    from sklearn.metrics import classification_report
    df_labels = reporting.y_columns_from_confusion_matrix(test_input)
    expected = pl.from_pandas(
        pd.DataFrame(
            classification_report(
                df_labels["y_true"].to_numpy(),
                df_labels["y_pred"].to_numpy(),
                output_dict=True
            )
        )
        .T
        .reset_index()
        .rename({"index": "label"}, axis=1)
    )

    polars.testing.assert_frame_equal(
        res.sort("label"),
        expected.sort("label"),
        check_dtypes=False
    )


@pytest.mark.parametrize("input_params", [
    (
        dict(
            y_true=np.array(list("ccc b b b aa a".replace(" ", ""))),
            y_pred=np.array(list("ccc c b a bb a".replace(" ", ""))),
        )
    ),
    (
        dict(
            y_true=np.array(
                list("a"*10 + "a" + "c"*20 + "c"*5 + "b" + "b" + "b"*15)
            ),
            y_pred=np.array(
                list("a"*10 + "b" + "c"*20 + "b"*5 + "a" + "c" + "b"*15)
            )
        )
    )
])
def text_confusion_matrix(input_params):
    y_true = input_params["y_true"]
    y_pred = input_params["y_pred"]
    res = reporting.classification_report(y_true, y_pred)

    from sklearn.metrics import classification_report
    expected = pl.from_pandas(
        pd.DataFrame(
            classification_report(y_true, y_pred, output_dict=True)
        )
        .T
        .reset_index()
        .rename({"index": "label"}, axis=1)
    )

    polars.testing.assert_frame_equal(
        res.sort("label"),
        expected.sort("label")
    )

