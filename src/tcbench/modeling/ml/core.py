from __future__ import annotations

import polars as pl
import numpy as np

import pathlib
import itertools

from typing import Tuple, List, Dict, Any, Iterable, Callable
from numpy.typing import NDArray
from rich.console import RenderableType

from sklearn.preprocessing import LabelEncoder

from dataclasses import dataclass
from collections.abc import Iterator

from tcbench import fileutils
from tcbench.cli import richutils
from tcbench.modeling import (
    splitting, 
    datafeatures,
    reporting,
)
from tcbench.datasets.core import (
    Dataset,
)
from tcbench.modeling.datafeatures import (
    DEFAULT_EXTRA_COLUMNS
)
from tcbench.modeling.columns import (
    COL_APP,
    COL_ROW_ID,
    COL_SPLIT_INDEX,
)
from tcbench.modeling.enums import MODELING_FEATURE




def compose_hyperparams_grid(hyperparams: Dict[str, Any]) -> Tuple[Dict[str, Any]]:
    opts = []
    for key, value in hyperparams.items():
        if not isinstance(value, (list, tuple)):
            opts.append(((key, value)))
        else:
            opts.append(zip(itertools.repeat(key, len(value)), value))
    return tuple(dict(pairs) for pairs in itertools.product(*opts))


class ClassificationResults:
    def __init__(
        self, 
        df_feat: pl.DataFrame,
        labels: NDArray,
        y_true: NDArray | None = None,
        y_pred: NDArray | None = None,
        split_index: int | None = None,
        name: str = "test",
        with_reports: bool = True,
        model: MLModel | None = None,
    ):
        self.labels = labels
        self.model = model
        self.name = name
        self.df_feat = df_feat
        self.split_index = split_index

        if self.df_feat is not None:
            if y_true is not None:
                self.df_feat = self.df_feat.with_columns(
                    y_true=pl.Series(y_true),
                    y_pred=pl.Series(y_pred),
                )
            self.df_feat = self.df_feat.with_columns(
                split_index=pl.lit(split_index) if split_index else None
            ) 
        self.confusion_matrix = None
        self.confusion_matrix_normalized = None
        self.classification_report = None
        if with_reports:
            self.compute_reports()

    @property
    def y_true(self) -> NDArray:
        return self.df_feat["y_true"].to_numpy()

    @property
    def y_pred(self) -> NDArray:
        return self.df_feat["y_pred"].to_numpy()

    def _weighted_metric(self, metric: str) -> float | None:
        if self.classification_report is None:
            return None
        return (
            self.classification_report
            .filter(pl.col("label") == "weighted avg")
            [metric]
            .to_numpy()
            [0]
        )

    @property
    def weighted_f1(self) -> float | None:
        return self._weighted_metric("f1-score")

    @property
    def weighted_recall(self) -> float | None:
        return self._weighted_metric("recall")

    @property
    def weighted_precision(self) -> float | None:
        return self._weighted_metric("precision")

    def compute_reports(self) -> None:
        if not (
            ("y_true" in self.df_feat.columns) 
            or ("y_pred" in self.df_feat.columns)
        ):
            raise RuntimeError("MissingColumns: y_true or y_pred are missing!")

        y_true = self.y_true
        y_pred = self.y_pred

        self.confusion_matrix = reporting.confusion_matrix(
            y_true, 
            y_pred, 
            expected_labels=self.labels,
            order="samples", 
            descending=True, 
            normalize=False
        )
        self.confusion_matrix_normalized = reporting.confusion_matrix(
            y_true, 
            y_pred, 
            expected_labels=self.labels,
            order="samples", 
            descending=True, 
            normalize=True
        )
        self.classification_report = reporting.classification_report_from_confusion_matrix(
            self.confusion_matrix, order="samples", descending=True
        )

    def set_split_index(self, n: int) -> None:
        self.df_feat = self.df_feat.with_columns(
            split_index=pl.literal(n)
        )

    def save(
        self, 
        save_to: pathlib.Path, 
        name: str = "",
        with_progress: bool = True, 
        echo: bool = False
    ) -> None:
        save_to = pathlib.Path(save_to)
        fileutils.create_folder(save_to, echo=echo)

        if name == "":
            name = self.name

        with richutils.SpinnerProgress("Saving...", visible=echo):
            self.df_feat.write_parquet(save_to / f"{name}_df_feat.parquet")
            if self.confusion_matrix is not None:
                fileutils.save_csv(
                    self.confusion_matrix,
                    save_to / f"{name}_confusion_matrix.csv",
                    echo=echo
                )
            if self.confusion_matrix_normalized is not None:
                fileutils.save_csv(
                    self.confusion_matrix_normalized,
                    save_to / f"{name}_confusion_matrix_normalized.csv",
                    echo=echo,
                )
            if self.classification_report is not None:
                fileutils.save_csv(
                    self.classification_report,
                    save_to / f"{name}_classification_report.csv",
                    echo=echo,
                )
            fileutils.save_pickle(
                self.labels, 
                save_to / f"{name}_labels.pkl",
                echo=echo
            )

    @classmethod
    def load(cls, folder: pathlib.Path, name: str = "test", echo: bool = False) -> ClassificationResults:
        folder = pathlib.Path(folder)
        model = fileutils.load_if_exists(
            folder / "model.pkl", 
            echo=echo,
            error_policy="return",
        )
        labels = fileutils.load_if_exists(
            folder / f"{name}_labels.pkl", 
            echo=echo,
            error_policy="raise"
        )
        df_feat = fileutils.load_if_exists(
            folder / f"{name}_df_feat.parquet", 
            echo=echo,
            error_policy="return"
        )
        confusion_matrix = fileutils.load_if_exists(
            folder / f"{name}_confusion_matrix.csv", 
            echo=echo,
            error_policy="raise"
        )
        confusion_matrix_normalized = fileutils.load_if_exists(
            folder / f"{name}_confusion_matrix_normalized.csv", 
            echo=echo,
            error_policy="raise"
        )
        classification_report = fileutils.load_if_exists(
            folder / f"{name}_classification_report.csv", 
            echo=echo,
            error_policy="raise"
        )
        clsres = ClassificationResults(
            df_feat=df_feat,
            labels=labels,
            y_true=None,
            y_pred=None,
            model=model,
            with_reports=False
        )
        clsres.confusion_matrix = confusion_matrix
        clsres.confusion_matrix_normalized = confusion_matrix_normalized
        clsres.classification_report = classification_report
        return clsres


# @dataclass
# class MultiClassificationResults:
#     train: ClassificationResults = None
#     val: ClassificationResults = None
#     test: ClassificationResults = None
#     model: MLModel = None
#
#     @classmethod
#     def load(
#         cls, 
#         folder: pathlib.Path, 
#         name_train: str = "train",
#         name_test: str = "test",
#         name_val: str = "val",
#         echo: bool = False,
#     ) -> MultiClassificationResults:
#         folder = pathlib.Path(folder)
#
#         clsres = MultiClassificationResults(
#             model = MLModel.load(folder)
#         )
#         if name_train and (folder / f"{name_train}_df_feat.parquet").exists():
#             clsres.train = ClassificationResults.load(
#                 folder, name=name_train, echo=echo
#             )
#         if name_test and (folder / f"{name_test}_df_feat.parquet").exists():
#             clsres.test = ClassificationResults.load(
#                 folder, name=name_test, echo=echo
#             )
#         if name_val and (folder / f"{name_val}_df_feat.parquet").exists():
#             clsres.val = ClassificationResults.load(
#                 folder, name=name_val, echo=echo
#             )
#         return clsres
#
#     @property
#     def f1_train(self) -> float:
#         if self.train is not None:
#             return self.train.f1
#         return None
#
#     @property
#     def f1_test(self) -> float:
#         if self.test is not None:
#             return self.test.f1
#         return None
#
#     @property
#     def f1_val(self) -> float:
#         if self.val is not None:
#             return self.val.f1
#         return None


@dataclass
class SplitData:
    X_train: NDArray | None
    y_train: NDArray | None
    X_test: NDArray | None
    y_test: NDArray | None
    df_train_feat: pl.DataFrame | None
    df_test_feat: pl.DataFrame | None
    split_index: int
    labels: List[str]
    features: List[str]


class MLDataLoaderException(Exception):
    pass


class MLDataLoader(Iterator):
    def __init__(
        self,
        dset: Dataset,
        features: MODELING_FEATURE | Iterable[MODELING_FEATURE],
        df_splits: pl.DataFrame | None,
        split_indices: List[int] | None = None,
        y_colname: str = COL_APP,
        index_colname: str = COL_ROW_ID,
        series_len: int | None = None,
        series_pad: int | None = None,
        extra_colnames: Iterable[str] = DEFAULT_EXTRA_COLUMNS,
        shuffle_train: bool = True,
        seed: int = 1,
    ):
        self.dset = dset
        self.y_colname = y_colname
        self.index_colname = index_colname
        self.df_splits = df_splits
        self.features = [features] if isinstance(features, MODELING_FEATURE) else features
        self.extra_colnames = extra_colnames
        self.series_len = series_len
        self.series_pad = series_pad
        self.shuffle_train = shuffle_train
        self.seed = seed

        if not dset.is_loaded:
            raise RuntimeError(
                f"Dataset {dset.name} is not loaded!"
                " Call the Dataset .load() method before creating a DataLoader()"
            )

        self._df = dset.df
        if isinstance(self._df, pl.DataFrame):
            self._df = self._df.lazy()

        self._df_splits = dset.df_splits if df_splits is None else df_splits
        if self._df_splits is None:
            raise RuntimeError(
                f"No data splits found for {dset.name}! "
                "Either provide them when creating the DataLoader "
                "or verify that the Dataset was loaded and has data splits"
            )
        elif isinstance(self._df_splits, pl.DataFrame):
            self._df_splits = self._df_splits.lazy()


        self._labels = self._get_labels(self._df, y_colname)
        self._split_indices = self._get_split_indices(
            self._df_splits, split_indices
        )

        self._iter_idx: int | None = None


    def _get_labels(
        self, 
        df: pl.LazyFrame,
        y_colname: str,
    ) -> List[str]:
        return (
            df
            .select(y_colname)
            .unique()
            .sort(by=y_colname)
            .collect()
            .to_numpy()
            .squeeze()
            .tolist()
        )

    def _get_split_indices(
        self, 
        df: pl.LazyFrame,
        split_indices: Iterable[int] | None
    ) -> NDArray:
        if split_indices is not None:
            return np.array(split_indices)
        return (
            df
            .select(COL_SPLIT_INDEX)
            .collect()
            .to_numpy()
            .squeeze()
        )

    @property
    def labels(self) -> List[str]:
        return self._labels

    @property
    def num_splits(self) -> int:
        return len(self._split_indices)

    def _dataprep(
        self, 
        df: pl.DataFrame | pl.LazyFrame, 
        *,
        shuffle: bool = False, 
        seed: int = 1
    ) -> Tuple[NDArray, NDArray, pl.DataFrame]:
        _df = df if isinstance(df, pl.DataFrame) else df.collect()
        if shuffle:
            _df = _df.sample(
                fraction=1, 
                shuffle=True, 
                seed=seed,
            )

        _series_len = self.series_len if self.series_len is not None else -1
        _series_pad = self.series_pad if self.series_pad is not None else -1
        return datafeatures.features_dataprep(
            _df,
            self.features,
            series_len=_series_len,
            series_pad=_series_pad,
            y_colname=self.y_colname,
            extra_colnames=self.extra_colnames,
        )

    def _verify_labels(self, df: pl.LazyFrame, split_index: int) -> None:
        expected = self._labels
        found = self._get_labels(df, self.y_colname)
        if expected == found:
            return 

        if len(expected) != len(found):
            raise MLDataLoaderException(
                f"Split {split_index} expected {len(expected)} labels but found {len(found)}"
            )
        msg = ""
        if len(found) > len(expected):
            extra_labels = sorted(set(found) - set(expected))
            msg = f"Split {split_index} has extra labels ({len(extra_labels)}):"
            msg += ", ".join(extra_labels)
        else:
            missing_labels = sorted(set(expected) - set(found))
            msg = f"Split {split_index} has missing labels ({len(missing_labels)}):"
            msg += ", ".join(missing_labels)
        raise MLDataLoaderException(msg)

    def _get_split_data(
        self, 
        split_index: int,
        with_train: bool = True,
        with_test: bool = True,
    ) -> SplitData:
        _df_train = None
        _df_test = None
        _X_train = None
        _X_test = None
        _y_train = None
        _y_test = None
        _df_train_feat = None
        _df_test_feat = None
        _features = []

        _df_train, _df_test = splitting.get_train_test_splits(
            self._df, 
            self._df_splits,
            split_index=split_index,
            index_colname=self.index_colname
        )
        self._verify_labels(_df_train, split_index)
        self._verify_labels(_df_test, split_index)

        if with_train:
            _X_train, _y_train, _df_train_feat = \
                self._dataprep(
                    _df_train,
                    shuffle=self.shuffle_train,
                    seed=self.seed + split_index
                )
            _features = [
                col
                for col in _df_train_feat.columns
                if col not in (self.y_colname, *self.extra_colnames)
            ]
        else:
            _df_train = None

        if with_test:
            _X_test, _y_test, _df_test_feat = \
                self._dataprep(
                    _df_test,
                    shuffle=False,
                )
            _features = [
                col
                for col in _df_test_feat.columns
                if col not in (self.y_colname, *self.extra_colnames)
            ]
        else:
            _df_test = None

        return SplitData(
            X_train=_X_train,
            X_test=_X_test,
            y_train=_y_train,
            y_test=_y_test,
            df_train_feat=_df_train_feat,
            df_test_feat=_df_test_feat,
            split_index=split_index,
            labels=self.labels,
            features=_features,
        )

    def __next__(self) -> SplitData:
        if self._iter_idx is None:
            self._iter_idx = 0
        elif self._iter_idx == self.num_splits:
            raise StopIteration()
        split_index = self._split_indices[self._iter_idx]
        self._iter_idx += 1
        return self._get_split_data(split_index)

    def __iter__(self) -> Iterator[SplitData]:
        self._iter_idx = None
        return self

    def train_loader(self, split_index: int) -> SplitData:
        return self._get_split_data(split_index, with_train=True, with_test=False)

    def test_loader(self, split_index: int) -> SplitData:
        return self._get_split_data(split_index, with_train=False, with_test=True)

    def train_test_loader(self, split_index: int) -> SplitData:
        return self._get_split_data(split_index, with_train=True, with_test=True)

    def train_loaders(self) -> Iterable[SplitData]:
        for split_index in self._split_indices:
            yield self._get_split_data(split_index, with_train=True, with_test=False)

    def test_loaders(self) -> Iterable[SplitData]:
        for split_index in self._split_indices:
            yield self._get_split_data(split_index, with_train=False, with_test=True)

    def train_test_loaders(self) -> Iterable[SplitData]:
        for split_index in self._split_indices:
            yield self._get_split_data(split_index, with_train=True, with_test=True)



class MLModel:
    def __init__(
        self,
        labels: Iterable[str],
        features: Iterable[MODELING_FEATURE],
        model_class: Callable,
        seed: int = 1,
        *,
        hyperparams: Dict[str, Any],
    ):
        self.labels = labels
        self.features = features
        self.hyperparams = hyperparams
        self.seed = seed

        self._params = dict()
        self._label_encoder = self._fit_label_encoder(self.labels)
        self._model = model_class(**hyperparams)

    @property
    def name(self) -> str:
        return self._model.__class__.__name__

    def __rich_docs__(self) -> RenderableType:
        from rich.table import Table
        from rich import box
        from rich.console import Console
        from rich.pretty import pprint

        table1 = Table(
            'Python repr()',
            box=box.MINIMAL,
            expand=True,
            pad_edge=False,
            padding=(0,0,0,0),
            show_edge=False,
        )
        console = Console()
        with console.capture() as capture:
            pprint(self._model, console=console)
        table1.add_row(capture.get())

        table2 = Table(
            'Hyperparams',
            box=box.MINIMAL,
            expand=True,
            pad_edge=False,
            padding=(0,0,0,0),
            show_edge=False,
        )
        table2.add_row(
            self._model.__doc__
        )

        grid = Table(
            title=self.name, 
            title_justify="left",
            expand=True,
            box=None,
            show_header=False,
            show_footer=False,
            pad_edge=False,
            padding=(0,0,0,0),
            show_edge=False,
        )
        grid.add_column('')
        grid.add_row()
        grid.add_row(table1)
        grid.add_row(table2)
        return grid


        return "No documentation available."

    def _fit_label_encoder(self, labels) -> LabelEncoder:
        label_encoder = LabelEncoder()
        label_encoder.fit(self.labels)
        return label_encoder

    def encode_y(self, y: str | Iterable[str]) -> NDArray:
        if isinstance(y, str):
            y = [y]
        return self._label_encoder.transform(y)

    def decode_y(self, y: int | Iterable[int]) -> NDArray:
        if isinstance(y, int):
            y = [y]
        return self._label_encoder.inverse_transform(y)

    def fit(self, X: NDArray, y: NDArray) -> NDArray:
        self._model.fit(X, self.encode_y(y))
        return self.decode_y(self._model.predict(X))

    def predict(self, X) -> NDArray:
        return self.decode_y(self._model.predict(X))

    @classmethod
    def load(cls, path: pathlib.Path, echo: bool = False) -> MLModel:
        path = pathlib.Path(path)
        if path.is_dir():
            path /= "tcbench_model.pkl"
        return fileutils.load_pickle(path, echo=echo)

    def save(self, save_to: pathlib.Path, echo: bool = False) -> MLModel:
        save_to = pathlib.Path(save_to)
        if not save_to.exists():
            save_to.mkdir(parents=True)
        fileutils.save_pickle(
            self, save_to / "tcbench_mlmodel.pkl", echo=echo
        )
        fileutils.save_yaml(
            self.hyperparams, 
            save_to / "hyperparams.yml",
            echo=echo
        )
        fileutils.save_yaml(
            self._params,
            save_to / "params.yml",
            echo=echo
        )
        return self

    @property
    def size(self) -> int:
        return -1


class MLTester:
    def __init__(
        self,
        model: MLModel,
        dataloader: MLDataLoader,
        split_index: int,
        save_to: pathlib.Path | None = None,
        name: str = ""
    ):
        self.model = model
        self.dataloader = dataloader
        self.save_to = save_to
        self.name = name
        self.split_index = split_index

    def on_test_loop_iteration_end(
        self, 
        model: MLModel,
        split_data: SplitData, 
        y_pred: NDArray,
    ) -> ClassificationResults:
        name = self.name
        if name == "":
            name = "test"
        elif name.startswith("test_"):
            name = f"test_{self.name}"
        clsres = ClassificationResults(
            df_feat=split_data.df_test_feat,
            labels=split_data.labels,
            y_true=split_data.y_test,
            y_pred=y_pred,
            split_index=split_data.split_index,
            name=name,
            model=model,
        )
        if self.save_to:
            clsres.save(self.save_to, name=name, echo=False)
        return clsres

    def test_loop(self, model: MLModel, split_data: SplitData) -> ClassificationResults:
        y_pred = model.predict(split_data.X_test)
        return self.on_test_loop_iteration_end(model, split_data, y_pred)

    def test(self) -> ClassificationResults:
        split_data = self.dataloader.test_loader(self.split_index)
        return self.test_loop(self.model, split_data)


class MLTrainer(MLTester):
    def __init__(
        self,
        model: MLModel,
        dataloader: MLDataLoader,
        split_index: int,
        save_to: pathlib.Path | None = None,
        name: str = "",
        evaluate_train: bool = True,
        evaluate_test: bool = True,
    ):
        super().__init__(
            model=model,
            dataloader=dataloader,
            save_to=save_to,
            name=name,
            split_index=split_index,
        )
        self.evaluate_train = evaluate_train
        self.evaluate_test = evaluate_test

    def on_train_loop_iteration_end(
        self, 
        model: MLModel,
        split_data: SplitData, 
        y_pred: NDArray,
    ) -> ClassificationResults:
        name = self.name
        if name == "":
            name = "train"
        elif name.startswith("train_"):
            name = f"train_{self.name}"
        clsres = ClassificationResults(
            df_feat=split_data.df_train_feat,
            labels=split_data.labels,
            y_true=split_data.y_train,
            y_pred=y_pred,
            split_index=split_data.split_index,
            name=self.name,
            model=model,
        )
        if self.save_to:
            clsres.save(self.save_to, name=name, echo=False)
        return clsres

    def train_loop(
        self, 
        model: MLModel, 
        split_data: SplitData
    ) -> Tuple[ClassificationResults | None, ClassificationResults]:
        if split_data.X_train is None or split_data.y_train is None:
            raise RuntimeError("SplitData has None X_train or y_train")
        y_pred = model.fit(
            split_data.X_train,
            split_data.y_train
        )

        clsres_train = None
        if self.evaluate_train: 
            clsres_train = self.on_train_loop_iteration_end(model, split_data, y_pred)
        clsres_test = self.test_loop(model, split_data)

        return clsres_train, clsres_test
        
    def fit(self) -> Tuple[ClassificationResults | None, ClassificationResults]:
        split_data = self.dataloader.train_test_loader(self.split_index)
        clsres_train, clsres_test = self.train_loop(self.model, split_data)
        return clsres_train, clsres_test

    def save(self, echo: bool = False) -> None:
        if self.save_to is not None:
            self.model.save(self.save_to, echo)
