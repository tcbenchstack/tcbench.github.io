from __future__ import annotations

import polars as pl

import pathlib
import itertools

from typing import Tuple, List, Dict, Any, Iterable, Callable
from numpy.typing import NDArray
from sklearn.preprocessing import LabelEncoder

from dataclasses import dataclass

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
)


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
        y_true: NDArray = None,
        y_pred: NDArray = None,
        split_index: int = None,
        name: str = "test",
        with_reports: bool = True,
        model: MLModel = None,
    ):
        self.labels = labels
        self.model = model
        self.name = name
        self.df_feat = df_feat

        if y_true is not None:
            self.df_feat = self.df_feat.with_columns(
                y_true=pl.Series(y_true),
                y_pred=pl.Series(y_pred),
            )
        self.df_feat = self.df_feat.with_columns(
            split_index=pl.Series(split_index) if split_index else None
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

    @property
    def f1(self) -> NDArray:
        if self.classification_report is None:
            return None
        return (
            self.classification_report
            .filter(pl.col("label") == "weighted avg")
            ["f1-score"]
            .to_numpy()
            [0]
        )

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
        name: str = "test", 
        with_progress: bool = True, 
        echo: bool = False
    ) -> ClassificationResults:
        save_to = pathlib.Path(save_to)
        fileutils.create_folder(save_to, echo=echo)

        with richutils.SpinnerProgress("Saving...", visible=echo):
            self.df_feat.write_parquet(save_to / f"{name}_df_feat.parquet")
            if self.confusion_matrix is not None:
                fileutils.save_csv(
                    self.confusion_matrix,
                    save_to / f"{name}_confusion_matrix.csv",
                    echo=echo
                )
                fileutils.save_csv(
                    self.confusion_matrix_normalized,
                    save_to / f"{name}_confusion_matrix_normalized.csv",
                    echo=echo,
                )
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
            error_policy="raise"
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

@dataclass
class MultiClassificationResults:
    train: ClassificationResults = None
    val: ClassificationResults = None
    test: ClassificationResults = None
    model: MLModel = None

    @classmethod
    def load(
        cls, 
        folder: pathlib.Path, 
        name_train: str = "train",
        name_test: str = "test",
        name_val: str = "val",
        echo: bool = False,
    ) -> MultiClassificationResults:
        folder = pathlib.Path(folder)

        clsres = MultiClassificationResults(
            model = MLModel.load(folder)
        )
        if name_train and (folder / f"{name_train}_df_feat.parquet").exists():
            clsres.train = ClassificationResults.load(
                folder, name=name_train, echo=echo
            )
        if name_test and (folder / f"{name_test}_df_feat.parquet").exists():
            clsres.test = ClassificationResults.load(
                folder, name=name_test, echo=echo
            )
        if name_val and (folder / f"{name_val}_df_feat.parquet").exists():
            clsres.val = ClassificationResults.load(
                folder, name=name_val, echo=echo
            )
        return clsres

    @property
    def f1_train(self) -> float:
        if self.train is not None:
            return self.train.f1
        return None

    @property
    def f1_test(self) -> float:
        if self.test is not None:
            return self.test.f1
        return None

    @property
    def f1_val(self) -> float:
        if self.val is not None:
            return self.val.f1
        return None


class MLDataLoader:
    def __init__(
        self,
        dset: Dataset,
        features: Iterable[str],
        df_splits: pl.DataFrame,
        split_index: int = 1,
        y_colname: str = COL_APP,
        index_colname: str = COL_ROW_ID,
        series_len: int = None,
        series_pad: int = None,
        extra_colnames: Iterable[str] = DEFAULT_EXTRA_COLUMNS,
        shuffle_train: bool = True,
        seed: int = 1
    ):
        self.dset = dset
        self.y_colname = y_colname
        self.index_colname = index_colname
        self._labels = dset.df[y_colname].unique().sort().to_list()
        self.df_splits = df_splits
        self.split_index = split_index
        self.features = features
        self.extra_colnames = extra_colnames
        self.series_len = series_len
        self.series_pad = series_pad
        self.shuffle_train = shuffle_train
        self.seed = seed

        self._df_train, self._df_test = splitting.get_train_test_splits(
            self.dset.df,
            self.df_splits,
            self.split_index,
            self.index_colname
        )

        self._X_train, self._y_train, self._df_train_feat = \
            self.dataprep(
                self._df_train,
                shuffle=shuffle_train,
                seed=seed
            )
        self._X_test, self._y_test, self._df_test_feat = \
            self.dataprep(
                self._df_test,
                shuffle=False,
            )

        self._feature_names = [
            col
            for col in self._df_train_feat.columns
            if col not in (self.y_colname, *self.extra_colnames)
        ]

    @property
    def labels(self) -> List[str]:
        return self._labels

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names

    @property
    def train(self) -> pl.DataFrame:
        return self._df_train

    @property
    def train_feat(self) -> pl.DataFrame:
        return self._df_train_feat

    @property
    def X_train(self) -> NDArray:
        return self._X_train

    @property
    def y_train(self) -> NDArray:
        return self._y_train

    @property
    def test(self) -> pl.DataFrame:
        return self._df_test

    @property
    def X_test(self) -> NDArray:
        return self._X_test

    @property
    def y_test(self) -> NDArray:
        return self._y_test
        
    @property
    def test_feat(self) -> pl.DataFrame:
        return self._df_test_feat

    def dataprep(
        self, 
        df: pl.DataFrame, 
        shuffle: bool = False, 
        seed: int = 1
    ) -> Tuple[NDArray, NDArray, pl.DataFrame]:
        if shuffle:
            df = df.sample(
                fraction=1, 
                shuffle=True, 
                seed=seed,
            )
        return datafeatures.features_dataprep(
                df,
                self.features,
                self.series_len,
                self.series_pad,
                self.y_colname,
                self.extra_colnames,
            )



class MLModel:
    def __init__(
        self,
        labels: Iterable[str],
        feature_names: Iterable[str],
        model_class: Callable,
        seed: int = 1,
        **hyperparams: Dict[str, Any],
    ):
        self.labels = labels
        self.feature_names = feature_names
        self.hyperparams = hyperparams
        self.seed = seed

        self._label_encoder = self._fit_label_encoder(self.labels)
        self._model = model_class(**hyperparams)

    @property
    def name(self) -> str:
        self._model.__class__.__name__

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

    def predict(self, X) -> ClassificationResults:
        return self.decode_y(self._model.predict(X))

    @classmethod
    def load(cls, path: pathlib.Path, echo: bool = False) -> MLModel:
        path = pathlib.Path(path)
        if path.is_dir():
            path /= "tcbench_model.pkl"
        return fileutils.load_pickle(path, echo=echo)

    def save(self, save_to: pathlib.Path, echo: bool = False) -> MLModel:
        if save_to:
            save_to = pathlib.Path(save_to)
            fileutils.save_pickle(
                self, save_to / "tcbench_model.pkl", echo=echo
            )
        return self

    @property
    def size(self) -> int:
        return None


class MLTester:
    def __init__(
        self,
        model: MLModel,
        dataloader: MLDataLoader,
        split_index: int = None
    ):
        self.model = model
        self.dataloader = dataloader
        self.split_index = split_index

    def predict(
        self, 
        name: str = "test", 
        save_to: pathlib.Path = None, 
        echo: bool = False,
    ) -> ClassificationResults:
        y_pred = self.model.predict(
            self.dataloader.X_test,
        )
        clsres = ClassificationResults(
            df_feat=self.dataloader.test_feat,
            labels=self.model.labels,
            y_true=self.dataloader.y_test,
            y_pred=y_pred,
            split_index=self.split_index,
            name=name,
            model=self.model,
        )
        if save_to:
            clsres.save(
                save_to, name=name, echo=echo
            )
        return clsres

class MLTrainer(MLTester):
    def fit(
        self, 
        name: str = "train", 
        save_to: pathlib.Path = None,
        echo: bool = False,
    ) -> ClassificationResults:
        y_pred = self.model.fit(
            self.dataloader.X_train,
            self.dataloader.y_train
        )
        clsres = ClassificationResults(
            df_feat=self.dataloader.train_feat,
            labels=self.model.labels,
            y_true=self.dataloader.y_train,
            y_pred=y_pred,
            split_index=self.split_index,
            model=self.model,
            name=name,
        )
        if save_to:
            clsres.save(
                save_to, name=name, echo=echo
            )
        return clsres

