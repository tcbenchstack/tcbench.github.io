from __future__ import annotations

import polars as pl
import itertools
import multiprocessing
import os

from typing import Iterable

from tcbench import (
    DATASET_NAME,
    DATASET_TYPE,
    get_dataset,
)
from tcbench.core import MultiprocessingWorkerKWArgs
from tcbench.cli import richutils
from tcbench.datasets.core import Dataset
from tcbench.modeling import (
    MODELING_METHOD_NAME, 
    mlmodel_factory,
)
from tcbench.modeling.columns import (
    COL_SPLIT_INDEX,
    COL_BYTES,
    COL_PACKETS,
    COL_ROW_ID,
)
from tcbench.modeling.ml.core import (
    MLDataLoader,
    MLTrainer,
    ClassificationResults,
    MultiClassificationResults,
    compose_hyperparams_grid,
)

DEFAULT_TRACK_EXTRA_COLUMNS = (
    COL_BYTES, 
    COL_PACKETS, 
    COL_ROW_ID
)


def _train_loop_worker(params: MultiprocessingWorkerKWArgs) -> MultiClassificationResults:
#    dataset_name: DATASET_NAME,
#    dataset_type: DATASET_TYPE,
#    method_name: MODELING_METHOD_NAME,
#    features: Iterable[str],
#    series_len: int = 10,
#    series_pad: int = None,
#    seed: int = 1,
#    save_to: pathlib.Path = None,
#    split_index: int = 1,
#    track_train: bool = False,
#    track_extra_columns: Iterable[str] = DEFAULT_TRACK_EXTRA_COLUMNS,
#) -> MultiClassificationResults:
    track_extra_columns = params.track_extra_columns
    if track_extra_columns is None:
        track_extra_columns = []

    dset = get_dataset(params.dataset_name)

    columns = [dset.y_colname, dset.index_colname]
    for col in itertools.chain(params.features, track_extra_columns):
        col = str(col)
        if col not in columns:
            columns.append(col)

    dset.load(
        params.dataset_type, 
        min_packets=params.series_len,
        columns=columns,
        echo=params.echo,
    )

    ldr = MLDataLoader(
        dset,
        features=params.features,
        df_splits=dset.df_splits,
        split_index=params.split_index,
        series_len=params.series_len,
        series_pad=params.series_pad,
        extra_colnames=track_extra_columns,
        shuffle_train=True,
        seed=params.seed,
    )

    mdl = mlmodel_factory(
        params.method_name,
        labels=ldr.labels,
        feature_names=ldr.feature_names,
        seed=params.seed,
        **params.method_hyperparams,
    )

    trainer = MLTrainer(mdl, ldr)

    clsres_train = trainer.fit(name="train")
    clsres_test = trainer.predict(name="test")
    if params.save_to:
        if params.track_train:
            clsres_train.save(params.save_to, echo=params.echo)
        clsres_test.save(params.save_to, echo=params.echo)
        mdl.save(params.save_to, echo=params.echo)

    clsres = MultiClassificationResults(
        train=clsres_train,
        test=clsres_test,
        model=mdl,
    )

    return clsres


def train_loop(
    dataset_name: DATASET_NAME,
    dataset_type: DATASET_TYPE,
    method_name: MODELING_METHOD_NAME,
    features: Iterable[str],
    series_len: int = 10,
    seed: int = 1,
    save_to: pathlib.Path = None,
    track_train: bool = False,
    num_workers: int = 1,
    split_indices: Iterable[int] = None,
    method_hyperparams: Dict[str, Any] = None,
) -> Iterable[MultiClassificationResults]:
    if split_indices is None:
        dset = get_dataset(dataset_name).load(dataset_type, lazy=True, echo=False)
        split_indices = (
            dset
            .df_splits
            .select(pl.col(COL_SPLIT_INDEX))
            .collect()
            .to_series()
            .to_list()
        )

    if method_hyperparams is None:
        method_hyperparams = dict()

    hyperparams_grid = compose_hyperparams_grid(method_hyperparams)

    for hyperparams_idx, hyperparams_curr in enumerate(hyperparams_grid, start=1):
        with (
            richutils.Progress(
                description="Train...", 
                total=len(split_indices)
            ) as progress,
#            multiprocessing.get_context("spawn").Pool(
#                #processes=num_workers, 
#                processes=2,
#                maxtasksperchild=1
#            ) as pool,
        ):
            params = [
                MultiprocessingWorkerKWArgs(
                    dataset_name=dataset_name,
                    dataset_type=dataset_type,
                    method_name=method_name,
                    method_hyperparams=hyperparams_curr,
                    features=features,
                    series_len=series_len,
                    series_pad=None,
                    seed=split_index,
                    save_to=(
                        save_to / f"split_{split_index:02d}" 
                        if save_to 
                        else None
                    ),
                    split_index=split_index,
                    track_train=track_train,
                    track_extra_columns=DEFAULT_TRACK_EXTRA_COLUMNS,
                    echo=False,
                )
                for split_index in split_indices
            ]
#            for clsres in pool.imap_unordered(_train_loop_worker, params):
            for p in params:
                _train_loop_worker(p)
                progress.update()
#    return _train_loop_worker(
#        dataset_name=dataset_name,
#        dataset_type=dataset_type,
#        method_name=method_name,
#        features=features,
#        series_len=series_len,
#        seed=seed,
#        save_to=save_to,
#        track_train=track_train
#    )
