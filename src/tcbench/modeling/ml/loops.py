from __future__ import annotations

import itertools
import pathlib
import multiprocessing

from typing import Iterable, Any, Dict

from tcbench import (
    DATASET_NAME,
    DATASET_TYPE,
    get_dataset,
)
from tcbench.datasets import Dataset
from tcbench.cli import richutils
from tcbench.modeling import (
    MODELING_METHOD_NAME, 
    MODELING_FEATURE,
    mlmodel_factory,
)
from tcbench.modeling.columns import (
    COL_BYTES,
    COL_PACKETS,
    COL_ROW_ID,
    COL_APP,
)
from tcbench.modeling.ml.core import (
    MLDataLoader,
    MLTrainer,
#    MultiClassificationResults,
#    compose_hyperparams_grid,
)

DEFAULT_TRACK_EXTRA_COLUMNS = (
    COL_BYTES, 
    COL_PACKETS, 
    COL_ROW_ID
)



def _load_dataset(
    dataset_name: DATASET_NAME,
    features: Iterable[MODELING_FEATURE],
    series_len: int = 30,
    y_colname: str = COL_APP,
    index_colname: str = COL_ROW_ID,
    extra_colnames: Iterable[str] = DEFAULT_TRACK_EXTRA_COLUMNS,
) -> Dataset:

    dset = get_dataset(dataset_name)

    if extra_colnames is None:
        extra_colnames = []
    columns = [y_colname, index_colname]
    for col in itertools.chain(features, extra_colnames):
        col = str(col)
        if col not in columns:
            columns.append(col)

    dset.load(
        DATASET_TYPE.CURATE,
        min_packets=series_len,
        columns=columns,
        echo=False,
    )
    return dset

def _train_loop_worker(trainer: MLTrainer):
    return trainer.fit()

def train_loop(
    dataset_name: DATASET_NAME,
    method_name: MODELING_METHOD_NAME,
    features: Iterable[MODELING_FEATURE],
    series_len: int = 30,
    seed: int = 1,
    save_to: pathlib.Path = None,
    track_train: bool = False,
    num_workers: int = 1,
    split_indices: Iterable[int] = None,
    method_hyperparams: Dict[str, Any] = None,
    y_colname: str = COL_APP,
    index_colname: str = COL_ROW_ID,
    extra_colnames: Iterable[str] = DEFAULT_TRACK_EXTRA_COLUMNS,
    with_progress: bool = True,
) -> Any:


    if method_hyperparams is None:
        method_hyperparams = dict()

    dset = _load_dataset(
        dataset_name=dataset_name,
        features=features,
        series_len=series_len,
        y_colname=y_colname,
        index_colname=index_colname,
        extra_colnames=extra_colnames
    )

    dataloader = MLDataLoader(
        dset,
        features=features,
        df_splits=dset.df_splits,
        split_indices=split_indices,
        y_colname=y_colname,
        index_colname=index_colname,
        series_len=series_len,
        series_pad=None,
        extra_colnames=extra_colnames,
        shuffle_train=True,
        seed=1,
    )

    models = [
        mlmodel_factory(
            method_name,
            labels=dataloader.labels,
            feature_names=features,
            seed=split_index,
            **method_hyperparams,
        )
        for split_index in range(dataloader.num_splits)
    ]


    with richutils.Progress(
        total=dataloader.num_splits,
        description="Prepare trainers..."
    ) as progress:
        trainers = []
        for model, split_index in zip(
            models,
            dataloader.split_indices
        ):
            trainers.append(
                MLTrainer(
                    model,
                    dataloader,
                    split_index,
                    save_to=None,
                    name="train",
                    evaluate_train=True,
                    evaluate_test=True
                )
            )
            progress.update()

    with (
        richutils.TrainerLivePerformance(
            columns=[
                richutils.TrainerPerformanceTableColumn(
                    name="split_index"
                ),
                richutils.TrainerPerformanceTableColumn(
                    name="f1_train", fmt=".5f", track="max", summary="avg"
                ),
                richutils.TrainerPerformanceTableColumn(
                    name="f1_test", fmt=".5f", track="max", summary="avg"
                ),
            ],
            total=dataloader.num_splits,
            description="Train...",
            visible=with_progress,
        ) as progress,
        multiprocessing.Pool(processes=2, maxtasksperchild=1) as pool,
    ):
        for clsres_train, clsres_test in pool.imap_unordered(_train_loop_worker, trainers): 
            progress.add_row(
                split_index=clsres_test.split_index,
                f1_train=clsres_train.weighted_f1,
                f1_test=clsres_test.weighted_f1,
            )
