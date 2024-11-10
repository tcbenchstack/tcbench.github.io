from __future__ import annotations

import itertools
import pathlib
import multiprocessing

from typing import Iterable, Any, Dict, Tuple, List

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
    ClassificationResults
)

DEFAULT_TRACK_EXTRA_COLUMNS = (
    COL_BYTES, 
    COL_PACKETS, 
    COL_ROW_ID
)

def _flatten_hyperparams_grid(
    hyperparams_grid: Dict[str, Tuple[Any]] | None
) -> List[Dict[str, Any]]:
    if hyperparams_grid is None:
        return [dict()]

    params_name = sorted(hyperparams_grid.keys())
    values = [hyperparams_grid[par] for par in params_name]
    res = []
    for params_value in itertools.product(*values):
        res.append(
            dict(zip(params_name, params_value))
        )
    return res


def _load_dataset(
    dataset_name: DATASET_NAME,
    features: Iterable[MODELING_FEATURE],
    series_len: int = 30,
    y_colname: str = COL_APP,
    index_colname: str = COL_ROW_ID,
    extra_colnames: Iterable[str] | None = DEFAULT_TRACK_EXTRA_COLUMNS,
    echo: bool = True
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
        echo=echo,
    )
    return dset

def _train_loop_worker(
    trainer: MLTrainer
) -> Tuple[MLTrainer, ClassificationResults | None, ClassificationResults]:
    return trainer, *trainer.fit()

def _train_loop_single(
    dataloader: MLDataLoader,
    method_name: MODELING_METHOD_NAME,
    features: Iterable[MODELING_FEATURE],
    seed: int = 1,
    save_to: pathlib.Path | None = None,
    track_train: bool = False,
    num_workers: int = 1,
    hyperparams: Dict[str, Any] | None = None,
    with_progress: bool = True,
    name: str = "",
):
    if save_to is not None and not save_to.exists():
        save_to.mkdir(parents=True)

    if hyperparams is None:
        hyperparams = dict()

    models = [
        mlmodel_factory(
            method_name,
            labels=dataloader.labels,
            features=features,
            seed=seed + split_index,
            **hyperparams,
        )
        for split_index in range(dataloader.num_splits)
    ]

    trainers = []
    with richutils.Progress(
        total=dataloader.num_splits,
        description="Prepare trainers..."
    ) as progress:
        for model, split_index in zip(
            models,
            dataloader.split_indices
        ):
            subfolder = None
            if save_to is not None:
                subfolder = save_to / f"split_{split_index:02d}"
            trainers.append(
                MLTrainer(
                    model,
                    dataloader,
                    split_index,
                    save_to=subfolder,
                    name=name,
                    evaluate_train=track_train,
                    evaluate_test=True
                )
            )
            progress.update()

    num_workers = min(num_workers, multiprocessing.cpu_count())

    table_columns = [
        richutils.LiveTableColumn(
            name="split_index"
        ),
        richutils.LiveTableColumn(
            name="f1_train", fmt=".5f", track="max", summary="avg"
        ),
        richutils.LiveTableColumn(
            name="f1_test", fmt=".5f", track="max", summary="avg"
        ),
    ]
    if not track_train:
        table_columns.pop(1)

    with (
        richutils.ProgressLiveTable(
            columns=table_columns,
            total=dataloader.num_splits,
            description="Train...",
            visible=with_progress,
        ) as progress,
        multiprocessing.Pool(
            processes=num_workers, 
            maxtasksperchild=1
        ) as pool,
    ):
        for trainer, clsres_train, clsres_test in (
            pool.imap_unordered(_train_loop_worker, trainers)
        ): 
            trainer.save()
            row = dict(
                split_index=clsres_test.split_index,
                f1_test=clsres_test.weighted_f1,
            )
            if clsres_train is not None:
                row["f1_train"] = clsres_train.weighted_f1
            progress.add_row(**row)

def train_loop(
    dataset_name: DATASET_NAME,
    method_name: MODELING_METHOD_NAME,
    features: Iterable[MODELING_FEATURE],
    series_len: int = 30,
    seed: int = 1,
    save_to: pathlib.Path | None = None,
    track_train: bool = False,
    num_workers: int = 1,
    split_indices: List[int] | None = None,
    hyperparams_grid: Dict[str, Tuple[Any]] | None = None,
    y_colname: str = COL_APP,
    index_colname: str = COL_ROW_ID,
    extra_colnames: Iterable[str] = DEFAULT_TRACK_EXTRA_COLUMNS,
    with_progress: bool = True,
    name: str = "",
) -> Any:

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
        seed=seed,
    )

    for idx, hyperparams in enumerate(
        _flatten_hyperparams_grid(hyperparams_grid), 
        start=1
    ):
        subfolder = save_to
        if save_to is not None:
            subfolder = save_to / f"grid_{idx:03d}"

        _train_loop_single(
            dataloader=dataloader,
            method_name=method_name,
            features=features,
            seed=seed,
            save_to=subfolder,
            track_train=track_train,
            num_workers=num_workers,
            hyperparams=hyperparams,
            with_progress=with_progress,
            name=name,
        )

