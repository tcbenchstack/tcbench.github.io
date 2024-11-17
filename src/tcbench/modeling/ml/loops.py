from __future__ import annotations

import numpy as np

import itertools
import pathlib
import multiprocessing

from typing import Iterable, Any, Dict, Tuple, List

from tcbench.core import Pool1N, save_params
from tcbench import (
    DATASET_NAME,
    DATASET_TYPE,
    get_dataset,
    cli,
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
    COL_SPLIT_INDEX,
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
    echo: bool = False
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
        lazy=True,
    )
    return dset
        

@save_params("save_to", "split_index", echo=True)
def _trainer_init(
    dataset_name: DATASET_NAME,
    method_name: MODELING_METHOD_NAME,
    features: Iterable[MODELING_FEATURE],
    *,
    series_len: int = 30,
    seed: int = 1,
    save_to: pathlib.Path | None = None,
    track_train: bool = False,
    split_index: int = 1,
    hyperparams: Dict[str, Any] | None = None,
    y_colname: str = COL_APP,
    index_colname: str = COL_ROW_ID,
    extra_colnames: Iterable[str] = DEFAULT_TRACK_EXTRA_COLUMNS,
    name: str = "",
    num_workers: int = 1,
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
        split_indices=[split_index],
        y_colname=y_colname,
        index_colname=index_colname,
        series_len=series_len,
        series_pad=None,
        extra_colnames=extra_colnames,
        shuffle_train=True,
        seed=seed,
    )

    if save_to is not None and not save_to.exists():
        save_to.mkdir(parents=True)
    subfolder = None
    if save_to is not None:
        subfolder = save_to / f"split_{split_index:02d}"

    if hyperparams is None:
        hyperparams = dict()

    max_workers = multiprocessing.cpu_count() 

    model = mlmodel_factory(
        method_name,
        labels=dataloader.labels,
        features=features,
        seed=seed + split_index,
        num_workers=(
            max_workers 
            if num_workers == 1 else 
            int(np.ceil(max_workers / num_workers))
        ),
        hyperparams=hyperparams,
    )

    return MLTrainer(
        model,
        dataloader,
        split_index,
        save_to=subfolder,
        name=name,
        evaluate_train=track_train,
        evaluate_test=True
    )


def _trainer_loop_worker(
    trainer: MLTrainer
) -> Tuple[MLTrainer, ClassificationResults | None, ClassificationResults]:
    return trainer, *trainer.fit()


def _trainer_loop(
    trainers: List[MLTrainer],
    *,
    num_workers: int = 1,
    with_progress: bool = True,
    track_train: bool = True,
):
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
            total=len(trainers),
            description="Train...",
            visible=with_progress,
        ) as progress,
        Pool1N(
            processes=num_workers, 
            maxtasksperchild=1
        ) as pool,
    ):
        for trainer, clsres_train, clsres_test in (
            pool.imap_unordered(_trainer_loop_worker, trainers)
        ): 
            trainer.save()
            row = dict(
                split_index=clsres_test.split_index,
                f1_test=clsres_test.weighted_f1,
            )
            if clsres_train is not None:
                row["f1_train"] = clsres_train.weighted_f1
            progress.add_row(**row)

def print_hyperparams_grid(hyperparams_grid: Dict[str, Any] | None) -> None:
    from tcbench.cli.richutils import console
    from rich.table import Table
    from rich import box

    if hyperparams_grid is None or len(hyperparams_grid) == 0:
        console.print("No hyperparam grid!")
        return

    table = Table(
        title="Hyperparams Grid",
        title_justify="left",
        show_edge=False,
        pad_edge=True,
        box=box.HORIZONTALS,
        show_header=True,
        show_footer=True,
    )
    table.add_column("Name")
    table.add_column("Num.", justify="right")
    table.add_column("Values")
    total = 1
    for param_name in sorted(hyperparams_grid.keys()):
        param_value = hyperparams_grid[param_name]
        num_params = 1
        if isinstance(param_value, (list, set, tuple)):
            num_params = len(param_value)
        total *= num_params
        table.add_row(
            param_name,
            str(len(param_value)),
            str(param_value)
        )
    table.columns[0].footer = "Grid size"
    table.columns[1].footer = str(total)

    console.print()
    console.print(table)


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

    if split_indices is None:
        dset = _load_dataset(
            dataset_name=dataset_name,
            features=features,
            series_len=series_len,
            y_colname=y_colname,
            index_colname=index_colname,
            extra_colnames=extra_colnames,
            echo=False
        )
        split_indices = (
            dset
            .df_splits
            .select(COL_SPLIT_INDEX)
            .collect()
            .to_numpy()
            .squeeze()
            .tolist()
        )          

    _flat_grid = _flatten_hyperparams_grid(hyperparams_grid)
    for idx, hyperparams in enumerate(_flat_grid, start=1):
        trainers = []
        with richutils.Progress(
            total=len(split_indices),
            description="Prepare trainers..."
        ) as progress:
            for split_index in split_indices:
                trainer = _trainer_init(
                    dataset_name=dataset_name,
                    method_name=method_name,
                    features=features,
                    series_len=series_len,
                    seed=seed,
                    save_to=save_to,
                    track_train=track_train,
                    split_index=split_index,
                    hyperparams=hyperparams,
                    y_colname=y_colname,
                    index_colname=index_colname,
                    extra_colnames=extra_colnames,
                    name=name,
                    num_workers=num_workers,
                )
                trainers.append(trainer)
                progress.update()

        cli.logger.log(f"Grid ({idx}/{len(_flat_grid)}): {hyperparams}")
        _trainer_loop(
            trainers,
            num_workers=num_workers,
            with_progress=with_progress,
            track_train=track_train
        )
