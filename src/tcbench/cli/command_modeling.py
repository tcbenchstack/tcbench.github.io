from __future__ import annotations

import rich_click as click

from typing import Iterable, Dict, Tuple, Any, List
import pathlib
import sys

from tcbench import (
    DATASET_NAME,
    DATASET_TYPE,
)
from tcbench.cli import clickutils
from tcbench.modeling.ml import loops
from tcbench.modeling import (
    MODELING_FEATURE,
    MODELING_METHOD_NAME,
    factory,
)

@click.group()
@click.pass_context
def modeling(ctx):
    """Handle modeling experiments."""
    pass


@modeling.command(name="run")
@click.option(
    "--dset-name",
    "-d",
    "dataset_name",
    required=False,
    type=clickutils.CHOICE_DATASET_NAME,
    callback=clickutils.parse_dataset_name,
    help="Dataset name.",
    default=None,
)
@click.option(
    "--dset-type",
    "-t",
    "dataset_type",
    required=False,
    type=clickutils.CHOICE_DATASET_TYPE,
    callback=clickutils.parse_dataset_type,
    help="Dataset type.",
    #default=click.Choice(DATASET_TYPE.CURATE),
)
@click.option(
    "--method",
    "-m",
    "method_name",
    required=True,
    type=clickutils.CHOICE_MODELING_METHOD_NAME,
    callback=clickutils.parse_modeling_method_name,
    help="Modeling method.",
    default=None,
)
@click.option(
    "--series-len",
    "-s",
    "series_len",
    required=True,
    type=int,
    help="Clip packet series to the specified length.",
    default=10,
)
@click.option(
    "--output-folder",
    "-o",
    "save_to",
    required=False,
    default=pathlib.Path("./model"),
    type=pathlib.Path,
    help="Output folder."
)
@click.option(
    "--workers",
    "-w",
    "num_workers",
    required=False,
    default=1,
    type=int,
    help="Number of parallel workers."
)
@click.option(
    "--split-indices",
    "-i",
    "split_indices",
    required=False,
    default="",
    type=tuple,
    callback=clickutils.parse_raw_text_to_list_int,
    help="List of splits to use.",
)
@click.option(
    "--features",
    "-f",
    "features",
    required=True,
    type=clickutils.CHOICE_MODELING_FEATURE,
    callback=clickutils.parse_raw_text_to_list,
    help="List of features to use.",
)
@click.option(
    "--track-train",
    "-t",
    "track_train",
    required=False,
    is_flag=True,
    default=False,
    help="If enabled, save performance information about train splits.",
)
@click.option(
    "--name",
    "-n",
    "run_name",
    required=False,
    type=str,
    default="",
    help="Name of the run.",
)
@click.option(
    "--seed",
    "-s",
    "seed",
    required=False,
    type=int,
    default=1,
    help=\
        "Seed for dataprep and model initialization. "
        "The value specified is summmed to the split index."
)
@click.argument(
    "hyperparams_grid",
    nargs=-1,
    type=click.UNPROCESSED,
    callback=clickutils.parse_remainder,
)
@clickutils.save_commandline('save_to')
@click.pass_context
def run(
    ctx, 
    dataset_name: DATASET_NAME, 
    dataset_type: DATASET_TYPE,
    method_name: MODELING_METHOD_NAME,
    series_len: int,
    save_to: pathlib.Path,
    num_workers: int,
    split_indices: List[int],
    features: List[MODELING_FEATURE],
    track_train: bool,
    run_name: str,
    seed: int,
    hyperparams_grid: Dict[str, Tuple[Any]],
) -> None:
    """Run a campaign."""

    loops.train_loop(
        dataset_name=dataset_name,
        method_name=method_name,
        series_len=series_len,
        features=features,
        save_to=save_to,
        num_workers=num_workers,
        split_indices=split_indices,
        hyperparams_grid=hyperparams_grid,
        track_train=track_train,
        name=run_name,
        seed=seed,
    )


@modeling.command(name="docs")
@click.option(
    "--method",
    "-m",
    "method_name",
    required=True,
    type=clickutils.CHOICE_MODELING_METHOD_NAME,
    callback=clickutils.parse_modeling_method_name,
    help="Modeling method.",
    default=None,
)
def hyperparam_docs(
    method_name: MODELING_METHOD_NAME,
):
    """Shows hyper parameters documentations for modelingn algorithms."""

    from tcbench.cli.richutils import console

    mdl = factory.mlmodel_factory(
        name=method_name,
        labels=["a"],
        features=[MODELING_FEATURE.PKTS_SIZE],
        seed=1,
    )

    console.print(mdl.__rich_docs__())
