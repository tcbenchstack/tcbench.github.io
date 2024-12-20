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
    mlmodel_factory,
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
    metavar="DATASET_NAME",
    required=True,
    type=clickutils.CHOICE_DATASET_NAME,
    callback=clickutils.parse_dataset_name,
    help=clickutils.format_help_message(
        "Dataset name.",
        clickutils.CHOICE_DATASET_NAME.choices,
    ),
    default=None,
)
@click.option(
    "--dset-type",
    "-t",
    "dataset_type",
    metavar="DATASET_TYPE",
    required=False,
    type=clickutils.CHOICE_DATASET_TYPE,
    callback=clickutils.parse_dataset_type,
    default=str(DATASET_TYPE.CURATE),
    help=clickutils.format_help_message(
        "Dataset type.",
        clickutils.CHOICE_DATASET_TYPE.choices,
        default=str(DATASET_TYPE.CURATE)
    ),
)
@click.option(
    "--method",
    "-m",
    "method_name",
    metavar="METHOD_NAME",
    required=True,
    type=clickutils.CHOICE_MODELING_METHOD_NAME,
    callback=clickutils.parse_modeling_method_name,
    help=clickutils.format_help_message(
        "Modeling method.",
        choices=clickutils.CHOICE_MODELING_METHOD_NAME.choices,
    )
)
@click.option(
    "--series-len",
    "-s",
    "series_len",
    required=False,
    type=int,
    help=clickutils.format_help_message(
        "Clip packet series to the specified length.",
        default=str(10),
    ),
    default=10,
)
@click.option(
    "--output-folder",
    "-o",
    "save_to",
    required=False,
    default=pathlib.Path("./model"),
    type=pathlib.Path,
    help=clickutils.format_help_message(
        "Output folder.",
        default="./model"
    ),
)
@click.option(
    "--workers",
    "-w",
    "num_workers",
    required=False,
    default=1,
    type=int,
    help=clickutils.format_help_message(
        "Number of parallel workers.",
        default=str(1),
    )
)
@click.option(
    "--split-indices",
    "-i",
    "split_indices",
    required=False,
    default="",
    type=tuple,
    callback=clickutils.parse_raw_text_to_list_int,
    help=clickutils.format_help_message(
        "List of splits to use.",
        default="Use all predefined splits"
    )
)
@click.option(
    "--features",
    "-f",
    "features",
    required=True,
    type=list,
    callback=clickutils.parse_list_modeling_feature,
    help=clickutils.format_help_message(
        "List of features to use.",
        choices=clickutils.CHOICE_MODELING_FEATURE.choices,
    ),
)
@click.option(
    "--save-train",
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
    help=clickutils.format_help_message(
        "Name of the run.",
        default="No name is associated to the run"
    )
)
@click.option(
    "--seed",
    "-s",
    "seed",
    required=False,
    type=int,
    default=1,
    help=clickutils.format_help_message(
        "Seed for dataprep and model initialization. "
        "The value specified is summmed to the split index.",
        default=str(1)
    )
)
@click.option(
    "--dry-run",
    "-D",
    "dry_run",
    required=False,
    is_flag=True,
    default=False,
    help="Show a summary the modeling grid search (if any)"
)
@click.argument(
    "hyperparams_grid",
    nargs=-1,
    type=click.UNPROCESSED,
    callback=clickutils.parse_remainder,
)
@clickutils.save_commandline("save_to", cli_skip_option="dry_run")
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
    dry_run: bool,
    hyperparams_grid: Dict[str, Tuple[Any]],
) -> None:
    """Run a campaign."""

    if dry_run:
        loops.print_hyperparams_grid(hyperparams_grid)
        sys.exit(0)

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
def docs(
    method_name: MODELING_METHOD_NAME,
):
    """Shows hyper parameters documentations for modelingn algorithms."""

    from tcbench.cli.richutils import console

    mdl = mlmodel_factory(
        name=method_name,
        labels=["a"],
        features=[MODELING_FEATURE.PKTS_SIZE],
        seed=1,
    )

    console.print(mdl.__rich_docs__())
