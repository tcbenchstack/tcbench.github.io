from __future__ import annotations
import rich_click as click

from typing import Iterable, Dict, Tuple
import pathlib

from tcbench import (
    DATASET_NAME,
    DATASET_TYPE,
)
from tcbench.cli import clickutils
#from tcbench.cli.clickutils import (
#    CLICK_CHOICE_DATASET_NAME,
#    CLICK_CHOICE_DATASET_TYPE,
#    CLICK_CHOICE_MODELING_METHOD_NAME,
#    CLICK_PARSE_DATASET_NAME,
#    CLICK_PARSE_DATASET_TYPE,
#    CLICK_PARSE_MODELING_METHOD_NAME,
#    CLICK_PARSE_STR_TO_LIST_INT,
#)
from tcbench.modeling.ml import loops
from tcbench.modeling.ml.core import MultiClassificationResults
from tcbench.modeling import (
    MODELING_FEATURE,
    MODELING_METHOD_NAME,
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
@click.argument(
    "method_hyperparams",
    nargs=-1,
    type=click.UNPROCESSED,
    callback=clickutils.parse_remainder,
)
@click.pass_context
def run(
    ctx, 
    dataset_name: DATASET_NAME, 
    dataset_type: DATASET_TYPE,
    method_name: MODELING_METHOD_NAME,
    series_len: int,
    save_to: pathlib.Path,
    num_workers: int,
    split_indices: Iterable[int],
    method_hyperparams: Dict[str, Tuple[Any]],
) -> Iterable[MultiClassificationResults]:
    """Run an experiment or campaign."""
    return loops.train_loop(
        dataset_name=dataset_name,
        dataset_type=dataset_type,
        method_name=method_name,
        series_len=series_len,
        features=(
            MODELING_FEATURE.PKTS_SIZE, 
            MODELING_FEATURE.PKTS_DIR,
        ),
        save_to=save_to,
        num_workers=num_workers,
        split_indices=split_indices,
        method_hyperparams=method_hyperparams,
    )

