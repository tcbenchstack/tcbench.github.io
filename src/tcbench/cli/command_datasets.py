from __future__ import annotations
import rich_click as click

import pathlib
import shutil

#from tcbench.libtcdatasets import datasets_utils
import tcbench
from tcbench import cli
from tcbench.cli import clickutils


@click.group()
@click.pass_context
def datasets(ctx):
    """Install/Remove traffic classification datasets."""
    pass


@datasets.command(name="info")
@click.pass_context
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
def info(ctx, dataset_name):
    """Show the meta-data related to supported datasets."""
    catalog = tcbench.get_datasets_catalog()
    if dataset_name is not None:
        cli.logger.log(catalog[dataset_name])
    else:
        cli.logger.log(catalog)


@datasets.command(name="install")
@click.pass_context
@click.option(
    "--dset-name",
    "-d",
    "dataset_name",
    required=True,
    type=clickutils.CHOICE_DATASET_NAME,
    callback=clickutils.parse_dataset_name,
    help="Dataset name.",
)
@click.option(
    "--no-download",
    "-D",
    "no_download",
    type=click.BOOL,
    default=False,
    is_flag=True,
    help="Avoid (re)downloading the raw data.",
)
#@click.option(
#    "--input-folder",
#    "-i",
#    "input_folder",
#    required=False,
#    type=pathlib.Path,
#    default=None,
#    help="Folder where to find pre-downloaded tarballs.",
#)
@click.option(
    "--num-workers",
    "-w",
    "num_workers",
    required=False,
    type=int,
    default=-1,
    help=(
        "Number of parallel workers to use when processing the data. "
        "Default: use all available cores."
    ),
)
def install(
    ctx, 
    dataset_name: tcbench.DATASET_NAME, 
    no_download: bool,
    num_workers: int,
):
    """Install a dataset."""
    dset = tcbench.get_dataset(dataset_name)
    dset.install(no_download, num_workers)


#def _ls_files(dataset_name):
#    rich_obj = datasets_utils.get_rich_tree_parquet_files(dataset_name)
#    cli.console.print(rich_obj)
#
#
#@datasets.command(name="lsfiles")
#@click.pass_context
#@click.option(
#    "--name",
#    "-n",
#    "dataset_name",
#    required=False,
#    type=CLICK_TYPE_DATASET_NAME,
#    callback=CLICK_CALLBACK_DATASET_NAME,
#    default=None,
#    help="Dataset name.",
#)
#def lsfiles(ctx, dataset_name):
#    """Tree view of the datasets parquet files."""
#    _ls_files(dataset_name)
#

@datasets.command(name="schema")
@click.pass_context
@click.option(
    "--dset-name",
    "-d",
    "dataset_name",
    required=True,
    type=clickutils.CHOICE_DATASET_NAME,
    callback=clickutils.parse_dataset_name,
    help="Dataset name.",
    default=None,
)
@click.option(
    "--dset-type",
    "-t",
    "dataset_type",
    required=True,
    type=clickutils.CHOICE_DATASET_TYPE,
    callback=clickutils.parse_dataset_type,
    help="Dataset type.",
)
def schema(ctx, dataset_name, dataset_type):
    """Show dataset schema."""
    dset_schema = tcbench.get_dataset_schema(dataset_name, dataset_type)
    cli.console.print(dset_schema)


#@datasets.command(name="samples-count")
#@click.pass_context
#@click.option(
#    "--name",
#    "-n",
#    "dataset_name",
#    required=False,
#    type=CLICK_TYPE_DATASET_NAME,
#    callback=CLICK_CALLBACK_DATASET_NAME,
#    default=None,
#    help="Dataset to install.",
#)
#@click.option(
#    "--min-pkts",
#    "min_pkts",
#    required=False,
#    type=click.Choice(("-1", "10", "1000")),
#    default="-1",
#    show_default=True,
#    help="",
#)
#@click.option(
#    "--split",
#    "split",
#    required=False,
#    type=click.Choice(("human", "script", "0", "1", "2", "3", "4")),
#    default=None,
#    help="",
#)
#def report_samples_count(ctx, dataset_name, min_pkts, split):
#    """Show report on number of samples per class."""
#    with cli.console.status("Computing...", spinner="dots"):
#        min_pkts = int(min_pkts)
#        if min_pkts == -1 and split is not None:
#            if dataset_name != datasets_utils.DATASETS.UCDAVISICDM19:
#                min_pkts = 10
#
#        df_split = None
#        if dataset_name == datasets_utils.DATASETS.UCDAVISICDM19 or split is None:
#            df = datasets_utils.load_parquet(dataset_name, min_pkts, split)
#        else:
#            df = datasets_utils.load_parquet(dataset_name, min_pkts, split=None)
#            df_split = datasets_utils.load_parquet(dataset_name, min_pkts, split=split)
#
#    title = "unfiltered"
#    if dataset_name == datasets_utils.DATASETS.UCDAVISICDM19:
#        if split is not None:
#            title = f"filtered, split: {split}"
#    else:
#        title = []
#        if min_pkts != -1:
#            title.append(f"min_pkts: {min_pkts}")
#        if split:
#            title.append(f"split: {split}")
#        if title:
#            title = ", ".join(title)
#        else:
#            title = "unfiltered"
#
#    if df_split is None:
#        if (
#            dataset_name == datasets_utils.DATASETS.UCDAVISICDM19
#            and min_pkts == -1
#            and split is None
#        ):
#            ser = df.groupby(["partition", "app"])["app"].count()
#        else:
#            ser = df["app"].value_counts()
#
#        richutils.rich_samples_count_report(ser, title=title)
#    else:
#        richutils.rich_splits_report(df, df_split, split_index=split, title=title)
#
#
#@datasets.command(name="import")
#@click.pass_context
#@click.option(
#    "--name",
#    "-n",
#    "dataset_name",
#    required=True,
#    type=click.Choice([DATASETS.UCDAVISICDM19.value, DATASETS.UTMOBILENET21.value], case_sensitive=False),
#    default=None,
#    help="Dataset name.",
#)
#@click.option(
#    "--archive",
#    "path_archive",
#    required=False,
#    type=pathlib.Path,
#    default=None,
#    help="Path of an already downloaded curated archive.",
#)
#def import_datasets(ctx, dataset_name, path_archive):
#    """Fetch and install the curated version of the dataset."""
#    datasets_utils.import_dataset(dataset_name, path_archive)
#    cli.console.print()
#    cli.console.print("Files installed")
#    _ls_files(dataset_name)
#
#
#@datasets.command(name="delete")
#@click.pass_context
#@click.option(
#    "--name",
#    "-n",
#    "dataset_name",
#    required=False,
#    type=CLICK_TYPE_DATASET_NAME,
#    callback=CLICK_CALLBACK_DATASET_NAME,
#    default=None,
#    help="Dataset to delete.",
#)
#def delete_dataset(ctx, dataset_name):
#    """Delete a dataset."""
#    folder = datasets_utils.get_dataset_folder(dataset_name)
#    if not folder.exists():
#        cli.console.print(f"[red]Dataset {dataset_name} is not installed[/red]")
#    else:
#        with cli.console.status(f"Deleting {dataset_name}...", spinner="dots"):
#            shutil.rmtree(str(folder))
