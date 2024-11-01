from __future__ import annotations

from typing import Any

import polars as pl

import rich.console
import rich.tree as richtree
import rich.table as richtable
import rich.panel as richpanel
from rich import box as richbox

from collections import UserDict

from tcbench.datasets import (
    _mirage,
    _ucdavis,
    _utmobilenet,
    DATASET_NAME,
    DATASET_TYPE,
)
from tcbench.datasets.core import (
    Dataset, 
    DatasetSchema,
    DatasetMetadata
)

_DATASET_NAME_TO_CLASS = {
    DATASET_NAME.UCDAVIS19: _ucdavis.UCDavis19,
    DATASET_NAME.MIRAGE19: _mirage.Mirage19,
    DATASET_NAME.MIRAGE20: _mirage.Mirage20,
    DATASET_NAME.MIRAGE22: _mirage.Mirage22,
    DATASET_NAME.MIRAGE24: _mirage.Mirage24,
    DATASET_NAME.UTMOBILENET21: _utmobilenet.UTMobilenet21
}

class DatasetsCatalog(UserDict):
    def __init__(self):
        super().__init__()
        for dset_name, dset_class in _DATASET_NAME_TO_CLASS.items():
            self.data[str(dset_name)] = dset_class()

    def __getitem__(self, key: Any) -> DatasetMetadata:
        return self.data[str(key)]

    def __contains__(self, key: Any) -> bool:
        return str(key) in self.data

    def __setitem__(self, key: Any, value: Any) -> None:
        raise ValueError(f"{self.__class__.__name__} is immutable")

    def __rich__(self) -> richtree.Tree:
        tree = richtree.Tree("Datasets")
        for dset_name in sorted(self.keys()):
            dset_metadata = self[dset_name]    
            table = richtable.Table(
                show_header=False, 
                box=None,
                show_footer=False, 
                pad_edge=False,
                expand=True,
            )
            table.add_column("")
            table.add_row(f"[bold]{dset_name}[/bold]")
            table.add_row(
                richpanel.Panel(
                    dset_metadata.__rich__(),
                    box=richbox.ROUNDED,
                    expand=True,
                )
            )
            table.add_row("")
            tree.add(table)
        return tree

    def __rich_console__(self,
        console: rich.console.Console,
        options: rich.console.ConsoleOptions,
    ) -> rich.console.RenderResult:
        yield self.__rich__()


def get_datasets_catalog() -> DatasetsCatalog:
    return DatasetsCatalog()

def get_dataset(name: DATASET_NAME) -> Dataset:
    return DatasetsCatalog()[name]

def get_dataset_schema(
    dataset_name: DATASET_NAME, 
    dataset_type: DATASET_TYPE
) -> DatasetSchema:
    return get_dataset(dataset_name).get_schema(dataset_type)

def get_dataset_polars_schema(
    dataset_name: DATASET_NAME, 
    dataset_type: DATASET_TYPE
) -> pl.schema.Schema:
    return get_dataset_schema(dataset_name, dataset_type)
