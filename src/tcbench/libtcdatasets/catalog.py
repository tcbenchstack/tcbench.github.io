from __future__ import annotations

import polars as pl

import rich.console
import rich.tree as richtree

from collections import UserDict

from tcbench.libtcdatasets import (
    dataset_mirage,
    dataset_ucdavis,
    dataset_utmobilenet,
)
from tcbench.libtcdatasets.core import (
    Dataset, 
    DatasetSchema
)

from tcbench.libtcdatasets.constants import (
    DATASET_NAME,
)

_DATASET_NAME_TO_CLASS = {
    DATASET_NAME.UCDAVIS19: dataset_ucdavis.UCDavis19,
    DATASET_NAME.MIRAGE19: dataset_mirage.Mirage19,
    DATASET_NAME.MIRAGE20: dataset_mirage.Mirage20,
    DATASET_NAME.MIRAGE22: dataset_mirage.Mirage22,
    DATASET_NAME.MIRAGE24: dataset_mirage.Mirage24,
    DATASET_NAME.UTMOBILENET21: dataset_utmobilenet.UTMobilenet21
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
            node = richtree.Tree(dset_name)
            node.add(dset_metadata.__rich__())
            tree.add(node)
        return tree

    def __rich_console__(self,
        console: rich.console.Console,
        options: rich.console.ConsoleOptions,
    ) -> rich.console.RenderResult:
        yield self.__rich__()


def datasets_catalog() -> DatasetsCatalog:
    return DatasetsCatalog()

def get_dataset(name: DATASET_NAME) -> Dataset:
    return DatasetsCatalog()[name]

def get_dataset_schema(
    dset_name: DATASET_NAME, 
    dset_type: DATASET_TYPE
) -> DatasetSchema:
    return get_dataset(dset_name).get_schema(dset_type)

def get_dataset_polars_schema(
    dset_name: DATASET_NAME, 
    dset_type: DATASET_TYPE
) -> pl.schema.Schema:
    return get_dataset_schema(dset_name, dset_type)
