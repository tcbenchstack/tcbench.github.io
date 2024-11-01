from __future__ import annotations
from collections import UserDict

from typing import Dict, Any

import pathlib
import os.path
import yaml

from tcbench.libtcdatasets.constants import DATASETS_DEFAULT_INSTALL_ROOT_FOLDER
from tcbench import fileutils

__version__ = "0.0.22"

TCBENCHRC_PATH = pathlib.Path(os.path.expandvars("$HOME")) / ".tcbenchrc"


def _init_tcbenchrc() -> Dict[str, Any]:
    data=dict(
        datasets=dict(
            install_folder=str(DATASETS_DEFAULT_INSTALL_ROOT_FOLDER)
        )
    )
    fileutils.save_yaml(data, TCBENCHRC_PATH, echo=False)

def is_valid_config(param_name:str, param_value: str) -> bool:
    if param_name not in {
        "datasets.install_folder"
    }:
        return False
    return True

class TCBenchRC(UserDict):
    def __init__(self):
        super().__init__()
        if not TCBENCHRC_PATH.exists():
            _init_tcbenchrc()
        self.load()

    @property
    def install_folder(self):
        return pathlib.Path(self.data["datasets"]["install_folder"])

    def save(self):
        fileutils.save_yaml(self.data, TCBENCHRC_PATH)

    def __getitem__(self, key: str) -> str:
        curr_level = self.data     
        key_levels = key.split(".")[::-1]
        while key_levels:
            try:
                curr_level = curr_level[key_levels.pop()]
            except KeyError:
                raise KeyError(key)
        return curr_level

    def __setitem__(self, key: str, value = str) -> None:
        curr_level = self.data 
        key_levels = key.split(".")[::-1]
        while len(key_levels) > 1:
            curr_level = curr_level[key_levels.pop()]
        curr_level[key_levels[0]] = value

    def load(self):
        self.data = fileutils.load_yaml(TCBENCHRC_PATH, echo=False)

        if "datasets" not in self.data:
            raise RuntimeException(f"""missing "datasets" section in {TCBENCHRC_PATH}""")
        if "install_folder" not in self.data["datasets"]:
            raise RuntimeException(f"""missing "datasets.install_folder" in {TCBENCHRC_PATH}""")


def get_config():
    return TCBenchRC()

from tcbench.libtcdatasets.catalog import (
    datasets_catalog,
    get_dataset,
    get_dataset_schema,
    get_dataset_polars_schema,
)
from tcbench.libtcdatasets.constants import (
    DATASET_NAME,
    DATASET_TYPE,
)


##########################
# OLDER CONTENT
##########################


DEFAULT_AIM_REPO = pathlib.Path("./aim-repo")
DEFAULT_ARTIFACTS_FOLDER = pathlib.Path("./aim-repo/artifacts")

DEFAULT_CAMPAIGN_AUGATLOAD_AUGMENTATIONS = (
    "noaug",
    "rotate",
    "horizontalflip",
    "colorjitter",
    "packetloss",
    "changertt",
    "timeshift",
)
DEFAULT_CAMPAIGN_AUGATLOAD_SEEDS = (12345, 42, 666)
DEFAULT_CAMPAIGN_AUGATLOAD_FLOWPICDIMS = (32, 64, 1500)
DEFAULT_CAMPAIGN_AUGATLOAD_PKTSERIESLEN = (10, 30)

DEFAULT_CAMPAIGN_CONTRALEARNANDFINETUNE_FLOWPICDIMS = (32, 64, 1500)
DEFAULT_CAMPAING_CONTRALEARNANDFINETUNE_SEEDS_CONTRALEARN = (12345, 1, 2, 3, 4)
DEFAULT_CAMPAIGN_CONTRALEARNANDFINETUNE_SEEDS_FINETUNE = (12345, 1, 2, 3, 4)
DEFAULT_CAMPAIGN_CONTRALEARNANDFINETUNE_AUGMENTATIONS = "changertt,timeshift"
DEFAULT_CAMPAIGN_CONTRALEARNANDFINETUNE_VALID_AUGMENTATIONS = tuple([
    aug_name
    for aug_name in DEFAULT_CAMPAIGN_AUGATLOAD_AUGMENTATIONS
    if aug_name != "noaug"
])

#from tcbench.libtcdatasets.datasets_utils import (
#    get_datasets_root_folder,
#    get_dataset_folder,
#    DATASETS,
#    load_parquet,
#)

#from tcbench.modeling import (
#    MODELING_DATASET_TYPE,
#    MODELING_INPUT_REPR_TYPE,
#    #MODELING_METHOD_TYPE,
#)
