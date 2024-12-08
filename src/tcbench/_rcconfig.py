from __future__ import annotations

from typing import Dict, Any
from collections import UserDict

import pathlib
import os

from tcbench.datasets._constants import DATASETS_DEFAULT_INSTALL_ROOT_FOLDER
from tcbench import fileutils

TCBENCHRC_PATH = pathlib.Path(os.path.expandvars("$HOME")) / ".tcbenchrc"


def init_tcbenchrc() -> Dict[str, Any]:
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
            init_tcbenchrc()
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
            raise RuntimeError(
                f"""missing "datasets" section in {TCBENCHRC_PATH}"""
            )
        if "install_folder" not in self.data["datasets"]:
            raise RuntimeError(
                f"""missing "datasets.install_folder" in {TCBENCHRC_PATH}"""
            )


