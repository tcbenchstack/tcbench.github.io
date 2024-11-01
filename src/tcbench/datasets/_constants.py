from __future__ import annotations

from tcbench.fileutils import _get_module_folder
from tcbench.core import StringEnum

import sys

_module_folder = _get_module_folder(__name__)

DATASETS_RESOURCES_FOLDER = _module_folder / "resources"
DATASETS_RESOURCES_METADATA_FNAME = DATASETS_RESOURCES_FOLDER / "DATASETS_METADATA.yml"
DATASETS_DEFAULT_INSTALL_ROOT_FOLDER = _module_folder / "installed_datasets"

APP_LABEL_BACKGROUND = "_background_"
APP_LABEL_ALL = "_all_"
