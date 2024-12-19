from tcbench.core import StringEnum

from tcbench.modeling.columns import (
    COL_PKTS_SIZE,
    COL_PKTS_DIR,
    COL_PKTS_IAT,
    COL_PKTS_TCP_WINSIZE,
    COL_PKTS_SIZE_TIMES_DIR,
)

class MODELING_DATASET_TYPE(StringEnum):
    """An enumeration to specify which type of dataset to load"""
    TRAIN_VAL = "train_val_datasets"
    TEST = "test_dataset"
    TRAIN_VAL_LEFTOVER = "train_val_leftover_dataset"
    FINETUNING = "for_finetuning_dataset"


class MODELING_INPUT_REPR_TYPE(StringEnum):
    pass


class MODELING_FEATURE(StringEnum):
    PKTS_SIZE = COL_PKTS_SIZE
    PKTS_DIR = COL_PKTS_DIR
    PKTS_IAT = COL_PKTS_IAT
    PKTS_TCP_WINSIZE = COL_PKTS_TCP_WINSIZE
    PKTS_SIZE_TIMES_DIR = COL_PKTS_SIZE_TIMES_DIR


class MODELING_METHOD_NAME(StringEnum):
    XGBOOST = "ml.xgboost"
    RANDOM_FOREST = "ml.randomforest"
