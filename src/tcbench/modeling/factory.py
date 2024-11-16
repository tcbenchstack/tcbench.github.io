from __future__ import annotations

from typing import Iterable, Dict, Any

from tcbench.modeling.enums import (
    MODELING_METHOD_NAME,
    MODELING_FEATURE,
)
from tcbench.modeling.ml import (
    classifiers as mlclassifiers,
    core as mlcore
)

MODEL_NAME_TO_CLASS = {
    MODELING_METHOD_NAME.XGBOOST: mlclassifiers.XGBoostClassifier
}


def mlmodel_factory(
    name: MODELING_METHOD_NAME,
    labels: Iterable[str],
    features: Iterable[MODELING_FEATURE],
    seed: int = 1,
    num_workers: int = 1,
    **hyperparams: Dict[str, Any]
) -> mlcore.MLModel:
    cls = MODEL_NAME_TO_CLASS.get(name, None)
    if cls:
        return cls(
            labels=labels,
            features=features,
            seed=seed,
            num_workers=num_workers,
            **hyperparams
        )
    raise RuntimeError(f"ModelNotFound: unrecognized model name {name}")
