from __future__ import annotations

from typing import Iterable, Dict, Any, Callable

from tcbench.modeling.ml.core import MLModel
from tcbench.modeling.enums import (
    register_model_class,
    MODELING_FEATURE,
    MODELING_METHOD_NAME,
	MODEL_NAME_TO_CLASS,
)


@register_model_class(MODELING_METHOD_NAME.XGBOOST)
class XGBoostClassifier(MLModel):
    def __init__(
        self,
        labels: Iterable[str],
        features: Iterable[MODELING_FEATURE],
        *,
        seed: int = 1,
        num_workers: int | None = None,
        hyperparams: Dict[str, Any] | None = None,
    ): 
        if hyperparams is None:
            hyperparams = dict()
        if "random_state" not in hyperparams:
            hyperparams["random_state"] = seed
        if num_workers is not None and "n_jobs" not in hyperparams:
            hyperparams["n_jobs"] = num_workers

        import xgboost
        super().__init__(
            xgboost.XGBClassifier,
            labels,
            features,
            seed=seed,
            num_workers=num_workers,
            hyperparams=hyperparams
        )


@register_model_class(MODELING_METHOD_NAME.RANDOM_FOREST)
class RandomForest(MLModel):
    def __init__(
        self,
        labels: Iterable[str],
        features: Iterable[MODELING_FEATURE],
        *,
        seed: int = 1,
        num_workers: int | None = None,
        hyperparams: Dict[str, Any] | None = None,
    ):
        if hyperparams is None:
            hyperparams = dict()
        if "random_state" not in hyperparams:
            hyperparams["random_state"] = seed
        if num_workers is not None and "n_jobs" not in hyperparams:
            hyperparams["n_jobs"] = num_workers
        
        import sklearn.ensemble
        super().__init__(
            sklearn.ensemble.RandomForestClassifier,
            labels,
            features,
            seed=seed,
            num_workers=num_workers,
            hyperparams=hyperparams,
        )


def factory(
    name: MODELING_METHOD_NAME,
    labels: Iterable[str],
    features: Iterable[MODELING_FEATURE],
    seed: int = 1,
    num_workers: int = 1,
    *,
    hyperparams: Dict[str, Any] | None = None
) -> MLModel:
    cls = MODEL_NAME_TO_CLASS.get(name, None)
    if cls:
        return cls(
            labels=labels,
            features=features,
            seed=seed,
            num_workers=num_workers,
            hyperparams=hyperparams,
        )
    raise RuntimeError(f"ModelNotFound: unrecognized model name {name}")
