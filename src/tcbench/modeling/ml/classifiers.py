from __future__ import annotations

from typing import Iterable, Dict, Any, Callable

from tcbench.modeling.ml.core import MLModel
from tcbench.modeling.enums import MODELING_FEATURE

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
