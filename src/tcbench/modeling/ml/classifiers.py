from __future__ import annotations

from typing import Iterable, Dict, Any

import xgboost as xgb

from tcbench.modeling.ml.core import MLModel
from tcbench.modeling.enums import MODELING_FEATURE

class XGBoostClassifier(MLModel):
    def __init__(
        self,
        labels: Iterable[str],
        features: Iterable[MODELING_FEATURE],
        seed: int = 1,
        num_workers: int | None = None,
        *,
        hyperparams: Dict[str, Any],
    ): 
        if num_workers is not None and "n_jobs" not in hyperparams:
            hyperparams["n_jobs"] = num_workers

        super().__init__(
            labels=labels,
            features=features,
            model_class=xgb.XGBClassifier,
            seed=seed,
            hyperparams=hyperparams,
        )
