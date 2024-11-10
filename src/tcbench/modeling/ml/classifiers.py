from __future__ import annotations

from typing import List, Iterable, Dict, Any

import xgboost as xgb

from tcbench.modeling.ml.core import MLModel
from tcbench.modeling.enums import MODELING_FEATURE

class XGBoostClassifier(MLModel):
    def __init__(
        self,
        labels: Iterable[str],
        feature_names: Iterable[MODELING_FEATURE],
        seed: int = 1,
        **hyperparams: Dict[str, Any],
    ): 
        super().__init__(
            labels=labels,
            feature_names=feature_names,
            model_class=xgb.XGBClassifier,
            seed=seed,
            **hyperparams,
        )
        
