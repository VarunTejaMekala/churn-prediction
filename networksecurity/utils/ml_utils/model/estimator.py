import sys
import pandas as pd
import numpy as np

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


class ChurnModel:
    """
    Wraps preprocessor + trained model for end-to-end inference.

    FIX: renamed from NetworkModel → ChurnModel (correct domain name).
    FIX: added predict_proba() support (needed for threshold-based prediction).
    FIX: added input type guard — accepts both DataFrame and ndarray.
    """

    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def _transform(self, x):
        """Applies preprocessor. Accepts DataFrame or ndarray."""
        try:
            if isinstance(x, np.ndarray):
                return self.preprocessor.transform(
                    pd.DataFrame(x)
                )
            return self.preprocessor.transform(x)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def predict(self, x, threshold: float = 0.35) -> np.ndarray:
        """Returns binary predictions using probability threshold."""
        try:
            x_transformed = self._transform(x)
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(x_transformed)[:, 1]
                return (probs >= threshold).astype(int)
            return self.model.predict(x_transformed)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def predict_proba(self, x) -> np.ndarray:
        """Returns churn probability scores."""
        try:
            x_transformed = self._transform(x)
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(x_transformed)[:, 1]
            raise AttributeError("Underlying model does not support predict_proba")
        except Exception as e:
            raise NetworkSecurityException(e, sys)
