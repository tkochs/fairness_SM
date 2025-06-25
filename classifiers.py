from enum import Enum
# from sklvq import GMLVQ, LGMLVQ, GLVQ
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from typing import Self, Optional

import pandas as pd
import numpy as np

###############################################################################
#    Small Wrapper around sklearn classifiers and xgboost
###############################################################################


class Classifier:
    name: str = "Classifier (Generic)"

    def fit(self, x: pd.DataFrame, y: pd.Series) -> Self:
        self._check_input(x, y)
        self.model.fit(x, y)
        self._is_fitted = True
        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        assert self._is_fitted
        return self.model.predict(x)

    def _check_input(self, x: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        assert isinstance(x, pd.DataFrame), f"Expected pandas dataframe but got {
            type(x)}"
        assert isinstance(y, (pd.DataFrame, np.ndarray, type(None))
                          ), f"Expected pandas Series but got {type(y)}"
        assert self.name != "Classifier (Generic)", "Please call a specifc classifier not Interface"

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "_is_fitted") and self._is_fitted

    def get_params(self, *args, **kwargs) -> dict:
        return {**self.model.get_params()}


class Classifiers(Enum):
    KNN: str = "KNN"
    LVQ: str = "LVQ"
    SVM: str = "SVM"
    RF: str = "RF"
    DT: str = "DT"
    XGBOOST: str = "XGBoost"
    LOGISTIC: str = "Logistic Regression"
    MLP: str = "MLP"
    BAYES: str = "Bayes"

###############################################################################
#    Logistic Regression
###############################################################################


class Logistic(Classifier):
    _type = Classifiers.LOGISTIC

    def __init__(self,  max_iter=1000, *args, **kwargs):
        self.name = "Logistic Regression"
        self.model = LogisticRegression(max_iter=max_iter, *args, **kwargs)

###############################################################################
#    Naive Bayes
###############################################################################


class Bayes(Classifier):
    _type = Classifiers.BAYES

    def __init__(self, *args, **kwargs):
        self.name = "Bayes"
        self.model = GaussianNB(*args, **kwargs)


###############################################################################
#    Mulitlayer Perceptron
###############################################################################


class MLP(Classifier):
    _type = Classifiers.MLP

    def __init__(self, *args, **kwargs):
        self.name = "MLP"
        self.model = MLPClassifier(*args, **kwargs)

###############################################################################
#    KNN
###############################################################################


class Weights(Enum):
    uniform: str = "uniform"
    distance: str = "distance"


class KNNClassifier(Classifier):
    _type = Classifiers.KNN

    def __init__(self, k: int = 5, weights: Weights = Weights.uniform, **kwargs):
        self.k = k
        self.model = KNeighborsClassifier(
            n_neighbors=self.k, weights=weights.value)

    def get_params(self, *args, **kwargs) -> dict:
        return self.model.get_params() | {'weights': self.weights}

###############################################################################
#    XGBoost : Gradient boosted trees
###############################################################################


class XGBoost(Classifier):
    name = "XGBoost"
    _type = Classifiers.XGBOOST

    def __init__(self, *args, **kwargs):
        self._estimator_type = "classifier"
        self.model = XGBClassifier(*args, **kwargs)

###############################################################################
#    RF
###############################################################################


class RF(Classifier):
    name = "RF"
    _type = Classifiers.RF

    def __init__(self, *args, **kwargs):
        self._estimator_type = "classifier"
        self.model = RandomForestClassifier(*args, **kwargs)

###############################################################################
#    DT
###############################################################################


class DT(Classifier):
    name = "DT"
    _type = Classifiers.DT

    def __init__(self, *args, **kwargs):
        self._estimator_type = "classifier"
        self.model = DecisionTreeClassifier()

###############################################################################
#    SVM
################################################################################


class SVM(Classifier):
    name = "SVM"
    _type = Classifiers.SVM

    def __init__(self, *args, kernel=RBF, **kwargs):
        self._estimator_type = "classifier"
        self.model = SVC(kernel=kernel)
