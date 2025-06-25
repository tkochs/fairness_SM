from typing import Self, Optional
from sklearn.preprocessing import StandardScaler, TargetEncoder
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from aif360.datasets import BinaryLabelDataset
from datasets import DataSet
import pandas as pd
import unittest
import numpy as np


RANDOM_STATE: int = 42


class Processor:
    def fit(self, x: DataSet, y: pd.Series) -> Self:
        raise NotImplementedError

    def transform(self, x: DataSet) -> DataSet:
        raise NotImplementedError

    def fit_transform(self, x: DataSet, y: pd.Series) -> DataSet:
        return self.fit(x, y).transform(x)


class Preprocessor(Processor):
    """
        Do the preprocessing of data:
            1. Impute // is done in a 'lazy' way 1. try to fetch from disk
                                                 2. run imputation
            2. Target encode categoritcal values
            3. Standard scaler
    """

    def __init__(self, imputer: str, reweight_missing: bool = False):
        self.imputer = imputer
        self.reweight_missing = reweight_missing
        self._is_fitted = False
        # self.verbose = True

    def impute(self, x: DataSet, fit=False) -> DataSet:
        x = x.fetch_imputed(self.imputer)
        if fit:
            return x, x.binary_labels()
        else:
            return x

    def target_encode(self, x: DataSet,
                      y: Optional[pd.Series] = None,
                      fit: bool = False) -> DataSet:
        categoritcal_columns = x.df.select_dtypes(exclude="number").columns
        if len(categoritcal_columns) == 0:
            return x
        if fit:
            assert y is not None, "Trying to fit without Labels!"
            values, counts = np.unique(y, return_counts=True)
            self.encoder = TargetEncoder(cv=max(min_unique_classes, 2),
                                         random_state=RANDOM_STATE) if \
                (min_unique_classes := min(counts)) < 5 else\
                TargetEncoder(random_state=RANDOM_STATE)
            self.encoder.fit(x.df[categoritcal_columns], y)
        else:
            assert y is None, "Passing labels while not fitting"
        x.df[categoritcal_columns] = \
            self.encoder.transform(x.df[categoritcal_columns])
        return x

    def scale(self, x: DataSet, fit: bool = False) -> DataSet:
        cols = x.df.columns
        if fit:
            self.scaler = StandardScaler().fit(x.df)
        x.df = pd.DataFrame(self.scaler.transform(x.df), columns=cols)
        return x

    def fit(self, x: DataSet, y: pd.Series) -> Self:
        self.columns = x.df.columns
        if self.reweight_missing:
            missing_columns = x.df.isna().any(axis=1)
        x, y = self.impute(x, fit=True)
        self.check_data_after_impute(x)
        x = self.target_encode(x, y, fit=True)
        x = self.scale(x, fit=True)
        if self.reweight_missing:
            x = x.reweight(missing_columns)
        self._is_fitted = True
        return self

    def transform(self, x: DataSet) -> DataSet:
        assert self._is_fitted, "Call fit first"
        x = self.impute(x)
        # if fetched from disk we reintroduce the protected column
        x.df = x.df[self.columns]
        x = self.target_encode(x)
        x = self.scale(x)
        return x

    def check_data_after_impute(self, x: DataSet) -> None:
        assert x.df.shape[0] > 0, "No samples in training set"
        self.columns = x.df.columns


class Postprocessor(Processor):
    def __init__(self):
        self._is_fitted = False

    def fit(self, predictions: pd.Series, true_labels: pd.Series, attribute: pd.Series) -> Self:
        assert attribute is not None, "Please provide a valid attribute"
        minority = [{"protected": 0}]
        majority = [{"protected": 1}]
        self.eo_processor = EqOddsPostprocessing(majority, minority)
        true_data = self._series_to_labelset(attribute, true_labels)
        prediction_data = true_data.copy()
        prediction_data.labels = predictions.reshape(-1, 1)
        self.eo_processor.fit(true_data, prediction_data)
        self._is_fitted = True
        return self

    def transform(self, predictions, protected):
        assert self._is_fitted, "Please call fit before transform"
        pred = self._series_to_labelset(protected, predictions)
        return self.eo_processor.predict(pred).labels.ravel()

    def _series_to_labelset(self, protected_attribute, labels):
        assert isinstance(protected_attribute,
                          pd.Series), "Protected is not a series"
        data = pd.DataFrame({
            "label": labels,
            "protected": protected_attribute.values
        })
        binary_data = BinaryLabelDataset(df=data,
                                         label_names=["label"],
                                         protected_attribute_names=[
                                             "protected"]
                                         )
        return binary_data
###############################################################################
#                     Test Code after this line                               #
###############################################################################


class TestPreprocessor(unittest.TestCase):
    def test_encode(self):
        df = pd.DataFrame({
            "num1": np.random.randint(0, 10, 10),
            "num2": np.random.rand(10)
        })
        y = np.random.randint(0, 2, 10)
        ds = DataSet("mock")
        ds.df = df.copy()
        ds.labels = y
        preprocessor = Preprocessor("Simple")
        encoded = preprocessor.target_encode(ds, y, True)
        assert np.isclose(encoded.df, df).all()
        df = pd.DataFrame({
            "num1": np.random.randint(0, 10, 10),
            "num2": np.random.rand(10),
            "cat1": ["a", "a", "a", "a", "a", "a", "a", "a", "b", "b"],
            "cat2": ["c", "c", "c", "c", "a", "a", "a", "a", "b", "b"]
        })
        ds = DataSet("mock")
        ds.df = df.copy()
        ds.labels = y
        encoded = preprocessor.target_encode(ds, y, True)
        assert encoded.df["cat1"][0] != "a"
        test_val = df["num1"]
        encoded = preprocessor.target_encode(ds, y, True)
        assert (encoded.df["cat1"] != df["cat1"]).all(), \
            f"{encoded.df["cat1"]} != {df["cat1"]}"
        assert np.isclose(test_val, encoded.df["num1"]).all()


# run tests
if __name__ == "__main__":
    unittest.main()
