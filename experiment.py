from typing import Optional
from itertools import product
from pathlib import Path
from sklearn.base import clone
from sklearn.metrics import f1_score, balanced_accuracy_score
from fairlearn.metrics import equalized_odds_difference, equalized_odds_ratio, \
    demographic_parity_difference, true_positive_rate, true_negative_rate, \
    false_positive_rate, equal_opportunity_difference
from classifiers import Classifier
from processing import Preprocessor, Postprocessor
from datasets import DataSet
import os
import unittest
import numpy as np
import pandas as pd


RANDOM_STATE: int = 42
STORE_RAW_RESULTS: bool = False  # whether or not to store predictions
STORE_PROCESSED_RESULTS: bool = True# whether or not to store metrics evalutated on above predictions


class TrainingParamter:
    """
        Iterator => Return (int indicatng missing %,
                            Union(None, percentage artificial missing data))
    """

    def __init__(self, missing_index: list,
                 percentage_augmented_data: Optional[list] = None,
                 miss_forest: bool = False):
        assert isinstance(miss_forest, bool), "Please pass a boolean"
        if miss_forest and percentage_augmented_data is None:
            missing_index = [e for e in missing_index if e != 0]
        self.items = [(a, b)
                      for a in missing_index
                      for b in percentage_augmented_data]\
            if percentage_augmented_data is not None\
            else\
            [(a, None) for a in missing_index]
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.items):
            raise StopIteration
        item = self.items[self.index]
        self.index += 1
        return item

    def __len__(self):
        return len(self.items)


class Experiment:
    RESULTS_PATH = Path("data_store/results/")
    left = 0.10
    right = 0.15
    artifical_step_size = 0.05

    def __init__(self, id: str, dataset_name: str, imputer: str,
                 classifier: Classifier, missing_types: list[str],
                 *args,
                 test_sets: list[int] = range(5),
                 postprocessed: bool = False,
                 reweighting: bool = False):
        self.id = id
        self.dataset_name = dataset_name
        self.test_sets = test_sets
        self.imputer = imputer
        self.classifier = classifier
        self.missing_types = missing_types
        self.postprocessed = postprocessed
        self.reweighting = reweighting
        self._sanity_check()

        # init data storage
        self.RESULTS_PATH_RAW = self.RESULTS_PATH / \
            Path(f"results_raw_{self.dataset_name}_{self.imputer}_{self.id}.csv")
        self.RESULTS_PATH_PROCESSED = self.RESULTS_PATH / \
            Path(f"results_processed_{self.dataset_name}_{self.imputer}_{self.id}.csv")
        self.init_data()

        match id:
            case "missing_imputed" | "missing_imputed_postprocess" | "missing_imputed_reweighting":
                self.extension = "clean"
                self.missing_in_train = 1
                self.missing_in_test = 1
                self.train_param = TrainingParamter(range(5),
                                                    miss_forest=imputer == "MissForest")
            case "missing_imputed_test_only" |\
                    "missing_imputed_test_only_postprocess":
                self.extension = "clean"
                self.missing_in_train = 0
                self.missing_in_test = 1
                self.train_param = TrainingParamter([0],
                                                    miss_forest=imputer == "MissForest")
            case "imputed" | "imputed_postprocess" | "imputed_reweighting":
                self.extension = "clean"
                self.missing_in_train = 1
                self.missing_in_test = 0
                self.train_param = TrainingParamter(range(5),
                                                    miss_forest=imputer == "MissForest")
                self.test_sets = [0]  # only test on complete data
            case "artificial":
                self.extension = "artificial"
                self.missing_in_train = 0
                self.missing_in_test = 1
                self.train_param = TrainingParamter([0], np.arange(
                    self.left, self.right, self.artifical_step_size),
                    miss_forest=imputer == "MissForest")
            case "artificial_missing":
                self.extension = "artificial_missing"
                self.missing_in_train = 1
                self.missing_in_test = 1
                self.train_param = TrainingParamter(range(5), np.arange(
                    self.left, self.right, self.artifical_step_size),
                    miss_forest=imputer == "MissForest")
                # self.test_sets = range(5)
            case "artificial_important":
                self.extension = "artificial_important"
                self.missing_in_train = 0
                self.missing_in_test = 1
                self.train_param = TrainingParamter([0], np.arange(
                    self.left, self.right, self.artifical_step_size),
                    miss_forest=imputer == "MissForest")
            case "mock" | "mock2":
                # for testing purposes only
                return
            case _:
                raise NotImplementedError(f"{id}")

        required_attributes = {"missing_in_train", "missing_in_test",
                               "extension", "results_raw", "results_processed"}
        for attr in required_attributes:
            assert hasattr(self, attr)

    def run(self) -> None:
        print(f"Imputer: {self.imputer}")
        # first value of 'param' is the int indicating missing percentage
        print(self.missing_types)
        for missing_type, param, split in product(self.missing_types, self.train_param, range(5)):
            # [(mt, p, s)
            #  for mt in self.missing_types
            #  for p in self.train_param
            #  for s in range(5)]:
            train_set = DataSet(self.dataset_name, True,
                                param[0], split, missing_type, self.extension,
                                training_param=param,
                                missing_in_train=self.missing_in_train)
            print(f"Train: {param} fold: {split} MT: {missing_type}"\
                f" Postprocessed: {self.postprocessed}, Reweight: {self.reweighting}")
            preprocessor = Preprocessor(self.imputer, self.reweighting)\
                .fit(train_set, train_set.binary_labels())
            classifier = clone(self.classifier).fit(train_set.df,
                                                    train_set.binary_labels())
            if self.postprocessed:
                assert "MNAR" not in missing_type, \
                    "cannot be performed with MNAR"
                postprocessor = Postprocessor()\
                    .fit(classifier.predict(train_set.df),
                         train_set.binary_labels(),
                         train_set.protected_tested(True))
            for test_missing_percentage in self.test_sets:
                test_set = DataSet(self.dataset_name, False,
                                   test_missing_percentage,
                                   split, missing_type,
                                   self.extension,
                                   training_param=param,
                                   missing_in_train=self.missing_in_train)
                # rows with missing values
                missing_mask = test_set.df.isna().any(axis=1)
                print(f"Test: {test_missing_percentage} with train"
                      f"{param} fold: {split}")
                test_set = preprocessor.transform(test_set)
                predictions = classifier.predict(test_set.df)
                if self.postprocessed:
                    predictions = postprocessor.transform(predictions,
                                                          test_set.protected_tested(True))
                self.store_results(predictions,
                                   test_set.binary_labels(),
                                   test_set.protected_tested(),
                                   missing_mask,
                                   test_missing_percentage,
                                   param,
                                   split,
                                   missing_type)
        self.to_disk()

    def to_disk(self):
        print("writing to disk..")
        if not self.RESULTS_PATH.exists():
            os.makedirs(self.RESULTS_PATH)
        self.results_raw.to_csv(self.RESULTS_PATH_RAW, index=False)
        self.results_processed.to_csv(self.RESULTS_PATH_PROCESSED, index=False)

    def init_data(self):
        self.results_raw = pd.read_csv(self.RESULTS_PATH_RAW, dtype={'alpha': 'string'}) if\
            os.path.exists(self.RESULTS_PATH_RAW) else pd.DataFrame()
        self.results_processed = pd.read_csv(self.RESULTS_PATH_PROCESSED, dtype={'alpha': 'string'}) if\
            os.path.exists(self.RESULTS_PATH_PROCESSED) else pd.DataFrame()

    def store_results(self,
                      predictions: pd.Series,
                      labels: pd.Series,
                      protected: pd.Series,
                      missing_mask: pd.Series,
                      test_id: int,
                      train_id: object,
                      split: int,
                      missing_type: str) -> None:

        def add_results(df: pd.DataFrame, kind: str) -> None:
            match str(kind).lower():
                case "raw":
                    ids = self.results_raw[list(identifiers)].eq(
                        pd.Series(identifiers)).all(axis=1) if not self.results_raw.empty\
                        else np.array([False])
                    if ids.any():  # this experiment has been done before
                        # remove old results (overwrite)
                        self.results_raw = self.results_raw[~ids]
                    self.results_raw = pd.concat([df, self.results_raw])
                case "processed":
                    ids = self.results_processed[list(identifiers)].eq(
                        pd.Series(identifiers)).all(axis=1) if not self.results_processed.empty\
                        else np.array([False])
                    if ids.any():  # this experiment has been done before
                        # remove old results (overwrite)
                        self.results_processed = self.results_processed[~ids]
                    self.results_processed = pd.concat(
                        [df, self.results_processed])
                case _:
                    raise ValueError(f"{str(kind)}")
            
        n = len(predictions)
        for attribute in [labels, protected, missing_mask]:
            assert len(attribute) == n, "Make sure all value parameters are"\
                f"the same length: n={n}, len({attribute.name})={
                len(attribute)}"
        train_missing, alpha = train_id
        match (self.postprocessed, alpha, self.reweighting):
            case (False, None, False):
                processing = "No"
            case (True, None, False):
                processing = "Post"
            case (False, _, False):
                processing = "Pre"
            case (False, None, True):
                processing = "Reweighting"
            case (False, _, True):
                # setting alpha and reweighting is exclusive
                raise ValueError("non-valid processing step")
            case _:
                raise ValueError("non-valid processing step")
        identifiers = {
            "id": self.id,
            "data_set": self.dataset_name,
            "imputer": self.imputer,
            "classifier": self.classifier.name,
            "missing_type": missing_type,
            "missing_in_train": self.missing_in_train if test_id != 0 else 0,
            "missing_in_test": self.missing_in_test if train_missing != 0 else 0,
            "split": split,
            "test_missing_percentage": test_id,
            "train_missing_percentage": train_missing,
            "processing": processing,
            # round to 2nd decimal place
            "alpha": f"{alpha:.2f}" if alpha is not None else "No"
        }
        metrics = {
            "f1": f1_score(labels, predictions),
            "balanced_accuracy": balanced_accuracy_score(labels, predictions),
            "tpr": true_positive_rate(labels, predictions),
            "tnr": true_negative_rate(labels, predictions),
            "fpr": false_positive_rate(labels, predictions),
            "equal_opportunity_difference":
                equal_opportunity_difference(
                    labels, predictions, sensitive_features=protected),
            "equalized_odds_difference":
                equalized_odds_difference(
                    labels, predictions, sensitive_features=protected),
            "statistical_parity_difference":
                demographic_parity_difference(
                    labels, predictions, sensitive_features=protected),
        }
        if STORE_RAW_RESULTS:
            results_raw = {key: [val]*n for key, val in identifiers.items()} |\
                {"had_missing_feature": missing_mask,
                "prediction": predictions,
                "gt": labels,
                "protected": protected}
            results_raw = pd.DataFrame(results_raw)
            add_results(results_raw, "raw")
        if STORE_PROCESSED_RESULTS:
            results_processed = {key: [val] for key, val in identifiers.items()} |\
                {metric_name: scoring for metric_name, scoring in metrics.items()}
            results_processed = pd.DataFrame(results_processed)
            results_processed[list(metrics.keys())] = results_processed[list(metrics.keys())].map(lambda x: round(x, 4))
            add_results(results_processed, "processed")

    ##### private functions for internal checks #####
    def _sanity_check(self) -> None:
        # add all requirements for stuff to go wrong here
        # fail early to not waste time
        self._check_postprocess()
        self._check_reweighting()
        if self.imputer.lower() == "removed":
            assert self.id == "imputed", "No other experiment is valid"
            assert not self.reweighting, "We dont allow this combination"
        if self.imputer.lower() == "missforest":
            assert "test_only" not in self.id

    def _check_postprocess(self) -> None:
        if "postprocess" in self.id:
            assert self.postprocessed, \
                "Tagged as postprocessed but param not set"
        if self.postprocessed:
            assert not self.reweighting
            assert "postprocess" in self.id, \
                f"add postprocess tag! to {self.id}"

    def _check_reweighting(self) -> None:
        if "reweighting" in self.id:
            assert self.reweighting, "Tagged as reweighting but param not set"
        if self.reweighting:
            assert self.imputer.lower() != "removed", "Dont pass removed here"
            assert not self.postprocessed
            assert "reweighting" in self.id, \
                f"add reweighting tag! to {self.id}"


###############################################################################
#                     Test Code after this line                               #
###############################################################################


class TestExperiment(unittest.TestCase):
    def test_store_results(self):
        class MockClassifier:
            name = "Mock"
        e = Experiment("mock", "mock", "Simple", MockClassifier(), ["MAR"])
        e.RESULTS_PATH_RAW = Path("mock_raw.csv")
        e.RESULTS_PATH_PROCESSED = Path("mock_processed.csv")
        e.missing_in_test = 1
        e.missing_in_train = 0
        e.extension = "mock"
        e.init_data()
        # assert Experiment.RESULTS_PATH_RAW == Path(
        #     "/hri/localdisk/tkochs/results_raw.csv")
        # assert Experiment.RESULTS_PATH_PROCESSED == Path(
        #     "/hri/localdisk/tkochs/results_processed.csv")
        # remove mock objects if still present
        if e.RESULTS_PATH_RAW.exists():
            os.remove(e.RESULTS_PATH_RAW)
        if e.RESULTS_PATH_PROCESSED.exists():
            os.remove(e.RESULTS_PATH_PROCESSED)
        e.store_results([0, 1], [1, 1], [1, 0], [
                        0, 0],  0, (1, None), 4, "MAR")
        assert not e.results_raw.empty
        assert not e.results_processed.empty
        # e.to_disk()
        # assert e.RESULTS_PATH_RAW.exists()
        # assert e.RESULTS_PATH_PROCESSED.exists()
        df = e.results_raw  # pd.read_csv(e.RESULTS_PATH_RAW)
        print(df)
        assert df["prediction"][0] == 0, f"somwthing went terribly wrong\n{
            df.head()}"
        assert df.shape == (2, 16), f"{df.shape}"
        e.store_results([1, 1], [1, 1], [1, 0], [
                        0, 0],  0, (1, None), 4, "MAR")
        # e.to_disk()
        df = e.results_raw  # pd.read_csv(e.RESULTS_PATH_RAW)
        assert df.shape == (2, 16), f"{df.shape}"
        assert df["prediction"][0] == 1, f"somwthing went terribly wrong\n{
            df.head()}"
        e.to_disk()
        assert e.RESULTS_PATH_RAW.exists()
        assert e.RESULTS_PATH_PROCESSED.exists()
        e = Experiment("mock2", "mock", "Simple", MockClassifier(), ["MAR"])
        e.RESULTS_PATH_RAW = Path("mock_raw.csv")
        e.RESULTS_PATH_PROCESSED = Path("mock_processed.csv")
        e.missing_in_test = 1
        e.missing_in_train = 0
        e.extension = "mock"
        e.init_data()
        e.store_results([0, 1], [1, 1], [1, 0], [
                        0, 0],  0, (1, None), 4, "MAR")
        print(e.results_raw)
        assert ("mock" in set(e.results_raw["id"])) and (
            "mock2" in set(e.results_raw["id"]))
        e.store_results([0, 1], [1, 1], [1, 0], [
                        0, 0],  0, (1, None), 4, "MAR")
        assert e.results_raw.shape == (4, 16), f"{e.results_raw.shape}"
        e.store_results([0, 1], [1, 1], [1, 0], [
                        0, 0],  0, (1, None), 4, "MNAR")
        assert e.results_raw.shape == (6, 16), f"{e.results_raw.shape}"


class TestTrainingParam(unittest.TestCase):
    def test_miss_forrest(self):
        param = TrainingParamter([0])
        assert len(param) == 1, f"got {len(param)} instead"
        param = TrainingParamter([0, 1], miss_forest=True)
        assert len(param) == 1, f"got {len(param)} instead"


# run tests
if __name__ == "__main__":
    unittest.main()
