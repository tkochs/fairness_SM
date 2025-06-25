from typing import Self, Optional
from pathlib import Path
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from imputation import get_imputers
import numpy as np
import pandas as pd
import unittest
import datamanager as dm

RANDOM_STATE = 42
CV_SPLITS = 5


class DataSet:
    def __init__(self, name: str, train: bool = True, missing: int = 0,
                 split: int = 0, missing_type: str = "MCAR",
                 extension: str = "clean", *,
                 training_param: object = (None, None),
                 missing_in_train: bool = True,
                 significant: Optional[int] = None):
        self.name = name
        self.train = train
        self.missing = missing
        self.split = split
        self.missing_type = missing_type
        self.extension = extension
        self.trained_on, self.artificial_percentage = training_param
        self.significant = significant
        if not missing_in_train:
            self.trained_on = 0
        match extension:
            case "clean":
                if self.artificial_percentage is not None:
                    raise ValueError(
                        "When performing these experiments pass None here")
            case "artificial":
                if (not isinstance(self.artificial_percentage, float))\
                        or (self.artificial_percentage < 0.05)\
                        or (self.artificial_percentage > 0.95):
                    raise ValueError("add percentage between 0.05 and 0.95")

        # _fetching does NOT mutate self #
        if train:
            self.trained_on = None
            if name == "mock":
                return
            self.df, self.labels, self.protected = self._fetch_train()
        else:
            assert self.trained_on is not None, "Please refer to a train set"
            if name == "mock":
                return
            self.df, self.labels, self.protected = self._fetch_test()

    def binary_labels(self) -> pd.Series:
        return LabelEncoder().fit_transform(dm.make_label_binary(self.name,
                                                                 self.labels))

    def _fetch_train(self, missing: Optional[int] = None)\
            -> (pd.DataFrame, pd.Series, list):
        missing = self.missing if missing is None else missing
        match self.extension:
            case "clean":
                try:
                    return dm.load_missing(self.name, self.missing_type, True,
                                           [missing], [self.split])[self.name][0][0]
                except Exception:
                    return self._make_missing(missing)
            case "artificial" | "artificial_random":
                # case without missing values in train
                try:
                    return dm.fetch_artificial(self.name, self.missing_type,
                                               self.artificial_percentage,
                                               self.extension)[self.name][self.split]
                except Exception:
                    return self._make_artificial()
            case "artificial_important":
                # case without missing values in train
                try:
                    return dm.fetch_artificial(self.name, "IMPORTANT",
                                               self.artificial_percentage,
                                               self.extension)[self.name][self.split]
                except Exception:
                    return self._make_artificial()
            case "artificial_missing":
                # case WITH missing values in train
                return dm.fetch_artificial(self.name, self.missing_type,
                                           self.artificial_percentage,
                                           self.extension)[self.name][self.split][missing]
            # case "artificial_significant":
            #     assert self.significant is not None, \
            #         "Add number of sig. features"
            #     return dm.fetch_artificial(self.name, self.significant,
            #                                self.artificial_percentage,
            #                                self.extension)[self.name][self.split][missing]
            case _:
                raise NotImplementedError("implement construction for "
                                          f"{self.extension}")
        raise ValueError("Shouldn't reach here.")

    def _fetch_test(self, missing: Optional[int] = None) \
            -> (pd.DataFrame, pd.Series, list):
        missing = self.missing if missing is None else missing
        try:
            return dm.fetch_test(self.name, self.split, True,
                                 self.missing_type, missing)
        except Exception:
            return self._make_missing(missing, test=True)

    def _make_missing(self, missing_percentage: int, test: bool = False) -> (pd.DataFrame, pd.Series, list[pd.Series]):
        assert isinstance(missing_percentage, int)
        lookup = {0: 0.0,
                  1: 0.1,
                  2: 0.3,
                  3: 0.5,
                  4: 0.75}
        assert missing_percentage in lookup.keys()
        df, y, p = self._split(test)  # get CV split
        print(f"{self.split}, {self.missing}, {self.missing_type}, {self.train}")
        assert not df.isna().any().any(), "There shouldnt be missing vals in here"
        df, p = MissingGenerator(self.name, self.missing_type, self.train, df, p)\
            .induce(lookup[missing_percentage])
        path = f"data_store/{self.extension}/{self.missing_type}/{missing_percentage}/{self.name}/split{self.split}.csv" if self.train and test == False\
            else f"data_store/clean/{self.missing_type}/test/{missing_percentage}/{self.name}/split{self.split}.csv"
        dm.store_to_disk(
            path,
            df,
            y,
            p
        )
        return df, y, p

    def _split(self, test: bool = False) -> (pd.DataFrame, pd.Series, list):
        def get_from_disk():
            if self.train and not test:
                return dm.load_missing(self.name, self.missing_type, True, [0],
                                       [self.split])[self.name][0][0]
            else:
                return dm.fetch_data(f"clean/{self.missing_type}/test/0/{self.name}/split{self.split}.csv")
        try:
            # try to fetch dataset without missing values
            return get_from_disk()
        except Exception:
            # failed to fetch data from disk need to set up cross validation
            self._make_and_save_splits()
        # should not fail since we just stored these csv files
        return get_from_disk()

    def _make_and_save_splits(self):
        # get complete dataset from disk
        df, labels, protected = dm.fetch_data(f"datasets/{self.name}")
        # clean naturally missing values (assume columns with large
        # missing percentages have been removed)
        missing_rows = df.isna().any(axis=1)
        df = df[~missing_rows]
        labels = labels[~missing_rows]
        protected = [p[~missing_rows] for p in protected]
        # splitting
        cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True,
                             random_state=RANDOM_STATE)
        binary_labels = dm.make_label_binary(self.name, labels)
        for fold_idx, (train_indices, test_indices) in \
                enumerate(cv.split(df, binary_labels)):
            x_train, x_test = df.iloc[train_indices], df.iloc[test_indices]
            y_train, y_test = labels.iloc[train_indices], labels.iloc[test_indices]
            p_train, p_test = [p.iloc[train_indices] for p in protected], \
                [p.iloc[test_indices] for p in protected]
            for path, (x, y, p) in zip([f"data_store/clean/{self.missing_type}/0/{self.name}/split{fold_idx}.csv",
                                        f"data_store/clean/{self.missing_type}/test/0/{self.name}/split{fold_idx}.csv"],
                                       [(x_train, y_train, p_train),
                                        (x_test, y_test, p_test)]):
                dm.store_to_disk(
                    path,
                    x,
                    y,
                    p
                )

    def fetch_imputed(self, imputer: str) -> Self:
        if self.name == "census":
            self.df = self.impute(imputer)
            return self
        if imputer.lower() == "removed":
            return self._remove_missing()
        try:
            self.df, _, _ = dm.fetch_data(self.path(imputer))
        except Exception as e:
            print(f"Failed to fetch: {self.path(imputer)} : {e}")
            self.df = self.impute(imputer)
        return self

    def impute(self, imputer: str) -> pd.DataFrame:
        imputer, method_name = clone(get_imputers()[imputer]), imputer
        # get training data
        data, labels, protected = self._fetch_train(self.trained_on)
        imputer.fit(data)
        imputed_train = imputer.transform(data)
        print(f"{"*"*20} STORING: TRAIN {"*"*20}")
        dm.store_to_disk(self.path(method_name, True),
                         imputed_train, labels, protected, True)
        if not self.train:
            imputed_test = imputer.transform(self.df)
            print(f"{"*"*20} STORING: TEST {"*"*20}")
            dm.store_to_disk(self.path(method_name),
                             imputed_test, self.labels, self.protected)
            return imputed_test
        return imputed_train

    def _remove_missing(self) -> Self:
        missing_rows = self.df.isna().any(axis=1)
        self.df, self.labels, self.protected = self.df.dropna(), \
            self.labels[~missing_rows], \
            [protected[~missing_rows] for protected in self.protected]
        assert self.df.shape[0] == len(
            self.labels), "Inconsistent number of samples and labels"
        return self

    def reweight(self, indices: pd.Series) -> Self:
        self.df, self.labels, self.protected = \
            pd.concat([self.df, self.df[indices]]), \
            pd.concat([self.labels, self.labels[indices]]), \
            [pd.concat([protected, protected[indices]])
             for protected in self.protected]

    def protected_tested(self, binary: bool = False, protected: Optional[pd.Series] = None) -> pd.Series:
        assert "MNAR" not in self.missing_type or not self.train
        protected = dm.choose_protected(self.name,
                                        self.protected if protected is None
                                        else protected)
        return protected if not binary else dm.make_protected_binary(self.name,
                                                                     protected)

    def _make_artificial(self):
        assert self.train
        if "missing" in self.extension.lower():
            path = f"data_store/{self.extension}/{self.artificial_percentage:.2f}/{
                self.missing_type}/{self.missing}/{self.name}/split{self.split}.csv"
            try:
                df, y, p = dm.load_missing(self.name, self.missing_type, True,
                                           [self.missing], [self.split])[self.name][0][0]
            except Exception:
                df, y, p = self._make_missing(self.missing)
        else:
            path = f"data_store/{self.extension}/{self.artificial_percentage:.2f}/{
                self.missing_type}/train/{self.name}/split{self.split}.csv"
            df, y, p = self._split()  # get CV split
        match self.extension:
            case "artificial_random":
                sample = df.sample(
                    frac=self.artificial_percentage, random_state=RANDOM_STATE)
                MissingGenerator(self.name, self.missing_type,
                                 self.train, sample, None).induce(1)
            case "artificial_important":
                sample = df.sample(
                    frac=self.artificial_percentage, random_state=RANDOM_STATE)
                MissingGenerator(self.name, "IMPORTANT",
                                 self.train, sample, None).induce(1)
            case "artificial" | "artificial_missing":
                protected = self.protected_tested(True, p)
                sample = df[protected == 0].sample(
                    frac=self.artificial_percentage, random_state=RANDOM_STATE)
                MissingGenerator(self.name, self.missing_type,
                                 self.train, sample, p).induce(1)
            case _:
                raise NotImplementedError
        x = pd.concat([df, sample])
        y = pd.concat([y, y.loc[sample.index]])
        p = [pd.concat([protected, protected.loc[sample.index]])
             for protected in p]
        dm.store_to_disk(
            path,
            x,
            y,
            p
        )
        return x, y, p

    def path(self, imputer: str, train: Optional[bool] = None) -> str:
        # still to be extended
        """
        path structure
            imputer -> str: name of imputer
            split -> int: indicating which split for CV
            experiment typ -> str: (some id to identify the exp)
            'how much is missing' -> int: which translates to missing
                                          percentage
            train\\test -> str: whether its train or test
            \\trainset used -> Optional(int): only in case of test data => what
                               train missing percentage has been used to train
                               the imputer
            storage/imputed/imputer/which experiment type/missing mechanism/how much is missing/train\\test/\\trainset used/split
        """
        train = self.train if train is None else train
        missing = self.trained_on\
            if self.trained_on is not None and train\
            else self.missing
        artificial_percentage = f"{self.artificial_percentage:.2f}"\
            if self.artificial_percentage is not None\
            else ""
        if imputer is not None:
            path = Path("data_store/imputed") / imputer /\
                self.extension / artificial_percentage /\
                self.missing_type / str(missing) / self.name
        path = path / "train" / f"split{self.split}.csv" if train else \
            path / "test" / str(self.trained_on) / f"split{self.split}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


class MissingGenerator:
    def __init__(self, name: str, missing_type: str,
                 is_train: bool,
                 data: pd.DataFrame, protected: Optional[list[pd.Series]]):
        self.name = name
        self.data = data
        self.protected = protected
        self.missing_type = missing_type
        self.is_train = is_train
        self.missing_mask = data.isna()
        self.rng = np.random.default_rng(RANDOM_STATE)
        assert not self.missing_mask.any().any(), \
            f"Make sure there are no initial missing values: {self.name}, {is_train} {protected is None}"
        self.unstructured_features, self.structured_features = \
            self._missing_features()
        if protected is not None:
            binary_protected = dm.choose_protected(self.name, self.protected)
            binary_protected = dm.make_protected_binary(
                self.name, binary_protected)
            self.rows = ~binary_protected
        else:
            if self.missing_type == "IMPORTANT":
                self.unstructured_features = self.structured_features
            self.rows = np.ones(data.shape[0]) == 1

    def _apply_unstructured(self, percentage: float) -> None:
        # update missing missing_mask
        unstructured = self.data[self.unstructured_features][self.rows]
        probability_mask = np.ones_like(unstructured) * percentage
        self.missing_mask.loc[self.rows, self.unstructured_features] =\
            self.rng.random(probability_mask.shape) < probability_mask
        # handle protected
        if not self.is_train and "MNAR" in self.missing_type and self.protected is not None:
            self.data.drop(chosen := dm.choose_protected(self.name), axis=1)
            self.protected = [p for p in self.protected if p.name != chosen]

    def _apply_structured(self, percentage: float) -> None:
        # check if structure is valid else return
        if self.missing_type not in ["WS_MAR", "SS_MAR", "WS_MNAR", "SS_MNAR"]:
            return
        if self.missing_type in ["SS_MAR", "SS_MNAR"]:
            percentage = 1.0
        # update missing missing_mask
        structured = self.data[self.structured_features][self.rows]
        probability_mask = np.ones_like(structured) * percentage
        # set all rows where there is no unstructured missingness to 0
        probability_mask[~(self.missing_mask.loc[self.rows, :]).any(
            axis=1).values, :] = 0.0
        self.missing_mask.loc[self.rows, self.structured_features] =\
            self.rng.random(probability_mask.shape) < probability_mask
        # handle protected
        # no need, already taken care of here

    def _missing_features(self):
        """
        Based on BN from Quy et al. 2022
        """
        match self.name:
            case "adults":
                child_protected = ["relationship", "hours-per-week",
                                   "occupation", "education", "race"]
                parent_protected = []
                parent_class = ["capital-gain", "capital-loss",
                                "age"]
            case "census":
                parent_protected = []
                child_protected = ["industry"]
                parent_class = ["weeks-worked", "occupation"]
            case "credit":
                parent_protected = []
                child_protected = ["number-people-provided-maintenance-for"]
                parent_class = ["checking-account"]
            case "bankmarketing":
                parent_protected = ["job", "education"]
                child_protected = []
                parent_class = ["duration", "poutcome", "month"]
            case "default_of_credit_cards":
                parent_protected = ["age", "marriage", "limit-bal"]
                child_protected = []
                parent_class = ["pay_3"]
            case "compas":
                parent_protected = ["score_text"]
                child_protected = []
                parent_class = ["priors_count"]
            case "diabetes":
                parent_protected = []
                child_protected = ["race"]
                parent_class = ["number_outpatient"]
            case "ricci":
                parent_protected = ["oral"]
                child_protected = []
                parent_class = ["combine"]
            case "students_math":
                parent_protected = ["higher"]
                child_protected = []
                parent_class = ["g2"]
            case "students_porto":
                parent_protected = []
                child_protected = ["schoolsup"]
                parent_class = ["g1", "g2"]
            case "toy":
                parent_protected = []
                child_protected = ["c", "cb"]
                parent_class = ["x", "xb"]
            case "communities":
                parent_protected = ["PctKids2Par", "TotalPctDiv", "PctIlleg"]
                child_protected = []
                parent_class = ["PctIlleg", "PctKids2Par"]
            case "compas":
                parent_protected = ["score_text"]
                child_protected = []
                parent_class = ["score_text", "priors_count"]
            case "mock":
                parent_protected = ["feat1", "feat5"]
                child_protected = []
                parent_class = ["feat2"]
            case _:
                raise ValueError(f"Not a Valid Dataset: {self.name}")
        unstructured = parent_protected + child_protected
        structured = parent_class
        return unstructured, structured

    def induce(self, percentage: float) -> (pd.DataFrame, list[pd.Series]):
        assert not self.missing_mask.any().any()
        self._apply_unstructured(percentage)
        self._apply_structured(percentage)
        assert self.missing_mask.any().any(), f"{self.missing_mask}"
        self.protected = [self.data[p.name]
                          for p in self.protected] if self.protected is not None else None
        self.data[self.missing_mask] = pd.NA
        return self.data, self.protected


###############################################################################
#                     Test Code after this line                               #
###############################################################################


class TestDataSet(unittest.TestCase):
    def test_path(self):
        # test different scenarios for path creation
        # so far only paths for imputed dataset version is supported
        # might extend in future
        train_set = DataSet("mock", train=True, missing=2, split=0,
                            missing_type="MAR", extension="clean")
        path_train = str(train_set.path("Simple"))
        assert path_train == "data_store/imputed/Simple/clean/MAR/2/mock/train/split0.csv", path_train
        train_set = DataSet("mock", train=False, training_param=(4, None),
                            missing=2, split=0, missing_type="MAR",
                            extension="clean")
        path_train = str(train_set.path("Simple", True))
        path_test = str(train_set.path("Simple"))
        assert path_train == "data_store/imputed/Simple/clean/MAR/4/mock/train/split0.csv", path_train
        assert path_test == "data_store/imputed/Simple/clean/MAR/2/mock/test/4/split0.csv", path_test
        train_set = DataSet("mock", train=True, missing=2, split=0,
                            missing_type="MAR", extension="artificial",
                            training_param=(3, 0.05))
        assert train_set.trained_on is None, f"{train_set.trained_on}"
        path_train = str(train_set.path("Simple"))
        assert path_train == "data_store/imputed/Simple/artificial/0.05/MAR/2/mock/train/split0.csv", path_train
        test_set = DataSet("mock", train=False, missing=2, split=0,
                           missing_type="MAR", extension="artificial",
                           training_param=(3, 0.05), missing_in_train=False)
        path_test = str(test_set.path("Simple"))
        assert path_test == "data_store/imputed/Simple/artificial/0.05/MAR/2/mock/test/0/split0.csv", path_test
        test_set = DataSet("mock", train=False, missing=2, split=0,
                           missing_type="MAR", extension="artificial_missing",
                           training_param=(2, 0.05))
        path_test = str(test_set.path("Simple"))
        assert path_test == "data_store/imputed/Simple/artificial_missing/0.05/MAR/2/mock/test/2/split0.csv", path_test

    def test_fetching(self):
        # no asserts needed, will fail if it fails to fetch the data
        DataSet("adults", train=True, missing=2, split=0, missing_type="MAR",
                extension="clean")
        DataSet("adults", train=False, missing=2, split=0, missing_type="MAR",
                extension="clean", training_param=(0, None))
        with self.assertRaises(ValueError):
            DataSet("adults", train=False, missing=2, split=0,
                    missing_type="MAR", extension="clean",
                    training_param=(0, 0.4))
        DataSet("adults", train=True, missing=0, split=0, missing_type="MAR",
                extension="artificial", training_param=(0, 0.05))
        DataSet("adults", train=True, missing=0, split=0, missing_type="MAR",
                extension="artificial_important", training_param=(0, 0.05))
        # checking if actually correct df has been fecthed
        adults_train = pd.read_csv(
            "data_store/clean/MAR/0/adults/split0.csv")
        DataSet("adults", train=True, missing=0, split=0, missing_type="MAR",
                extension="artificial_important", training_param=(0, 0.05))
        ds = DataSet("adults", train=True, missing=0, split=0, missing_type="MAR",
                     extension="clean")
        # column names were distorted when saving to disk
        mapper = {col_with_label: just_col_name for (
            col_with_label, just_col_name) in zip(adults_train.columns, ds.df.columns)}
        adults_train = adults_train.drop("::label::class", axis=1)
        adults_train = adults_train.rename(columns=mapper)
        assert (adults_train.columns == ds.df.columns).all(), "Column mismatch"
        assert adults_train.shape == ds.df.shape, f"Shape mismatch: orginial {
            adults_train.shape}, rework {ds.df.shape}"
        assert (adults_train == ds.df).all().all(), \
            f"Something went wrong fetching:\n{adults_train.columns}"\
            f"\n{ds.df.columns}"


class TestMissingGenerator(unittest.TestCase):
    def reset_data(self):
        return pd.DataFrame({
            "feat1": [0, 1, 1, 0, 0],
            "feat2": [4, 2, 4, 9, 1],
            "feat3": ["Apple", "Apple", "Banana", "Banana", "Banana"],
            "feat4": [0.33, 0.93, 0.23, 0.55, 0.01],
            "feat5": ["Apple", "Banana", "Apple", "Banana", "Orange"],
        })

    def test_unstructed(self):
        data = self.reset_data()
        gen = MissingGenerator("mock", "MAR", True, data, [
            data[f] for f in ["feat3", "feat5"]])
        gen.induce(1)
        assert (gen.missing_mask == pd.DataFrame({
            "feat1": [True, True, False, False, False],
            "feat2": [False, False, False, False, False],
            "feat3": [False, False, False, False, False],
            "feat4": [False, False, False, False, False],
            "feat5": [True, True, False, False, False],
        })).all().all(), f"{gen.missing_mask}"
        data = self.reset_data()
        gen = MissingGenerator("mock", "SS_MAR", True, data, [
            data[f] for f in ["feat3", "feat5"]])
        gen.induce(1)
        assert (gen.missing_mask == pd.DataFrame({
            "feat1": [True, True, False, False, False],
            "feat2": [True, True, False, False, False],
            "feat3": [False, False, False, False, False],
            "feat4": [False, False, False, False, False],
            "feat5": [True, True, False, False, False],
        })).all().all(), f"{gen.missing_mask}"
        data = self.reset_data()
        gen = MissingGenerator("mock", "SS_MAR", True, data, [
            data[f] for f in ["feat3", "feat5"]])
        gen._apply_unstructured(1)
        gen.missing_mask[["feat1", "feat5"]] = np.array(
            [[True, False, False, False, False], [True, False, False, False, False]]).T
        gen._apply_structured(0)
        assert (gen.missing_mask == pd.DataFrame({
            "feat1": [True, False, False, False, False],
            "feat2": [True, False, False, False, False],
            "feat3": [False, False, False, False, False],
            "feat4": [False, False, False, False, False],
            "feat5": [True, False, False, False, False],
        })).all().all(), f"{gen.missing_mask}"
        data = self.reset_data()
        gen = MissingGenerator("mock", "SS_MAR", True, data, [
            data[f] for f in ["feat3", "feat5"]])
        gen._apply_unstructured(1)
        gen.missing_mask[["feat1", "feat5"]] = np.array(
            [[True, True, False, False, False], [True, False, False, False, False]]).T
        gen._apply_structured(0)
        assert (gen.missing_mask == pd.DataFrame({
            "feat1": [True, True, False, False, False],
            "feat2": [True, True, False, False, False],
            "feat3": [False, False, False, False, False],
            "feat4": [False, False, False, False, False],
            "feat5": [True, False, False, False, False],
        })).all().all(), f"{gen.missing_mask}"

        data = self.reset_data()
        gen = MissingGenerator("mock", "IMPORTANT", True, data, None)
        gen.induce(1)
        assert (gen.missing_mask == pd.DataFrame({
            "feat1": [False, False, False, False, False],
            "feat2": [True, True, True, True, True],
            "feat3": [False, False, False, False, False],
            "feat4": [False, False, False, False, False],
            "feat5": [False, False, False, False, False],
        })).all().all(), f"{gen.missing_mask}"


if __name__ == "__main__":
    unittest.main()
