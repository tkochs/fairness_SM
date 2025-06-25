from pathlib import Path
from functools import lru_cache
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from aif360.datasets import BinaryLabelDataset
from typing import Literal, Union
from ucimlrepo import fetch_ucirepo
from sklearn.datasets import make_classification
from scipy.stats import entropy, gaussian_kde
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import f_classif, SelectKBest
import os
import pickle
import unittest
import pandas as pd
import numpy as np
import logging


MISSING_TYPES = {
    "MCAR",
    "MAR",
    "MNAR",
    "WS_MAR",
    "WS_MNAR",
    "SS_MAR",
    "SS_MNAR",
}

STORAGE = Path("data_store/")


def split_ds(ds: dict, target: str, protected) -> tuple[pd.DataFrame, pd.Series, list[pd.Series]]:
    assert isinstance(ds, dict), f"Got instead {type(ds)}"
    assert isinstance(target, str)
    assert isinstance(protected, (str, list))
    return (ds["features"], ds["targets"][target], ds["features"][protected])


def get_toy() -> tuple[pd.DataFrame, pd.Series, list[pd.Series]]:
    path = STORAGE / "datasets/toy.csv"
    return load_data("toy", complete_path=path)["toy"]


@lru_cache
def get_adults() -> tuple[pd.DataFrame, pd.Series, list[pd.Series]]:
    ds = dict(get_dataset(id=2)["data"])
    protected = ["sex", "age", "race"]
    v, t, p = split_ds(ds, "income", protected)
    return v, t, [p[col] for col in protected]


def get_compas() -> tuple[pd.DataFrame, pd.Series, list[pd.Series]]:
    # target name: 'readmitted'
    path = STORAGE / "datasets/compas.csv"
    x, y, p = load_data("compas", path)["compas"]
    # remove all columns with more than ~40% missing values
    cols = x.isna().sum()/len(x) > 0.39
    cols = x.columns[cols]
    x = x.drop(cols, axis=1)
    return x, y, p


@lru_cache
def get_diabetes() -> tuple[pd.DataFrame, pd.Series, list[pd.Series]]:
    # target name: 'readmitted'
    ds = dict(get_dataset(id=296)["data"])
    x, y, p = split_ds(ds, "readmitted", "gender")
    # remove all columns with more than ~40% missing values
    for col, percentage in (x.isna().sum()/len(x)).items():
        if percentage > 0.39:
            x = x.drop(col, axis=1)
    return x, y, p


@lru_cache
def get_census() -> tuple[pd.DataFrame, pd.Series, list[pd.Series]]:
    # target name: 'income'
    ds = dict(get_dataset(id=117)["data"])
    protected = ["sex", "race"]
    ds["features"] = ds["features"].rename(columns={
        "AAGE": "age",
        "ACLSWKR": "workclass",
        "ADTINK": "industry",
        "ADTOCC": "occupation",
        "AHGA": "education",
        "AHSCOL": "enroll-in-edu-inst-last-wk",
        "AMARITL": "marital-status",
        "AMJIND": "major-industry",
        "AMJOCC": "major-occupation",
        "ARACE": "race",
        "AREORGN": "hispanic-origin",
        "ASEX": "sex",
        "AUNEM": "member-of-labor-union",
        "AUNTYPE": "reason-unemployment",
        "AWKSTAT": "full-or-parttime-employment-stat",
        "CAPGAIN": "capital-gains",
        "GAPLOSS": "capital-losses",
        "DIVVAL": "dividends-from-stocks",
        "FILESTAT": "tax-filer-status",
        "GRINREG": "region-previous-residence",
        "GRINST": "state-previous-residence",
        "HHDFMX": "detailed-household-and-family-stat",
        "HHDREL": "detailed-household-summary-in-household",
        "MARSUPWRT": "instance-weight",
        "MIGMTR1": "migration-code-change-in-msa",
        "MIGMTR3": "migration-code-change-in-reg",
        "MIGMTR4": "migration-code-move-within-reg",
        "MIGSAME": "live-house-1-year-ago",
        "MIGSUN": "migration-in-prev-res-in-sunbelt",
        "NOEMP": "num-persons-worked-for-employer",
        "PARENT": "family-members-under-18",
        "PEFNTVTY": "country-of-birth-father",
        "PEMNTVTY": "country-of-birth-mother",
        "PENATVTY": "country-of-birth-self",
        "PRCITSHP": "citizenship",
        "SEOTR": "own-business",
        "VETQVA": "fil-questionare",
        "VETYN": "veterans-benefits",
        "WKSWORK": "weeks-worked",
        "AHRSPAY": "wage-per-hour",
    })
    v, t, p = split_ds(ds, "income", protected)
    # columns with extreme number of missing values
    v = v.drop(["migration-code-change-in-msa",
                "migration-code-change-in-reg",
                "migration-code-move-within-reg",
                "migration-in-prev-res-in-sunbelt"], axis=1)
    return v, t, [p[col] for col in protected]


@lru_cache
def get_bankmarketing() -> tuple[pd.DataFrame, pd.Series, list[pd.Series]]:
    ds = dict(get_dataset(id=222)["data"])
    protected = ["marital", "age"]
    v, t, p = split_ds(ds, "y", protected)
    return v, t, [p[col] for col in protected]


@lru_cache
def get_creditcard() -> tuple[pd.DataFrame, pd.Series, list[pd.Series]]:
    ds = dict(get_dataset(id=144)["data"])
    protected = ["personal-status-and-sex", "age"]
    ds["features"] = ds["features"].rename(columns={
        "Attribute1": "checking-account",
        "Attribute2": "duration",
        "Attribute3": "credit-history",
        "Attribute4": "purpose",
        "Attribute5": "credit-amount",
        "Attribute6": "savings-account",
        "Attribute7": "employment-since",
        "Attribute8": "installment-rate",
        "Attribute9": "personal-status-and-sex",
        "Attribute10": "other-debtors",
        "Attribute11": "residence-since",
        "Attribute12": "property",
        "Attribute13": "age",
        "Attribute14": "other-installment",
        "Attribute15": "housing",
        "Attribute16": "existing-credits",
        "Attribute17": "job",
        "Attribute18": "number-people-provided-maintenance-for",
        "Attribute19": "telephone",
        "Attribute20": "foreign-worker",
    })
    status_map = {
        "A91": "male : divorced/separated",
        "A92": "female : divorced/separated/married",
        "A93": "male : single",
        "A94": "male : married/widowed",
        "A95": "female : single",
    }
    ds["features"].loc[:, protected[0]] =\
        ds["features"][protected[0]].replace(status_map)
    v, t, p = split_ds(ds, "class", protected)
    return v, t, [p[col] for col in protected]


@lru_cache
def get_communitiesandcrime() -> tuple[pd.DataFrame, pd.Series, list[pd.Series]]:
    # target name: 'ViolentCrimeProp'
    ds = dict(get_dataset(id=183)["data"])
    return split_ds(ds, "ViolentCrimesPerPop", "racepctblack")


@lru_cache
def get_students() -> tuple[pd.DataFrame, pd.Series, list[pd.Series]]:
    ds = dict(get_dataset(id=320)["data"])
    protected = ["sex", "age"]
    v, t, p = split_ds(ds, "class", protected)
    return v, t, [p[col] for col in protected]


@lru_cache
def get_default_of_credit_cards() -> tuple[pd.DataFrame, pd.Series, list[pd.Series]]:
    # target name: 'ViolentCrimeProp'
    ds = dict(get_dataset(id=350)["data"])
    protected = ["sex", "education", "marriage"]
    ds["features"] = ds["features"].rename(columns={
        "X1": "limit-bal",
        "X2": "sex",
        "X3": "education",
        "X4": "marriage",
        "X5": "age",
        "X6": "pay_0",
        "X7": "pay_2",
        "X8": "pay_3",
        "X9": "pay_4",
        "X10": "pay_5",
        "X11": "pay_6",
        "X12": "bill_amt1",
        "X13": "bill_amt2",
        "X14": "bill_amt3",
        "X15": "bill_amt4",
        "X16": "bill_amt5",
        "X17": "bill_amt6",
        "X18": "pay_amt1",
        "X19": "pay_amt2",
        "X20": "pay_amt3",
        "X21": "pay_amt4",
        "X22": "pay_amt5",
        "X23": "pay_amt6",
    })
    ds["features"].loc[:, protected[0]] =\
        ds["features"][protected[0]].replace({
            1: "male",
            2: "female",
        })
    v, t, p = split_ds(ds, "Y", protected)
    return v, t, [p[col] for col in protected]


def get_dataset(id: int, cache_dir: str = STORAGE / "datasets") -> dict:
    """
    Fetch a dataset by ID, using a local cache to minimize API calls.

    Args:
        dataset_id (int): The ID of the dataset to fetch.
        cache_dir (str): Directory to store cached datasets.

    Returns:
        dict: The dataset information as returned by fetch_ucirepo.
    """
    assert isinstance(id, int)
    assert isinstance(cache_dir, str)
    # Ensure the cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    # Path for the cached dataset
    cache_path = os.path.join(cache_dir, f"dataset_{id}.pkl")
    # Check if the dataset is already cached
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as cache_file:
            dataset = pickle.load(cache_file)
        return dataset
    # Fetch the dataset using the API
    dataset = fetch_ucirepo(id=id)
    # Cache the dataset for future use
    with open(cache_path, "wb") as cache_file:
        pickle.dump(dataset, cache_file)
    return dataset


def make_label_binary(ds: str, labels: pd.Series) -> pd.Series:
    """
    The fairness metrics except binary classification problems
    (Made Binary according to Quy et. al 2022)

    Args:
        ds (str): Name of the dataset
        labels (Series): Labels to for the classification/regression
    """
    assert isinstance(labels, (pd.DataFrame, pd.Series))
    assert isinstance(ds, str)
    if ds.endswith("_clean"):
        ds = ds[:-len("_clean")]
    match ds:
        # case for datasets with already binary labels
        case "adults" | "compas" | "census" | "bankmarketing" | "credit" | \
                "ricci" | "heart_disease" | "default_of_credit_cards" | \
                "students_math" | "students_porto" | "toy":
            return labels
        case "diabetes":
            positive = labels == "<30"
            negative = ~positive
            labels = labels.copy()
            labels[positive] = "positive"
            labels[negative] = "negative"
            return labels
        case "communities":
            high_crime = labels > 0.7
            low_crime = labels <= 0.7
            labels = labels.copy().astype(str)
            labels[high_crime] = "high crime"
            labels[low_crime] = "low crime"
            return labels
        case _:
            raise NotImplementedError(f"Dataset ({ds}) not Implemented yet!")


def make_protected_binary(dataset, attribute) -> list[bool]:
    """
    Makes the protected attribute binary
    1 => Majority Class
    0 => Minority Class

    Args:
        dataset (str): Name of Dataset
        attribute (Series): protected attribute as Series

    Returns:
        Binary version of attribute
    """
    assert isinstance(attribute, (pd.DataFrame, pd.Series))
    assert isinstance(dataset, str)
    assert not attribute.isna().any(), \
        "Make sure there arent any missing values in the protected attribute"
    if dataset.endswith("_clean"):
        dataset = dataset[:-len("_clean")]
    match dataset:
        case "adults":
            if attribute.name == "sex":
                return attribute == "Male"
            if attribute.name == "race":
                return attribute == "White"
        case "diabetes":
            return attribute == "Male"
        case "compas":
            if attribute.name == "sex":
                return attribute == "Male"
            if attribute.name == "race":
                return attribute == "Caucasian"
        case "census":
            return attribute == "White"
        case "bankmarketing":
            return attribute == "married"
        case "credit":
            if attribute.name == "personal-status-and-sex":
                replacement_map = {
                    "male : single": "1",
                    "male : married/widowed": "1",
                    "male : divorced/separated": "1",
                    "female : divorced/separated/married": "0",
                    "female : single": "0",
                }
                return attribute.replace(replacement_map).astype(int)
            if attribute.name == "age":
                return attribute > 25
        case "communities":
            # According to Quy et al. we will use 0.06 as threshold
            return attribute < 0.06
        case "ricci":
            return attribute == "W"
        case "default_of_credit_cards":
            return attribute == "male"
        case "students_math":
            if attribute.name == "sex":
                return attribute == "M"
            if attribute.name == "age":
                return attribute < 18
        case "students_porto":
            if attribute.name == "sex":
                return attribute == "M"
            if attribute.name == "age":
                return attribute < 18
#        case "heart_disease":
#            raise NotImplementedError("WTF is this feature??!?!?!?!")
        case "toy":
            if attribute.name == "pb":
                return attribute == 1
        case "mock":
            if attribute.name == "feat3":
                return attribute == "Banana"
        case _:
            raise NotImplementedError(
                f"{dataset}:{attribute.name} is not known yet\n{attribute[:5]}")
    raise ValueError


@lru_cache
def get_datasets(path="datasets.txt"):
    with open(path) as f:
        datasets = f.read().splitlines()
    return [ds.strip() for ds in datasets if ds.strip()[0] != '#']


def get_k_significant_features(
        data: pd.DataFrame,
        labels: pd.Series,
        k=3,
        metric=f_classif):
    """
    Compute k most relevant features of the datasets
    (using some classifier independent metrics)
    """
    def label_encode_categorical(data: pd.DataFrame) -> pd.DataFrame:
        encoder = LabelEncoder()
        categorical = data.select_dtypes(include=["object", "category"])\
            .columns
        for col in categorical:
            data[col] = encoder.fit_transform(data[col])
        return data

    values = data.copy()
    values = label_encode_categorical(values)
    # just use features with more than one unique value
    values = values.loc[:, values.nunique() > 1]
    selected = SelectKBest(metric, k=k).fit(values, labels).get_support()
    return values.columns[selected]


def get_mt_significant(ds_name: str) -> list:
    storage = STORAGE / "artificial_significant/0.05"
    ks = os.listdir(storage)
    ks.sort(key=lambda x: int(x[len("SIGNIFICANT_"):]))
    mts = []
    for k in ks:
        if ds_name in os.listdir(f"{storage}/{k}/train"):
            mts.append(k)
    return mts


def get_important_features(name_of_dataset: str, data: pd.DataFrame = None):
    """
    Args:
        name_of_dataset (str): Name of Dataset
        data (DataFrame or None): DataFrame

    Returns:
        list[important feaures] or (list[important_features], DataFrame)

    by important we assume connected to the label in BN
    (According to Quy et. al 2022)
    """
    match name_of_dataset:
        case "adults":
            parent_of_label = ["capital-gain", "capital-loss", "age"]
        case "census":
            parent_of_label = ["weeks-worked", "occupation"]
        case "credit":
            parent_of_label = ["checking-account"]
        case "bankmarketing":
            parent_of_label = ["duration", "poutcome", "month"]
        case "default_of_credit_cards":
            parent_of_label = ["pay_3"]
        case "compas":
            parent_of_label = ["priors_count"]
        case "diabetes":
            parent_of_label = ["number_outpatient"]
        case "ricci":
            parent_of_label = ["combine"]
        case "students_math":
            parent_of_label = ["g2"]
        case "students_porto":
            parent_of_label = ["g1", "g2"]
        case "toy":
            parent_of_label = ["x", "xb"]
        case "communities":
            parent_of_label = ["PctIlleg", "PctKids2Par"]
        case _:
            raise ValueError(
                f"Not a Valid Dataset: {name_of_dataset}")
    return parent_of_label if data is None else (parent_of_label, data[parent_of_label])


def _choose_protected(dataset_name: str) -> str:
    match dataset_name:
        case "adults" | "census" | "default_of_credit_cards" | "students_math"\
                | "students_porto":
            return "sex"
        case "bankmarketing":
            return "marital"
        case "diabetes":
            return "gender"
        case "ricci" | "compas":
            return "race"
        case "credit":
            return "personal-status-and-sex"
        case "toy":
            return "pb"
        case "communities":
            return "racepctblack"
        case "mock":
            return "feat3"
        case _:
            raise NotImplementedError(
                f"Please select a protected attribute for: [{dataset_name}] ")


def choose_protected(
        dataset_name: str,
        protected_attributes: object = None) -> pd.Series:
    assert isinstance(dataset_name, str)
    if protected_attributes is None:
        return _choose_protected(dataset_name)
    assert isinstance(protected_attributes,
                      (list, pd.Series, pd.DataFrame, tuple)), f"{type(protected_attributes)}"
    if not isinstance(protected_attributes, (list, tuple)):
        protected_attributes = [protected_attributes]
    if dataset_name.endswith("_clean"):
        dataset_name = dataset_name[:-len("_clean")]
    for attribute in protected_attributes:
        match dataset_name:
            case "adults" | "census" | "default_of_credit_cards" | \
                    "students_math" | "students_porto":
                if attribute.name.lower() == "sex":
                    return attribute
            case "bankmarketing":
                if attribute.name.lower() == "marital":
                    return attribute
            case "diabetes":
                if attribute.name.lower() == "gender":
                    return attribute
            case "ricci" | "compas":
                if attribute.name.lower() == "race":
                    return attribute
            case "credit":
                if attribute.name.lower() == "personal-status-and-sex":
                    return attribute
            case "toy":
                if attribute.name.lower() == "pb":
                    return attribute
            case "communities":
                if attribute.name.lower() == "racepctblack":
                    return attribute
            case "mock":
                if attribute.name.lower() == "feat3":
                    return attribute
            case _:
                raise NotImplementedError(f"Please select a protected attribute for: "
                                          f"[{dataset_name}] from {[p.name for p in protected_attributes]}")
    raise ValueError(
        f"There was not protected attribute choosen for [{dataset_name}]")


def store_to_disk(path, df, label, protected, overwrite=False) -> None:
    if overwrite:
        if os.path.exists(path):
            os.remove(path)
    if os.path.exists(path) and not overwrite:
        return
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    protected = protected if isinstance(
        protected, (list, tuple)) else [protected]
    protected_columns = [p.name for p in protected]
    df = df.rename(columns={p: "::protected::"+p for p in protected_columns})
    df["::label::"+label.name] = label
    df.to_csv(path, index=False)


@lru_cache
def fetch_test(ds, fold_idx, clean=False, mt=None, iteration=None):
    assert not isinstance(ds, (list, tuple)) and isinstance(ds, str)
    if clean:
        pre = "clean/"
    else:
        pre = "raw/"
    if mt is None:
        pre += "splits/"
        return load_data(ds, prefix=pre, suffix=f"/test/split{fold_idx}", storage=STORAGE)[ds]
    else:
        assert mt is not None, "Specify the type of missingness!"
        assert iteration is not None, "Specify how much of the data is missing!"
        return load_data(
            ds,
            prefix=f"{pre}{mt}/test/{iteration}/",
            suffix=f"/split{fold_idx}",
            storage=STORAGE)[ds]


def fetch_imputed(ds: str,
                  imputation_method: str,
                  clean: str,
                  mt: str,
                  iteration: str,
                  test: str,
                  split_id: str
                  ):
    iteration = str(iteration)
    split_id = str(split_id)
    args = [ds, imputation_method, clean, mt, iteration, test, split_id]
    for arg in args:
        assert isinstance(arg, str), f"Arg: {arg} with type {type(arg)}"
    ds, imputation_method, clean, mt, iteration, test, split_id = args
    iteration = "" if ("artificial" in clean and test == "train") \
        else f"{iteration}/"
    mt = "IMPORTANT" if "important" in iteration and test == "train" else mt
    return load_data(
        ds,
        prefix=f"imputed/{imputation_method}/{clean}/{mt}/{iteration}",
        suffix=f"/{test}/split{split_id}",
        storage=STORAGE)[ds]


def fetch_artificial(
        ds: Union[str, list],
        mt: str,
        percentage="",
        kind: str = "artificial") -> (pd.DataFrame, pd.Series, list):
    kind = kind.lower()
    if percentage != "":
        percentage = f"{percentage:.2f}/" if isinstance(percentage, float)\
            else f"{percentage}/"
    match kind:
        case "artificial" | "artificial_label" | "artificial_random":
            return load_split(ds, f"{kind}/{percentage}{mt}/", split_type="")
        case "artificial_missing":
            return load_missing(ds, mt, prefix=f"{kind}/{percentage}")
        case "artificial_important" | "artificial_significant":
            assert "IMPORTANT" in mt or "SIGNIFICANT_" in mt, \
                "Please check what misssing mechanism you are passing"
            return load_split(ds, f"{kind}/{percentage}{mt}/", split_type="")
        case _:
            raise NotImplementedError(f"Please check: {kind}")


def load_split(ds, clean=False, split_type="train", n_cv_splts=5):
    assert isinstance(clean, (str, bool))
    if split_type in {"test", "train"}:
        split_type = split_type + "/"
    ds = ds if isinstance(ds, (list, tuple)) else [ds]
    if isinstance(clean, bool):
        prefix = "clean/" if clean else "raw/"
        prefix = prefix + "splits/"
    else:
        prefix = clean + "train/"
    data = {name: [load_data(name, prefix=prefix, suffix=f"/{split_type}split{i}", storage=STORAGE)[
        name] for i in range(n_cv_splts)] for name in ds}
    return data


def load_missing(ds, mt, clean=False, samples=range(5), splits=range(5), prefix=None):
    """
    Returns dict
    {name -> Dataset [split_id] [iteration_id]}
    iteration_id === missingness percentage (0 -> 0%, 1 -> 10%,
                                             2 -> 25%, 3 -> 50%, 4 -> 75%)
    """
    if not isinstance(ds, (list, tuple)):
        ds = [ds]
    suffix = "/split"
    if prefix is None:
        prefix = "clean/" if clean else "raw/"
    data = {name: [[load_data(name, prefix=f"{prefix}{mt}/{j}/", suffix=f"{suffix}{i}", storage=STORAGE)[name]
                    for j in samples] for i in splits] for name in ds}
    return data


def load_data(
        names,
        prefix="raw/full/",
        suffix="",
        storage=STORAGE,
        *,
        complete_path=None) -> dict[pd.DataFrame]:
    if not isinstance(names, (list, tuple)):
        names = [names]
    dataframes = dict()
    for name in names:
        path = f"{storage}/{prefix}{name}{suffix}.csv" if complete_path is None\
            else complete_path
        assert os.path.exists(path), f"DataSet {path} doesn't exist"
        df = pd.read_csv(path)
        protected_cols = [
            c for c in df.columns if c.startswith("::protected::")]
        labels_col = [c for c in df.columns if c.startswith("::label::")]
        assert len(labels_col) == 1, "More then one label found"
        labels_col = labels_col[0]
        labels = pd.Series(df[labels_col])
        labels.name = labels_col[len("::label::"):]
        protected = [pd.Series(df[p]) for p in protected_cols]
        for p in protected:
            p.name = p.name[len("::protected::"):]
        df = df.drop(labels_col, axis=1)
        df = df.rename(columns={p: p[len("::protected::"):]
                                for p in protected_cols})
        dataframes[name] = (df, labels, protected)
    return dataframes


def fetch_data(path):
    path = str(path)
    if not path.startswith("data_store"):
        path = "data_store/" + path
    if not path.endswith(".csv"):
        path = path + ".csv"
    path = Path(path)
    return load_data(None, complete_path=path)[None]


class TestClass(unittest.TestCase):
    def test_read_datasets(self):
        datasets = get_datasets("datasets.txt")
        assert "#testcase" not in datasets, \
            "Fetched commented Datatset"
        assert "adults" in datasets, \
            "Didn't Fetch Adults Dataset"

    def test_fetch_data(self):
        fetch_data("datasets/toy")


if __name__ == "__main__":
    unittest.main()
