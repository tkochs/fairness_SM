from sklearn.impute import KNNImputer
from collections import defaultdict
from missforest import MissForest
# import unittest
import numpy as np
import pandas as pd
import miceforest as mf


def get_imputers(kwargs=defaultdict(dict)):
    return {
        "Simple": Simple(**kwargs["Simple"]) if "Simple" in kwargs
        else Simple(),
        "KNN": KnnImputer(**kwargs["KNN"]) if "KNN" in kwargs
        else KnnImputer(),
        "MissForest": MFimp(**kwargs["MissForest"]) if "MissForest" in kwargs
        else MFimp(),
        "Mice": Mice(**kwargs["Mice"]) if "Mice" in kwargs
        else Mice(random_state=42),
    }


class Imputer:
    def get_params(self, *args, **kwargs) -> dict:
        return dict()

    def fit_transform(self, data):
        return self.fit(data).transform(data)


class Simple(Imputer):
    name: str = "Simple"

    def fit(self, data: pd.DataFrame):
        self.categorical_columns = data.select_dtypes(
            exclude=["number"]).columns
        self.numeric_columns = data.select_dtypes(
            include=['number']).columns
        self.means = data[self.numeric_columns].mean()
        if len(self.categorical_columns) == 0:
            self.modes = None
        else:
            self.modes = data[self.categorical_columns].mode().iloc[0]
        return self

    def transform(self, data: pd.DataFrame):
        # Reconstruct the DataFrame with imputed values
        X_imputed = data.copy(deep=True)
        X_imputed[self.numeric_columns] = data[self.numeric_columns].fillna(
            self.means)
        if self.modes is not None:
            X_imputed[self.categorical_columns] = data[self.categorical_columns].fillna(
                self.modes)
        return X_imputed


class Mice(Imputer):
    name = "Mice"

    def __init__(self, iterations=5, mean_match_candidates=6, random_state=None):
        self.iterations = iterations
        self.mean_match_candidates = mean_match_candidates
        self.random_state = random_state

    def _to_float(self, x):
        cols = x.select_dtypes(include="number").columns
        x[cols] = x[cols].astype(float)

    def fit(self, x):
        # store categoric columns and move them to the dtype category
        x = x.copy()
        self.cols = x.columns
        self.cat_columns = x.select_dtypes(exclude="number").columns
        x[self.cat_columns] = x[self.cat_columns].astype("category")
        self.cat_types = x[self.cat_columns].dtypes
        self._to_float(x)
        self.imputer = mf.ImputationKernel(
            x,
            mean_match_candidates=self.mean_match_candidates,
            random_state=self.random_state,
            variable_schema=x.columns.to_list()
        )
        self.min_data = max(int(np.ceil(x.shape[0] * 0.009)), 15)
        self.imputer.mice(
            self.iterations,
            boosting="gbdt",
            min_sum_hessian_in_leaf=0.65,
            min_data_in_leaf=self.min_data,
        )
        return self

    def transform(self, x):
        assert isinstance(x, (pd.DataFrame, np.array)), f"{type(x)}"
        # store categoric columns and move them to the dtype category
        x = x.copy()[self.cols]
        self._to_float(x)
        x[self.cat_columns] = x[self.cat_columns].astype(self.cat_types)
        return self.imputer.impute_new_data(x, random_state=self.random_state) \
            .complete_data()

    def get_params(self, *args, **kwargs):
        return {
            "iterations": self.iterations,
            "mean_match_candidates": self.mean_match_candidates,
            "random_state": self.random_state
        }


class MFimp(Imputer):
    name = "MissForest"

    def __init__(self):
        self.encoder = None
        self.miss = None

    class RollingLabelEncoder:
        def __init__(self):
            self.encoder = dict()
            self.decoder = dict()

        def _check_array(self, x):
            x = np.array(x)
            assert len(x.shape) == 1 or (x.shape[1] == 1 and len(
                x.shape) <= 2), "Only pass 1D arrays"

        def fit(self, x):
            self._check_array(x)
            i = 0
            for e in x:
                if e in self.encoder:
                    continue
                self.encoder[e] = i
                self.decoder[i] = e
                i += 1
            self._is_fitted = True
            return self

        def extend(self, x):
            self._check_array(x)
            last_label = next(reversed(self.decoder))
            i = last_label + 1
            for e in x:
                if e not in self.encoder:
                    self.encoder[e] = i
                    self.decoder[i] = e
                    i += 1
            return self

        def transform(self, x):
            assert hasattr(self, "_is_fitted") and self._is_fitted
            self._check_array(x)
            return np.array([self.encoder[e] for e in x])

        def inverse_transform(self, x):
            assert hasattr(self, "_is_fitted") and self._is_fitted
            self._check_array(x)
            return np.array([self.decoder[e] for e in x])

    def _revert_label(self, x) -> None:
        """
        ATTENTION: Mutates 'x'
        """
        missing_mask = x.isna()
        if self.encoder is None:
            raise NotImplementedError
        decoded = pd.DataFrame(
            {col: self.encoder[col].inverse_transform(x[col]) for col in self.cat_columns})
        x[self.cat_columns] = decoded
        x[missing_mask] = np.nan

    def _label_encoding(self, x) -> None:
        """
        ATTENTION: Mutates 'x'
        """
        missing_mask = x.isna()
        if self.encoder is None:
            self.encoder = {col: self.RollingLabelEncoder().fit(
                x[col]) for col in self.cat_columns}
        try:
            encoded = pd.DataFrame(
                {col: self.encoder[col].transform(x[col]) for col in self.cat_columns})
        except Exception:
            self.encoder = {col: self.encoder[col].extend(
                x[col]) for col in self.cat_columns}
            encoded = pd.DataFrame(
                {col: self.encoder[col].transform(x[col]) for col in self.cat_columns})
        x[self.cat_columns] = encoded
        x[missing_mask] = np.nan

    def fit(self, x):
        # store categoric columns and move them to the dtype category
        x = x.copy()
        self.cols = x.columns
        print(self.cols)
        self.cat_columns = x.select_dtypes(exclude="number").columns
        self.num_columns = x.select_dtypes(include="number").columns
        x[self.num_columns] = x[self.num_columns].astype(float)
        self.imputer = MissForest(
            categorical=list(self.cat_columns)
        )
        self._label_encoding(x)
        if x.isna().sum().sum() != 0:
            self.imputer.fit(x=x)
        else:
            self.imputer = None
        self._is_fitted = True
        return self

    def transform(self, x):
        # store categoric columns and move them to the dtype category
        print(x.columns)
        assert hasattr(self, "_is_fitted") and \
            self._is_fitted is True, "Call fit before using Transform"
        if self.imputer is None or x.isna().sum().sum() == 0:
            return x
        x = x[self.cols]
        self._label_encoding(x)
        imputed = self.imputer.transform(x=x)
        self._revert_label(imputed)
        # idk imputing shuffeled labels hope this fixes it lol
        imputed = imputed[self.cols]
        return imputed


class KnnImputer(Imputer):
    name = "KNN"

    def __init__(self, k=5):
        self.k = k
        self.imputer = KNNImputer(n_neighbors=k)

    def fit(self, x: pd.DataFrame):
        self.data = x
        self.cols = x.columns
        x = one_hot_encode_dataframe(x, drop_nans=True)
        self.encoded_cols = x.columns
        self.imputer.fit(x)
        return self

    def transform(self, df):
        df = one_hot_encode_dataframe(df, drop_nans=True)
        not_in_test = [
            col for col in self.encoded_cols if col not in df.columns]
        df = pd.concat([df, pd.DataFrame(np.zeros((df.shape[0], len(
            not_in_test))), columns=not_in_test)], axis=1)[self.encoded_cols]
        imputed = self.imputer.transform(df)
        return reverse_one_hot_encoding(pd.DataFrame(imputed, columns=self.encoded_cols), self.cols)[self.cols]

    def get_params(self, *args, **kwargs):
        return {"k": self.k}


def reverse_one_hot_encoding(
        encoded_df: pd.DataFrame,
        original_columns: pd.Index,
        missing_token: str = "missing") -> pd.DataFrame:
    """
    Reverses the one-hot encoding for specified columns, and leaves other columns unchanged.

    Args:
    - encoded_df (pd.DataFrame): DataFrame containing one-hot encoded columns along with non-encoded columns.
    - original_columns (list): List of original column names corresponding to the one-hot encoded columns.

    Returns:
    - pd.DataFrame: DataFrame with one-hot encoded columns reversed and other columns unchanged.
    """
    assert isinstance(encoded_df, pd.DataFrame), "Not a DataFrame"
    assert isinstance(original_columns, pd.Index), "Not a Column Index"
    # Iterate through the original columns to reverse encoding only for those
    # columns
    # Make a copy to avoid modifying the original DataFrame
    decoded_df = encoded_df.copy()

    for column in original_columns:
        # Identify columns in the dataframe that correspond to the one-hot encoding
        one_hot_columns = [col for col in encoded_df.columns if
                           col.startswith(column + "_") and col[len(column):] != ""]

        # Check if the column is present in the dataframe
        if one_hot_columns:
            # Reverse the one-hot encoding by finding the column with the value 1 in each row
            nan_rows = encoded_df[one_hot_columns].isna().any(axis=1)
            decoded_df[column] = ""
            decoded_df.loc[~nan_rows,
                           column] = encoded_df.loc[~nan_rows, one_hot_columns].idxmax(axis=1)
            decoded_df[column] = decoded_df[column].apply(
                lambda x: x.split('_')[-1])
            decoded_df.loc[decoded_df[column] == missing_token, column] = pd.NA
            decoded_df.loc[nan_rows, column] = pd.NA

            # Drop the one-hot encoded columns after decoding
            decoded_df.drop(columns=one_hot_columns, inplace=True)
    return decoded_df[original_columns]


def one_hot_encode_dataframe(
        dataframe: pd.DataFrame,
        *args,  # prevent setting drop_nans by accident
        drop_nans=False) -> pd.DataFrame:
    """
    Automatically detects categorical columns in each DataFrame and performs one-hot encoding.

    Parameters:
        dataframes (list of pd.DataFrame): List of DataFrames to process.

    Returns:
        list of pd.DataFrame: List of DataFrames with one-hot encoded categorical columns.
    """
    assert isinstance(dataframe, pd.DataFrame)
    assert len(args) == 0, "Too many positional Attributes"
    # assert drop_nans ^ encode_nans, "Set either drop or encode to True and refrain from setting both to True"

    df = dataframe.copy()
    # Detect categorical columns
    categorical_columns = df.select_dtypes(
        exclude=['number']).columns

    # Handle missing values in categorical columns (fill with 'missing')
    df[categorical_columns] = df[categorical_columns].fillna('missing')
    # Perform one-hot encoding
    encoded_df = pd.get_dummies(
        df, columns=categorical_columns, drop_first=False)
    if drop_nans:
        for column in categorical_columns:
            if column+"_missing" in encoded_df.columns:
                mm = encoded_df[column + "_missing"]
                one_hot_columns = [col for col in encoded_df.columns if
                                   col.startswith(column + "_")]
                encoded_df[one_hot_columns] *= 1.0
                encoded_df.loc[mm, one_hot_columns] = pd.NA
    return encoded_df
