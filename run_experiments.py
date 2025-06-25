import sys
import argparse
from sklearn.base import clone

from experiment import Experiment
from classifiers import XGBoost, Logistic
from utils import run_parallel

MISSING_TYPES = [
    # "MCAR",
    "MAR",
    "MNAR",
    "WS_MAR",
    "WS_MNAR",
    "SS_MAR",
    "SS_MNAR"]

MAR = ["MAR",
       # "WS_MAR",
       "SS_MAR"]

IMPUTER_VALID = [
    "Simple",
    "Mice",
    "MissForest",
    "KNN",
    "Removed"
]

AVAILABLE_DATASETS = {"adults", "diabetes", "bankmarketing", "default_of_credit_cards",
                      "census", "credit", "students_math", "students_porto", "toy", "ricci"}
# ds = "ricci"
# MODEL = Logistic()
MODEL = XGBoost()


def run_all():
    print("running everything all at once")
    tasks = []
    for ds in AVAILABLE_DATASETS:
        for imp in IMPUTER_VALID:
            tasks.append(lambda ds=ds, imp=imp: run_exp(ds, imp))
    run_parallel(tasks)
    sys.exit()


def run_exp(ds, imp):
    # Experiment("imputed", ds, imp, clone(MODEL), MISSING_TYPES).run()
    Experiment("missing_imputed", ds, imp, clone(MODEL), MISSING_TYPES,
               test_sets=range(5)).run()
    # Experiment("missing_imputed_test_only", ds, imp, clone(MODEL),
    #            MISSING_TYPES, test_sets=range(5)).run()
    # Experiment("imputed_reweighting", ds, imp,
    #            clone(MODEL), MAR, reweighting=True).run()
    Experiment("missing_imputed_reweighting", ds, imp, clone(MODEL), MAR,
               test_sets=range(5), reweighting=True).run()
    # Experiment("imputed_postprocess", ds, imp,
    #            clone(MODEL), MAR, postprocessed=True).run()
    Experiment("missing_imputed_postprocess", ds, imp, clone(MODEL), MAR,
               test_sets=range(5), postprocessed=True).run()
    # Experiment("missing_imputed_test_only_postprocess", ds, imp, clone(MODEL),
    #            MAR, test_sets=range(5), postprocessed=True).run()
    # Experiment("artificial", ds, imp, clone(MODEL), MAR).run()
    # Experiment("artificial_important", ds, imp, clone(MODEL), MAR).run()
    Experiment("artificial_missing", ds, imp, clone(MODEL), MAR).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("imputer", nargs='+', help="List of items to process")
    parser.add_argument("--ds")
    parser.add_argument("--id")
    args = parser.parse_args()
    ds = args.ds
    if args.ds is None:
        run_all()
    assert ds in AVAILABLE_DATASETS, f"Not a valid dataset {ds}"
    for imp in args.imputer:
        assert imp in IMPUTER_VALID, f"not valid {imp}"
        try:
            run_exp(ds, imp)
        except Exception as e:
            print(f"Not possible: imp={imp}, ds={ds} {e}")
            continue
