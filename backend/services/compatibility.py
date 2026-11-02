"""
FR-02 / DR-03: does a model's expected input match a dataset's feature columns?

Works from the model's recorded inputs, so it can run at validation time
(UC-04: "if a dataset was uploaded previously, check compatibility") as well
as right before an evaluation.
"""


def check(feature_names, n_features, feature_columns):
    """
    Return (ordered_features, reason). ordered_features is None when the pair is
    incompatible, and reason then says why.
    """
    if feature_names is not None:
        expected = list(feature_names)
        missing = [c for c in expected if c not in feature_columns]
        if missing:
            return None, f"Model expects feature(s) not found in the dataset: {', '.join(map(str, missing))}."
        return expected, None

    if n_features is not None and n_features != len(feature_columns):
        return None, (
            f"Model expects {n_features} feature(s) but the dataset has "
            f"{len(feature_columns)} (excluding the target)."
        )
    return list(feature_columns), None


def check_records(model_record, dataset_record):
    """Compatibility of two validated records, without loading any files."""
    inputs = model_record.get("required_inputs") or {}
    target = dataset_record["target_variable"]
    features = [c for c in dataset_record["columns"] if c != target]
    ordered, reason = check(inputs.get("feature_names_in"), inputs.get("n_features_in"), features)
    return ordered is not None, reason
