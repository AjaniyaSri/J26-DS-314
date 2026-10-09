from __future__ import annotations

import pandas as pd


def split_by_forecast_origin(frames, config):
    split_config = config["splits"]
    date_column = "date"
    date_ranges = {
        "train": (split_config["train_start"], split_config["train_end"]),
        "validation": (split_config["validation_start"], split_config["validation_end"]),
        "test": (split_config["test_start"], split_config["test_end"]),
    }
    output = {variant: {} for variant in frames}
    reference_counts = None
    for variant, frame in frames.items():
        for split_name, (start, end) in date_ranges.items():
            mask = frame[date_column].between(pd.Timestamp(start), pd.Timestamp(end), inclusive="both")
            subset = frame.loc[mask].copy().reset_index(drop=True)
            if not subset.empty:
                latest_target = subset["target_date"].max()
                next_cutoff = pd.Timestamp(end) + pd.Timedelta(days=1)
                if latest_target > next_cutoff:
                    raise AssertionError(f"{split_name} targets extend beyond the next 9 AM cutoff")
            output[variant][split_name] = subset
        counts = tuple(len(output[variant][name]) for name in date_ranges)
        if reference_counts is None:
            reference_counts = counts
        elif counts != reference_counts:
            raise AssertionError("Experiments have different split sample counts")

    # Labels in a block end on the next block's first forecast date, when that 9 AM reading is known.
    split_targets = {
        name: (items["train"]["target_date"], items["validation"]["target_date"], items["test"]["target_date"])
        for name, items in output.items()
    }
    for variant, (train_targets, validation_targets, test_targets) in split_targets.items():
        if train_targets.empty or validation_targets.empty or test_targets.empty:
            raise ValueError(f"{variant} has an empty train, validation, or test split")
        if train_targets.max() >= validation_targets.min() or validation_targets.max() >= test_targets.min():
            raise AssertionError(f"{variant} target dates overlap between splits")
    return output