import argparse
import os
from pathlib import Path

import numpy as np
import torch

from .ibed.signals_fast import InfluenceErrorSignals


def int_or_none(value):
    if str(value).lower() in ["none", "null", "na", "nan"]:
        return None
    return int(value)


def str_or_none(value):
    if value.lower() in ["none", "null", "na", "nan"]:
        return None
    return value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_path",
        required=True,
        type=str,
        help="Path to train dataset (TensorDataset)",
    )
    parser.add_argument(
        "--test_data_path",
        required=True,
        type=str,
        help="Path to test dataset (TensorDataset)",
    )
    parser.add_argument("--inf_mat_fp", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--sigs_savedir", required=False, type=str_or_none)
    return parser.parse_args()


def load_inf_mat_compute_sigs(train_ids, test_ids, inf_mat, y_true):
    self_inf_arr = inf_mat[:, 0]  # Take self influence column
    train_to_test_inf_mat = inf_mat[:, 1:].copy()  # Take train-to-test columns

    ies = InfluenceErrorSignals()

    sigs_df = ies.compute_train_signals_fast(
        self_inf_arr=self_inf_arr,
        train_to_test_inf_mat=train_to_test_inf_mat,
        train_samples_ids=train_ids,
        test_samples_ids=test_ids,
        y_true=y_true,
    )

    return sigs_df


if __name__ == "__main__":
    args = parse_args()

    train_data = torch.load(args.train_data_path)
    test_data = torch.load(args.test_data_path)

    train_labels = train_data.tensors[1].numpy()
    test_labels = test_data.tensors[1].numpy()

    savedir = args.sigs_savedir
    if savedir is None:
        savedir = Path(args.inf_mat_fp).parent
    else:
        Path(savedir).mkdir(parents=True, exist_ok=True)

    error_type = args.error

    y_true = np.concatenate((train_labels, test_labels))

    train_ids = np.arange(len(train_labels))
    test_ids = np.arange(len(test_labels)) + len(train_labels)

    with open(os.path.join(args.inf_mat_fp), "rb") as f:
        inf_mat = np.load(f, allow_pickle=True)

    sigs_df = load_inf_mat_compute_sigs(
        train_ids=train_ids, test_ids=test_ids, inf_mat=inf_mat
    )

    fname = "signals.csv"

    sigs_df.to_csv(os.path.join(savedir, fname), index=False)
