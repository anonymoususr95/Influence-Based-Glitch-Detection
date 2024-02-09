import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def bar_plot_from_pandas(
    df, x_label, y_label, y_lim_dict, save_dir, title=None, legend=True
):
    df.plot.bar(rot=60)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.ylim(**y_lim_dict)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    if not legend:
        plt.legend().remove()
    plt.savefig(os.path.join(save_dir, title + ".png"))
    plt.close()


def plot_AP_on_epochs(sigs_ap, title, save_dir):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    df_transposed = sigs_ap.T
    df_transposed.plot(kind="line", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Average Precision")
    plt.ylim((0, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, title + ".png"))


def ap_from_sig_file(signals_df, true_error):
    aveps = {}
    for c in sorted(signals_df.columns):
        aveps[c] = average_precision_score(true_error, signals_df[c])
    aveps_df = pd.DataFrame.from_dict(aveps, orient="index")
    return aveps_df


def plot_ap_from_sig_file(signals_df, name, save_dir, true_error):
    aveps_df = ap_from_sig_file(signals_df, true_error)
    bar_plot_from_pandas(
        aveps_df,
        title=name,
        x_label="Influence Error Signals",
        y_label="Average Precision",
        y_lim_dict={"bottom": 0, "top": 1},
        save_dir=save_dir,
        legend=False,
    )


def avep_per_signal_df(sigs_savedir, true_error):
    final_aveps_df = pd.DataFrame()
    for f in glob.glob(os.path.join(sigs_savedir, "*.csv")):
        ftext = os.path.splitext(f)[0]
        epoch = int(ftext.split("/")[-1].split("_")[-1][1])
        sigs_df = pd.read_csv(f)
        epoch_sigs_avep = ap_from_sig_file(sigs_df, true_error)
        if len(final_aveps_df.index) == 0:
            final_aveps_df.index = epoch_sigs_avep.index
        assert sum(final_aveps_df.index != epoch_sigs_avep.index) == 0
        final_aveps_df[epoch] = epoch_sigs_avep.values
        print()
    return final_aveps_df.reindex(sorted(final_aveps_df.columns), axis=1)


def str_or_none(value):
    if value.lower() in ["none", "null", "na", "nan"]:
        return None
    return value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sigs_fp", required=True, type=str, help="Path to signals (.csv format)"
    )
    parser.add_argument(
        "--error_col_fp",
        required=True,
        type=str,
        help="Path to error column file (.npy format)",
    )
    parser.add_argument(
        "--figs_savedir", required=False, type=str, help="Path to Figures"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    with open(args.error_col_fp, "rb") as f:
        error_col = np.load(f)

    sigs = pd.read_csv(args.sigs_fp)

    savedir = args.figs_savedir
    if savedir is None:
        savedir = Path(args.error_col_fp).parent
        savedir = os.path.join(savedir, "figures")
    else:
        Path(savedir).mkdir(parents=True, exist_ok=True)

    plot_ap_from_sig_file(
        signals_df=sigs, name=None, save_dir=savedir, true_error=error_col
    )
