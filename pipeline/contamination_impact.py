import argparse
import fnmatch
import json
import os

import matplotlib.pyplot as plt
import pandas as pd


def get_config(config_file_path):
    with open(config_file_path, "r") as f:
        conf_file = json.load(f)
    return conf_file


def extract_from_info_files(path):
    test_acc_cont_ratio_dict = {}
    for dirpath, dirs, files in os.walk(path):
        for filename in fnmatch.filter(files, "info.json"):
            with open(os.path.join(dirpath, filename), "r") as f:
                info_file = json.load(f)
                conf_file = get_config(info_file["config_path"])
                cont_ratio = info_file["contamination"]
                dname = conf_file["dataset"]["name"]
                error = (
                    conf_file["error"]["error"]
                    + "_"
                    + conf_file["error"]["error_subtype"]
                )
                if error not in test_acc_cont_ratio_dict:
                    test_acc_cont_ratio_dict[error] = {}
                if dname not in test_acc_cont_ratio_dict[error]:
                    test_acc_cont_ratio_dict[error][dname] = {}
                test_acc_cont_ratio_dict[error][dname][0] = info_file["clean_test_acc"]
                test_acc_cont_ratio_dict[error][dname][cont_ratio * 100] = info_file[
                    "dirty_test_acc"
                ]
    return test_acc_cont_ratio_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_path", required=True, type=str, help="Path to results folder"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    path = args.results_path

    info_df = extract_from_info_files(path)
    for e, item in info_df.items():
        a = pd.DataFrame.from_dict(item).transpose()
        a = a[sorted(a.columns)]
        a.plot.bar(rot=60)
        plt.xlabel("Error ratio")
        plt.ylabel("Test Accuracy")
        plt.title(f"Resnset perf. on {e}")
        plt.tight_layout()
        plt.show()
        plt.close()
    print()
