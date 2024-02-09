import argparse
import copy
import inspect
import json
import os
import os.path
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from config_manager import ConfigManager
from data_ops import dispatcher as data_dispatcher
from data_ops import subset_selection
from errors import dispatcher as error_dispatcher
from file_path_manager import FilePathMger
from model_train import train
from torch.utils.data import DataLoader, TensorDataset

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import pandas as pd
from sklearn.metrics import average_precision_score
from ibed.utils import load_model

from ibed.signals_fast import InfluenceErrorSignals
from ibed.influence_functions import TracInInfluenceTorch


def none_or_float(value):
    if str(value).lower() in ["none", "null", "na", "nan"]:
        return None
    return float(value)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, type=str, help="Configuration path (.json)"
    )
    parser.add_argument(
        "--device",
        required=False,
        type=str,
        default="cpu",
        help="Device (e.g. cuda or cpu)",
    )
    parser.add_argument(
        "--results_folder_prefix",
        required=False,
        default="",
        type=str,
        help="Prefix for results folder",
    )
    parser.add_argument(
        "--results_folder_suffix",
        required=False,
        default="",
        type=str,
        help="Suffix for results folder",
    )
    parser.add_argument(
        "--contamination",
        required=False,
        default=None,
        type=none_or_float,
        help="Contamination ratio of the dataset, default 0.01",
    )
    args = parser.parse_args()
    return args


def save_results_as_np(data, results_dir, fname):
    fpath = os.path.join(results_dir, fname + ".npy")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    with open(fpath, "wb") as f:
        np.save(f, data)


def save_results_as_json(data, results_dir, fname):
    fpath = os.path.join(results_dir, fname + ".json")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    with open(fpath, "w") as f:
        json.dump(data, f)


def save_dataset(dataset, labels, save_dir, fname):
    x_tensor = torch.stack([x for x, _ in dataset])
    y_tensor = torch.tensor(labels)
    fpath = os.path.join(save_dir, fname)
    torch.save(TensorDataset(x_tensor, y_tensor), fpath)


def calc_inf_mat(trainset, testset, model, trainable_layers_names, save_dir):
    tracin_cp = TracInInfluenceTorch(
        model_instance=model,
        save_models_path=save_dir,
        random_state=42,
        input_shape=None,
        learning_rate=None,
        batch_size=None,
        epochs=None,
        reg_strength=None,
    )

    self_inf_arr = tracin_cp.compute_self_influence(
        train_set=trainset,
        load_pretrained_model=True,
        batch_size=4096,
        layers=trainable_layers_names,
    )

    train_to_test_im = tracin_cp.compute_train_to_test_influence(
        train_set=trainset,
        test_set=testset,
        load_pretrained_model=True,
        batch_size=4096,
        layers=trainable_layers_names,
    )

    inf_mat = np.column_stack((self_inf_arr, train_to_test_im))

    return inf_mat


def model_train(cm, ckpt_savedir, trainset, testset, model, device, save_ckpts=True):
    if cm.get_training_env_conf().get_random_seed() is not None:
        torch.manual_seed(cm.get_training_env_conf().get_random_seed())
    trainloader = DataLoader(
        trainset,
        batch_size=cm.get_training_env_conf().get_batch_size(),
        shuffle=True,
        num_workers=15,
    )
    testloader = DataLoader(
        testset,
        batch_size=cm.get_training_env_conf().get_batch_size(),
        shuffle=False,
        num_workers=15,
    )
    if cm.get_flags().train_model():
        print(f"\n\nTRAINING MODEL {cm.get_model_conf().get_name()}")
        return train(
            model=model,
            epochs=cm.get_training_env_conf().get_epochs(),
            learning_rate=cm.get_training_env_conf().get_learning_rate(),
            reg_strength=cm.get_training_env_conf().model_regularization_strength(),
            save_dir=ckpt_savedir,
            train_loader=trainloader,
            test_loader=testloader,
            device=device,
            save_ckpts=save_ckpts,
        )


def bar_plot_from_pandas(
    df, title, x_label, y_label, y_lim_dict, save_dir, legend=True
):
    df.plot.bar(rot=60)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.ylim(**y_lim_dict)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    if not legend:
        plt.legend().remove()
    plt.savefig(os.path.join(save_dir, title + ".png"))
    # plt.show()
    plt.close()


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


if __name__ == "__main__":
    args = parse_args()
    config_file = args.config

    # READ CONFIG OR CONSOLE
    cm = ConfigManager(args.config)

    # DETERMINE FILE PATHS
    results_folder_prefix = args.results_folder_prefix
    results_folder_suffix = args.results_folder_suffix
    fpm = FilePathMger(
        cm.get_dataset_conf().get_name(),
        cm.get_error_conf().get_error(),
        cm.get_error_conf().get_error_subtype(),
        cm.get_model_conf().get_name(),
        results_folder_prefix,
        results_folder_suffix,
    )

    # LOAD MODEL
    clean_model = load_model(
        model_config=cm.get_model_conf(),
        input_shape=cm.get_dataset_conf().get_input_shape(),
        num_classes=cm.get_dataset_conf().get_total_classes(),
    )

    dirty_model = copy.deepcopy(clean_model)

    # LOAD - PREPARE DATASET
    trainset, testset = data_dispatcher[cm.get_dataset_conf().get_name()]().load_data(
        cm.get_dataset_conf().get_path()
    )

    random_seed = 42
    dataset_conf = cm.get_dataset_conf()
    train_labels = trainset.targets
    test_labels = testset.targets

    if type(train_labels) == torch.Tensor:
        all_train_labels = train_labels.numpy()
    else:
        all_train_labels = np.array(train_labels)

    if type(test_labels) == torch.Tensor:
        all_test_labels = test_labels.numpy()
    else:
        all_test_labels = np.array(test_labels)

    if args.contamination is not None:
        contamination = args.contamination
    else:
        contamination = cm.get_error_conf().contamination_ratio()
    fpm.set_suffix_name(f"contamination_{contamination}")

    if dataset_conf.get_train_subset_ratio() or dataset_conf.get_test_subset_ratio():
        res_dir_suffix = f"subset_{random_seed}"
    else:
        res_dir_suffix = "full"
    fpm.append_suffix_name(res_dir_suffix)

    if dataset_conf.get_train_subset_ratio() is not None:
        trainset, train_labels, train_ids = subset_selection(
            trainset,
            labels=all_train_labels,
            ratio=cm.get_dataset_conf().get_train_subset_ratio(),
            random_seed=random_seed,
        )
    else:
        train_ids = np.arange(len(trainset))

    save_results_as_np(train_ids, fpm.results_savedir(), FilePathMger.train_ids_fname())

    if dataset_conf.get_test_subset_ratio() is not None:
        testset, test_labels, test_ids = subset_selection(
            testset,
            labels=all_test_labels,
            ratio=cm.get_dataset_conf().get_test_subset_ratio(),
            random_seed=random_seed,
        )
    else:
        test_ids = np.arange(len(testset))

    save_results_as_np(test_ids, fpm.results_savedir(), FilePathMger.test_ids_fname())

    # INJECT ERRORS
    dirty_trainset = None
    if cm.get_flags().inject_error():
        print(
            f"\n\nINJECTING ERROR: "
            f"{cm.get_error_conf().get_error()} "
            f"{cm.get_error_conf().get_error_subtype()}"
        )
        full_error_name = (
            cm.get_error_conf().get_error()
            + "_"
            + cm.get_error_conf().get_error_subtype()
        )
        error_inj = error_dispatcher[full_error_name]
        dirty_trainset, error_col, dirty_train_labels = error_inj().inject(
            trainset,
            train_labels,
            random_seed=42,
            contamination_ratio=contamination,
            **cm.get_error_conf().get_kwargs(),
        )
        save_results_as_np(
            data=error_col,
            results_dir=fpm.results_savedir(),
            fname=FilePathMger.error_col(),
        )

    if res_dir_suffix != "full":
        save_dataset(
            dirty_trainset,
            dirty_train_labels,
            fpm.results_savedir(),
            "dirty_trainset.pt",
        )
        save_dataset(testset, test_labels, fpm.results_savedir(), "testset.pt")

    # MODEL TRAIN
    info = {}
    info["config_path"] = args.config
    info["contamination"] = contamination

    if cm.get_flags().run_clean_model():
        clean_model, train_loss, test_loss, train_acc, test_acc = model_train(
            cm,
            fpm.ckpt_savedir(clean=True),
            trainset,
            testset,
            clean_model,
            args.device,
            save_ckpts=True,
        )
        info["clean_train_acc"] = train_acc
        info["clean_test_acc"] = test_acc
        info["clean_train_loss"] = train_loss
        info["clean_test_loss"] = test_loss

    if cm.get_flags().train_model():
        (
            dirty_model,
            dirty_train_loss,
            dirty_test_loss,
            dirty_train_acc,
            dirty_test_acc,
        ) = model_train(
            cm,
            fpm.ckpt_savedir(clean=False),
            dirty_trainset,
            testset,
            dirty_model,
            args.device,
            save_ckpts=True,
        )
        info["dirty_train_acc"] = dirty_train_acc
        info["dirty_test_acc"] = dirty_test_acc
        info["dirty_train_loss"] = dirty_train_loss
        info["dirty_test_loss"] = dirty_test_loss
        save_results_as_json(info, fpm.results_savedir(), "info")

    # CALCULATE INFLUENCE MATRIX
    if cm.get_flags().calc_influence():
        print("\n\nCALCULATING INFLUENCE MATRIX")
        inf_mat = calc_inf_mat(
            dirty_trainset,
            testset,
            dirty_model,
            dirty_model.trainable_layer_names(),
            fpm.ckpt_savedir(clean=False),
        )
        save_results_as_np(inf_mat, fpm.results_savedir(), FilePathMger.inf_mat_fname())

    # CALCULATE SIGNALS

    if cm.get_flags().calc_signals():
        print("\n\nCALCULATING INFLUENCE SIGNALS")
        y_true = np.concatenate((dirty_train_labels, test_labels))

        train_ids = np.arange(len(dirty_trainset))
        test_ids = np.arange(len(test_labels)) + len(train_ids)

        ies = InfluenceErrorSignals()

        signals = ies.train_signals_names()

        self_inf_arr = inf_mat[:, 0]
        train_to_test_inf_mat = inf_mat[:, 1:]

        sigs_df = ies.compute_train_signals_fast(
            self_inf_arr=self_inf_arr,
            train_to_test_inf_mat=train_to_test_inf_mat,
            train_samples_ids=train_ids,
            test_samples_ids=test_ids,
            y_true=y_true,
        )

        sigs_df.to_csv(
            os.path.join(fpm.results_savedir(), FilePathMger.signals_fname() + ".csv"),
            index=False,
        )

    # CALCULATE AVERAGE PRECISION
    name = "signals_avep"
    plot_ap_from_sig_file(
        signals_df=sigs_df, name=name, save_dir=fpm.figures_dir(), true_error=error_col
    )
