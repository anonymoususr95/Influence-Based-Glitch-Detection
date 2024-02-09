import os.path
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset

from ibed.influence_functions import TracInInfluenceTorch
from .utils import train_test_loaders, run_model, calc_f1, save_as_np, save_as_json

import torch
from .signals_fast import InfluenceErrorSignals
from .error_injection import mislabelled_uniform, mislabelled_cb
from .pipeline.config_manager import ConfigManager
from .pipeline.data_ops import dispatcher as data_dispatcher
from .pipeline.subset_generator import gen_save_subset
from .pipeline.utils import load_model
from .pipeline.calc_influence import compute_inf_mat

def parse_args():
    parser = argparse.ArgumentParser()
    # Dataset arguments
    parser.add_argument("--data_name", required=True, type=str)
    parser.add_argument("--data_folder", required=True, type=str)
    parser.add_argument("--subset_ratio", required=False, type=float, default=None)
    parser.add_argument("--no_subset", action="store_true")
    # Model arguments
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--model_conf", required=True, type=str)
    parser.add_argument("--training_conf", required=False, type=str)
    # Error injection arguments
    parser.add_argument("--error", required=True, type=str)
    parser.add_argument("--contamination", required=True, type=float)
    parser.add_argument("--classes_to_cont", required=False, default=1, type=float)
    # Influence matrix arguments
    parser.add_argument("--inf_fn_conf", required=False, type=str, default=None)
    # Global arguments
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", required=True, type=str, default="cpu")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dfolder_name = (
        f"subset_{args.seed}_{args.subset_ratio}" if not args.no_subset else "full"
    )
    data_savedir = os.path.join("results", args.data_name, dfolder_name)

    # Generate Subset

    print(f"Generating the subset for {args.data_name} with ratio {args.subset_ratio}")

    trainset, testset, trainset_labels, testset_labels = data_dispatcher[
        args.data_name
    ]().load_data(args.data_folder)

    if not args.no_subset:
        trainset = gen_save_subset(
            trainset,
            trainset_labels,
            args.subset_ratio,
            data_savedir,
            "clean_train.pt",
            args.seed,
        )
        testset = gen_save_subset(
            testset,
            testset_labels,
            args.subset_ratio,
            data_savedir,
            "clean_test.pt",
            args.seed,
        )

        trainset_labels = trainset.tensors[1]
        testset_labels = testset.tensors[1]

    # Run clean model on subset

    model_savedir = os.path.join(
        data_savedir, args.model_name
    )

    cm = ConfigManager(model_conf=args.model_conf, training_conf=args.training_conf, inf_func_conf=args.inf_fn_conf)
    num_classes = len(trainset.tensors[1].unique())
    input_shape = trainset.tensors[0].shape[1:]

    clean_model = load_model(
        model_config=cm.model_conf, input_shape=input_shape, num_classes=num_classes
    )

    if not args.no_train_on_clean:
        print(f"Running the model config {cm.model_conf}")

        clean_trainloader, clean_testloader = train_test_loaders(
            train_set=trainset,
            test_set=testset,
            batch_size=cm.training_conf.get_batch_size(),
            seed=args.seed,
        )

        clean_ckpt_savedir = os.path.join(model_savedir, 'clean', 'ckpts')

        Path(clean_ckpt_savedir).mkdir(parents=True, exist_ok=True)

        clean_model, clean_m_info, clean_preds_info = run_model(
            trainloader=clean_trainloader,
            testloader=clean_testloader,
            clean_model=clean_model,
            cm=cm,
            ckpt_savedir=clean_ckpt_savedir,
            device=args.device,
        )

        save_as_json(clean_m_info, os.path.join(model_savedir, 'clean', 'model_info.json'))

    # Add Errors
    dirty_folder_name = 'dirty_train'



    error_savedir = Path(
        model_savedir,
        dirty_folder_name,
        args.error,
        "contamination_" + str(args.contamination),
        '' if not args.cont_all else 'cont_all'
    )

    error_col = None

    Path(error_savedir).mkdir(parents=True, exist_ok=True)

    if args.error == 'mislabelled_uniform':
        dirty_dataset, error_col = mislabelled_uniform(
            dataset=trainset,
            labels=trainset_labels,
            contamination_ratio=args.contamination,
            random_seed=args.seed
        )
    elif args.error == 'mislabelled_cb':
        dirty_dataset, error_col = mislabelled_cb(
            dataset=trainset,
            labels=trainset_labels,
            contamination_ratio=args.contamination,
            random_seed=args.seed,
            contaminate_all=args.cont_all
        )
    else:
        raise AssertionError(f'{args.error} is unknown options are ood_far, ood_near')

    torch.save(dirty_dataset, os.path.join(error_savedir, f'{dirty_folder_name}.pt'))

    error_info = {"contamination": args.contamination, "error": args.error}

    save_as_np(error_col, os.path.join(error_savedir, 'error_col.npy'))
    save_as_json(error_info, os.path.join(error_savedir, 'error_info.json'))

    trainset = dirty_dataset
    trainset_labels = dirty_dataset.tensors[1]

    # Run Dirty Model

    dirty_ckpt_savedir = os.path.join(error_savedir, 'ckpts')

    dirty_model = load_model(
        model_config=cm.model_conf, input_shape=input_shape, num_classes=num_classes
    )

    dirty_trainloader, dirty_testloader = train_test_loaders(
        train_set=trainset,
        test_set=testset,
        batch_size=cm.training_conf.get_batch_size(),
        seed=args.seed,
    )

    Path(dirty_ckpt_savedir).mkdir(parents=True, exist_ok=True)

    dirty_model, dirty_m_info, dirty_preds_info = run_model(
        trainloader=dirty_trainloader,
        testloader=dirty_testloader,
        clean_model=dirty_model,
        cm=cm,
        ckpt_savedir=dirty_ckpt_savedir,
        device=args.device,
    )

    save_as_json(dirty_m_info, os.path.join(error_savedir, 'model_info.json'))

    # Compute influence matrix

    self_inf_arr, train_test_im = None, None

    tracin_cp = TracInInfluenceTorch(
        model_instance=dirty_model,
        save_models_path=dirty_ckpt_savedir,
        random_state=args.seed,
        input_shape=None,
        learning_rate=None,
        batch_size=None,
        epochs=None,
        reg_strength=None,
    )

    fast_cp = False
    inf_batch_size = 4096

    if args.inf_fn_conf:
        inf_batch_size = cm.inf_func_conf.batch_size()
        fast_cp = cm.inf_func_conf.fast_cp()


    self_inf_arr, train_test_im = compute_inf_mat(
        tracin_cp,
        trainset,
        testset,
        dirty_model.trainable_layer_names(),
        e_id=None,
        column_wise=False,
        batch_size=inf_batch_size,
        fast_cp=fast_cp
    )

    save_as_np(self_inf_arr, os.path.join(error_savedir, 'self_inf.npy'))
    save_as_np(train_test_im, os.path.join(error_savedir, 'train_test_im.npy'))

    # Computing

    ies = InfluenceErrorSignals()

    _, nil_opt_labels = ies.nil_opt_fast(train_test_inf_mat=train_test_im,
                     y_train=trainset_labels,
                     y_test=testset_labels)

    im_new = tracin_cp.compute_train_to_test_influence(
        train_set=TensorDataset(trainset.tensors[0], torch.tensor(nil_opt_labels)),
        test_set=testset,
        load_pretrained_model=True,
        batch_size=inf_batch_size,
        layers=dirty_model.trainable_layer_names(),
        epoch_ids_to_consider=[cm.training_conf.get_epochs()],
        fast_cp=True,
    )

    cnci, _ = -ies.nil_fast(train_test_inf_mat=im_new, y_train=trainset_labels, y_test=testset_labels)

    # Compute influence signals

    si = self_inf_arr
    mai = ies.mai_fast(train_test_inf_mat=train_test_im)
    gd_class = ies.gd_class_fast(train_test_inf_mat=train_test_im, y_train=trainset_labels, y_test=testset_labels)
    mi = ies.mi_fast(train_test_inf_mat=train_test_im)

    print('CNCI F1', calc_f1(scores=cnci, error_col=error_col, contamination=args.contamination))
    print('SI F1', calc_f1(scores=si, error_col=error_col, contamination=args.contamination))
    print('MAI F1', calc_f1(scores=mai, error_col=error_col, contamination=args.contamination))
    print('GD-Class F1', calc_f1(scores=gd_class, error_col=error_col, contamination=args.contamination))
    print('MI F1', calc_f1(scores=mi, error_col=error_col, contamination=args.contamination))
