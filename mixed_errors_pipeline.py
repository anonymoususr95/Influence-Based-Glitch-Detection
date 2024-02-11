import itertools
import os.path
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, Subset, ConcatDataset

from ibed.experiments.anomaly_signal import compute_second_best_pil, calc_f1
from ibed.influence_functions import TracInInfluenceTorch
from ibed.signals_fast import InfluenceErrorSignals
from .utils import train_test_loaders, run_model, get_last_ckpt, save_as_np, save_as_json
from .pipeline.config_manager import ConfigManager
from .pipeline.subset_generator import gen_save_subset
from .pipeline.data_ops import dispatcher as data_dispatcher
from .error_injection import mislabelled_uniform, inj_anomalies

ood_datasets_dict = {
    'mnist': 'fmnist',
    'fmnist': 'mnist',
    'cifar10': 'fmnist',
}

import torch
from .pipeline.utils import load_model
from .pipeline.calc_influence import compute_inf_mat

def parse_args():
	parser = argparse.ArgumentParser()
	# Dataset arguments
	parser.add_argument("--subset_ratio", required=True, type=str)
	parser.add_argument("--dataset_name", required=True, type=str)

	# Model arguments
	parser.add_argument("--model_name", required=True, type=str)
	parser.add_argument("--model_conf", required=True, type=str)
	parser.add_argument("--training_conf", required=False, type=str)

	# Influence matrix arguments
	parser.add_argument("--inf_fn_conf", required=False, type=str, default=None)

	# Global arguments
	parser.add_argument("--device", required=False, type=str, default="cpu")
	parser.add_argument("--savedir", required=False, type=str, default=None)
	parser.add_argument("--seed", required=True, type=int, default=42)
	return parser.parse_args()

def compute_il_unreduced(train_test_inf, y_train, y_test, neg_only):
	train_test_inf_mat_tmp = train_test_inf.copy()
	il_values = None
	if neg_only:
		train_test_inf_mat_tmp[
			train_test_inf_mat_tmp > 0
			] = 0
	else:
		train_test_inf_mat_tmp[
			train_test_inf_mat_tmp < 0
			] = 0
	for l in np.unique(y_train):
		l_test_ids = np.where(y_test == l)[0]
		inf_of_l_samples_on_test_l_samples = train_test_inf_mat_tmp[:, l_test_ids]
		il_values_tmp = inf_of_l_samples_on_test_l_samples.sum(axis=1)
		if il_values is None:
			il_values = il_values_tmp.copy()
		else:
			il_values = np.vstack((il_values, il_values_tmp))
	return il_values


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

	error_names = []
	dirty_test_sets = []

	trainset, error_col = mislabelled_uniform(
		dataset=trainset,
		labels=trainset.tensors[1].numpy(),
		contamination_ratio=0.1,
		random_seed=args.seed
	)

	dirty_ids = np.where(error_col == 1)[0]
	clean_ids = np.where(error_col == 0)[0]
	clean_trainset, errors_trainset = Subset(trainset, clean_ids), Subset(trainset, dirty_ids)
	dirty_test_sets.append(errors_trainset)
	error_names.append(['mu'] * len(dirty_ids))

	# inject ood

	_, ood_testset, _, _ = data_dispatcher[
		ood_datasets_dict[args.dataset_name]
	]().load_data(args.data_folder)
	trainset, error_col = inj_anomalies(
		dataset=TensorDataset(trainset, trainset_labels),
		labels=trainset_labels.numpy(),
		ood_dataset=ood_testset,
		contamination_ratio=0.1,
		random_seed=args.seed,
		unclustered=False
	)
	dirty_ids = np.where(error_col == 1)[0]
	clean_ids = np.where(error_col == 0)[0]
	clean_trainset, errors_trainset = Subset(trainset, clean_ids), Subset(trainset, dirty_ids)
	dirty_test_sets.append(errors_trainset)
	error_names.append(['ua'] * len(dirty_ids))

	final_dataset = ConcatDataset([clean_trainset, *dirty_test_sets])
	final_dataset_samples = torch.stack([sample[0] for sample in final_dataset])
	final_dataset_labels = torch.tensor([sample[1] for sample in final_dataset])
	final_dataset = TensorDataset(final_dataset_samples, final_dataset_labels)

	clean_names = ['clean'] * len(clean_trainset)
	sample_names = np.array([*clean_names, *list(itertools.chain.from_iterable(error_names))])

	error_col = np.array([0] * len(sample_names))
	dirty_s = np.where(sample_names != 'clean')[0]
	error_col[dirty_s] = 1

	trainset_labels = final_dataset.tensors[1]
	testset_labels = testset.tensors[1]

	# Run clean model on subset

	if args.savedir is not None:
		error_savedir = args.savedir
	else:
		error_savedir = Path(args.train_data_fp).parent

	# Run Dirty Model

	dirty_ckpt_savedir = os.path.join(error_savedir, 'ckpts')

	input_shape = trainset.tensors[0].shape[1:]
	num_classes = args.num_classes

	cm = ConfigManager(model_conf=args.model_conf, training_conf=args.training_conf, inf_func_conf=args.inf_fn_conf)

	dirty_model = load_model(
		model_config=cm.model_conf, input_shape=input_shape, num_classes=num_classes
	)

	print(f"Training for dirty data the model {cm.model_conf}")

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


	# calculate CNCI

	ies = InfluenceErrorSignals()
	_, nil_opt_labels = ies.nil_opt_fast(train_test_inf_mat=train_test_im,
										 y_train=trainset_labels, y_test=testset_labels)

	print('Calculating counterfactual influence')

	im_new = tracin_cp.compute_train_to_test_influence(
		train_set=TensorDataset(trainset.tensors[0], torch.tensor(nil_opt_labels)),
		test_set=testset,
		load_pretrained_model=True,
		batch_size=inf_batch_size,
		layers=dirty_model.trainable_layer_names(),
		epoch_ids_to_consider=[cm.training_conf.get_epochs()],
		fast_cp=fast_cp,
	)

	# CNCI

	cnci, _ = -ies.nil_fast(train_test_inf_mat=im_new, y_train=trainset_labels, y_test=testset_labels)

	print('CNCI F1', calc_f1(scores=cnci, error_col=error_col, contamination=args.contamination))

	# PCID

	ies = InfluenceErrorSignals()

	positive_counterfactuals = compute_second_best_pil(train_test_inf=train_test_im,
													   y_train=trainset_labels,
													   y_test=testset_labels)

	im_new = tracin_cp.compute_train_to_test_influence(
		train_set=TensorDataset(trainset.tensors[0], torch.tensor(positive_counterfactuals)),
		test_set=testset,
		load_pretrained_model=True,
		batch_size=inf_batch_size,
		layers=dirty_model.trainable_layer_names(),
		epoch_ids_to_consider=[cm.training_conf.get_epochs()],
		fast_cp=True,
	)

	pil_unreduced = compute_il_unreduced(train_test_inf=train_test_im, y_train=trainset_labels, y_test=testset_labels)

	pil_unreduced_cf = compute_il_unreduced(train_test_inf=im_new, y_train=positive_counterfactuals, y_test=testset_labels)

	norm = np.linalg.norm(pil_unreduced - pil_unreduced_cf, ord=np.inf, axis=0)

	w = ies.mpi_fast(train_test_inf_mat=train_test_im)

	pcid = 1 / (w * norm)

	print('PCID F1', calc_f1(scores=pcid, error_col=error_col, contamination=args.contamination))
