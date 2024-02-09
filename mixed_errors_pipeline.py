import os.path
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset

from ibed.influence_functions import TracInInfluenceTorch
from ibed.signals_fast import InfluenceErrorSignals
from .utils import train_test_loaders, run_model, get_last_ckpt, save_as_np, save_as_json
from .pipeline.config_manager import ConfigManager

import torch
from .pipeline.utils import load_model
from .pipeline.calc_influence import compute_inf_mat

def parse_args():
	parser = argparse.ArgumentParser()
	# Dataset arguments
	parser.add_argument("--train_data_fp", required=True, type=str)
	parser.add_argument("--test_data_fp", required=True, type=str)
	parser.add_argument("--num_classes", required=True, type=int)

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

	trainset = torch.load(args.train_data_fp)
	testset = torch.load(args.test_data_fp)

	trainset_labels = trainset.tensors[1]
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


	# calculate CFI, CFID

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

	# CFI

	rnil, _ = ies.nil_fast(train_test_inf_mat=im_new, y_train=trainset_labels, y_test=testset_labels)
	rnil = pd.DataFrame(-rnil)

	rnil.to_csv(os.path.join(error_savedir, 'rnil.csv'), index=False)

	# CFID

	nil_unreduced = compute_il_unreduced(
		train_test_inf=train_test_im,
		y_train=trainset_labels,
		y_test=testset_labels,
		neg_only=True
	)

	pil_unreduced = compute_il_unreduced(
		train_test_inf=train_test_im,
		y_train=trainset_labels,
		y_test=testset_labels,
		neg_only=False
	)

	nil_cf_unreduced = compute_il_unreduced(
		train_test_inf=im_new,
		y_train=trainset_labels,
		y_test=testset_labels,
		neg_only=True
	)

	pil_cf_unreduced = compute_il_unreduced(
		train_test_inf=im_new,
		y_train=trainset_labels,
		y_test=testset_labels,
		neg_only=False
	)

	ncfid = np.linalg.norm(nil_unreduced - nil_cf_unreduced, axis=0)
	pcfid = np.linalg.norm(pil_unreduced - pil_cf_unreduced, axis=0)

	# Counter factual minimum rank influence difference
	ncfid_rank = pd.DataFrame(index=np.argsort(ncfid))
	ncfid_rank['ncfid_r'] = np.arange(len(ncfid))
	pcfid_rank = pd.DataFrame(index=np.argsort(pcfid))
	pcfid_rank['pcfid_r'] = np.arange(len(pcfid))
	cfid_ranks = pd.concat([ncfid_rank, pcfid_rank], axis=1).min(axis=1)

	cfid_ranks_df = pd.DataFrame(cfid_ranks)

	cfid_ranks_df.to_csv(os.path.join(error_savedir, 'cfid.csv'), index=False)
