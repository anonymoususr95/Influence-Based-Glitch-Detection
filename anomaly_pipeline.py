import os.path
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

from ibed.influence_functions import TracInInfluenceTorch
from .utils import train_test_loaders, run_model, get_last_ckpt, save_as_np, save_as_json, calc_f1

from torch.utils.data import TensorDataset

import torch
from .signals_fast import InfluenceErrorSignals
from .error_injection import inj_anomalies
from .pipeline.config_manager import ConfigManager
from .pipeline.data_ops import dispatcher as data_dispatcher
from .pipeline.subset_generator import gen_save_subset
from .pipeline.utils import load_model
from .pipeline.calc_influence import compute_inf_mat


def compute_il_unreduced(train_test_inf, y_train, y_test):
	train_test_inf_mat_tmp = train_test_inf.copy()
	il_values = None
	for l in np.unique(y_train):
		l_test_ids = np.where(y_test == l)[0]
		inf_of_l_samples_on_test_l_samples = train_test_inf_mat_tmp[:, l_test_ids]
		inf_of_l_samples_on_test_l_samples[
			inf_of_l_samples_on_test_l_samples < 0
			] = 0
		il_values_tmp = inf_of_l_samples_on_test_l_samples.sum(axis=1)
		if il_values is None:
			il_values = il_values_tmp.copy()
		else:
			il_values = np.vstack((il_values, il_values_tmp))
	return il_values, np.argmax(il_values, axis=0)

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
	parser.add_argument("--contamination", required=True, type=float)
	parser.add_argument("--ood_data_name", required=False, type=str, default=None)
	# Influence matrix arguments
	parser.add_argument("--inf_fn_conf", required=False, type=str, default=None)
	# Global arguments
	parser.add_argument("--seed", required=True, type=int)
	parser.add_argument("--device", required=True, type=str, default="cpu")

	return parser.parse_args()


def save_update_datasets(trainset, testset, dirty_dataset, fp, train_error):
	torch.save(dirty_dataset, fp)
	if train_error:
		trainset = dirty_dataset
	else:
		testset = dirty_dataset
	return trainset, testset


def compute_second_best_pil(train_test_inf, y_train, y_test):
	train_test_inf_mat_tmp = train_test_inf.copy()
	pil_values = None
	for l in np.unique(y_train):
		l_test_ids = np.where(y_test == l)[0]
		inf_of_l_samples_on_test_l_samples = train_test_inf_mat_tmp[:, l_test_ids]
		inf_of_l_samples_on_test_l_samples[
			inf_of_l_samples_on_test_l_samples < 0
			] = 0
		pil_values_tmp = inf_of_l_samples_on_test_l_samples.sum(axis=1)
		if pil_values is None:
			pil_values = pil_values_tmp.copy()
		else:
			pil_values = np.vstack((pil_values, pil_values_tmp))
	argpt = np.argpartition(-pil_values, kth=1, axis=0)
	return argpt[1, :]



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

	print(f"Running the model config {cm.model_conf.data} on clean.")

	clean_trainloader, clean_testloader = train_test_loaders(
		train_set=trainset,
		test_set=testset,
		batch_size=cm.training_conf.get_batch_size(),
		seed=args.seed,
	)

	clean_ckpt_savedir = os.path.join(model_savedir, 'clean', 'ckpts')

	Path(clean_ckpt_savedir).mkdir(parents=True, exist_ok=True)

	clean_model, clean_m_info, clean_m_preds_info = run_model(
		trainloader=clean_trainloader,
		testloader=clean_testloader,
		clean_model=clean_model,
		cm=cm,
		ckpt_savedir=clean_ckpt_savedir,
		device=args.device,
	)

	save_as_json(clean_m_info, os.path.join(model_savedir, 'clean', 'model_info.json'))

	# Add Errors

	error_savedir = os.path.join(
		model_savedir,
		'dirty_train',
		'anomalies',
		"contamination_" + str(args.contamination),
	)

	error_col = None

	dataset_to_infect = trainset
	dataset_to_infect_labels = trainset_labels

	Path(error_savedir).mkdir(parents=True, exist_ok=True)

	if args.ood_data_name is None:
		raise AssertionError('ood_data_name argument can not be none')
	_, ood_testset, _, ood_testset_labels = data_dispatcher[
		args.ood_data_name
	]().load_data(args.data_folder)
	dirty_dataset, error_col = inj_anomalies(
		dataset=dataset_to_infect,
		labels=dataset_to_infect_labels,
		ood_dataset=ood_testset,
		contamination_ratio=args.contamination,
		random_seed=args.seed,
	)
	trainset, testset = save_update_datasets(
		trainset=trainset,
		testset=testset,
		dirty_dataset=dirty_dataset,
		fp=os.path.join(error_savedir, 'dirty_train.pt'),
		train_error=args.inject_error_train
	)

	trainset_labels = trainset.tensors[1]
	testset_labels = testset.tensors[1]

	error_info = {"contamination": args.contamination, "error": args.error}

	save_as_np(error_col, os.path.join(error_savedir, 'error_col.npy'))
	save_as_json(error_info, os.path.join(error_savedir, 'error_info.json'))

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

	dirty_model, dirty_m_info, dirty_m_preds_info = run_model(
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

	if not args.no_inf_computation:
		if not args.no_train_on_dirty:
			dirty_model_last_ckpt = get_last_ckpt(dirty_ckpt_savedir)
			state_dict = torch.load(dirty_model_last_ckpt)
			dirty_model.load_state_dict(state_dict["model_state_dict"])

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

	# Compute influence signals

	y_train = trainset_labels.numpy()
	y_test = testset_labels.numpy()
	train_sigs_df, test_sigs_df = None, None

	# PCID

	ies = InfluenceErrorSignals()


	positive_counterfactuals = compute_second_best_pil(train_test_inf=train_test_im,
													   y_train=y_train,
													   y_test=y_test)



	im_new = tracin_cp.compute_train_to_test_influence(
		train_set=TensorDataset(trainset.tensors[0], torch.tensor(positive_counterfactuals)),
		test_set=testset,
		load_pretrained_model=True,
		batch_size=inf_batch_size,
		layers=dirty_model.trainable_layer_names(),
		epoch_ids_to_consider=[cm.training_conf.get_epochs()],
		fast_cp=True,
	)

	pil_unreduced = compute_il_unreduced(train_test_inf=train_test_im, y_train=y_train, y_test=y_test)

	pil_unreduced_cf = compute_il_unreduced(train_test_inf=im_new, y_train=positive_counterfactuals, y_test=y_test)

	norm = np.linalg.norm(pil_unreduced - pil_unreduced_cf, ord=np.inf, axis=0)

	w = ies.mpi_fast(train_test_inf_mat=train_test_im)

	pcid = 1/(w * norm)

	# other signals

	si = self_inf_arr
	mai = ies.mai_fast(train_test_inf_mat=train_test_im)
	gd_class = ies.gd_class_fast(train_test_inf_mat=train_test_im, y_train=y_train, y_test=y_test)
	mi = ies.mi_fast(train_test_inf_mat=train_test_im)

	print('PCID F1', calc_f1(scores=pcid, error_col=error_col, contamination=args.contamination))
	print('SI F1', calc_f1(scores=si, error_col=error_col, contamination=args.contamination))
	print('MAI F1', calc_f1(scores=mai, error_col=error_col, contamination=args.contamination))
	print('GD-Class F1', calc_f1(scores=gd_class, error_col=error_col, contamination=args.contamination))
	print('MI F1', calc_f1(scores=mi, error_col=error_col, contamination=args.contamination))
