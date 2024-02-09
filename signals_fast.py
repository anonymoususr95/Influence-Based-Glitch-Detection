import numpy as np
import pandas as pd
import sys


class InfluenceErrorSignals:
    def __init__(self):
        self.__self_influence = {
            "SI": self.si_fast,
        }
        self.__marginal_signals = {
            "MPI": self.mpi_fast,
            "MNI": self.mni_fast,
            "MAI": self.mai_fast,
            "MNIC": self.mnic_fast,
            "MI": self.mi_fast,
            "AAI": self.aai_fast
        }
        self.__conditional_signals = {
            "PIL": self.pil_fast,
            "NIL": self.nil_fast,
            "NIL_opt": self.nil_opt_fast,
            "PIL_opt": self.pil_opt_fast,
            "AIL": self.ail_fast,
            "#SLIP": self.num_slip_fast,
            "#SLIN": self.num_slin_fast,
            '#SLIN_opt': self.num_slin_opt_fast,
            # 'NIL_opt_avg': self.nil_opt_avg_fast,
            "GD-class": self.gd_class
        }

        self.__test_signals = {
            "NTOPC": self.ntopc
        }

        self.__var_signals = {}

        # self.__unwrapped_conditional_signals = {
        #     'NIL_all': self.nil_all_fast,
        #     '#SLIN_all': self.num_slin_all
        # }

    def train_signals_names(self):
        return {
            *self.__self_influence.keys(),
            *self.__marginal_signals.keys(),
            *self.__conditional_signals.keys(),
        }

    def test_signals_names(self):
        return {
            *self.__test_signals.keys(),
        }

    def compute_train_signals_fast(
        self,
        self_inf_arr,
        train_to_test_inf_mat,
        y_train,
        y_test,
        silent=False
    ):
        signal_vals = {}
        for sig_name in self.train_signals_names():
            if not silent:
                print(sig_name)
            if sig_name in self.__self_influence:
                signal_vals[sig_name] = self_inf_arr
            elif sig_name in self.__marginal_signals:
                ms, _ = self.__marginal_signals[sig_name](train_to_test_inf_mat)
                signal_vals[sig_name] = ms
            elif sig_name in self.__conditional_signals:
                cs, _ = self.__conditional_signals[sig_name](
                    train_to_test_inf_mat, y_train, y_test
                )
                signal_vals[sig_name] = cs
        return pd.DataFrame.from_dict(signal_vals)

    def compute_test_signals_fast(
            self,
            train_to_test_inf_mat,
            train_labels,
            y_preds
    ):
        signal_vals = {}
        for sig_name in self.test_signals_names():
            print(sig_name)
            signal_vals[sig_name] = self.__test_signals[sig_name](
                inf_mat=train_to_test_inf_mat,
                train_labels=train_labels,
                y_preds=y_preds
            )
        return pd.DataFrame.from_dict(signal_vals)

    def compute_signals_var_signals(
        self,
        self_inf_arr,
        train_to_test_inf_mat,
        train_samples_ids,
        test_samples_ids,
        y_true,
    ):
        y_train = y_true[train_samples_ids]
        y_test = y_true[test_samples_ids]
        signal_vals = {}
        for sig_name in self.train_signals_names():
            print(sig_name)
            if sig_name == "NIL_opt" or sig_name == "AIL":
                continue
            elif sig_name in self.__marginal_signals:
                _, msv = self.__marginal_signals[sig_name](train_to_test_inf_mat)
                signal_vals[sig_name] = msv
            elif sig_name in self.__conditional_signals:
                _, csv = self.__conditional_signals[sig_name](
                    train_to_test_inf_mat, y_train, y_test
                )
                signal_vals[sig_name] = csv
        return pd.DataFrame.from_dict(signal_vals)

    def si_fast(self, self_inf_mat):
        return np.diagonal(self_inf_mat)

    def pil_fast(
        self, train_test_inf_mat: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
    ):
        train_test_inf_mat_tmp = train_test_inf_mat.copy()
        pil_values = np.zeros(len(y_train))
        pil_values_var = np.zeros(len(y_train))
        for l in np.unique(y_train):
            l_train_ids = np.where(y_train == l)[0]
            l_test_ids = np.where(y_test == l)[0]
            inf_of_l_samples = train_test_inf_mat_tmp[l_train_ids, :]
            inf_of_l_samples_on_test_l_samples = inf_of_l_samples[:, l_test_ids]
            inf_of_l_samples_on_test_l_samples[
                inf_of_l_samples_on_test_l_samples < 0
            ] = 0
            pil_values[l_train_ids] = inf_of_l_samples_on_test_l_samples.sum(axis=1)
            pil_values_var[l_train_ids] = inf_of_l_samples_on_test_l_samples.var(axis=1)
        return pil_values, pil_values_var

    def nil_fast(
        self, train_test_inf_mat: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
    ):
        train_test_inf_mat_tmp = train_test_inf_mat.copy()
        nil_values = np.zeros(len(y_train))
        nil_values_var = np.zeros(len(y_train))
        for l in np.unique(y_train):
            l_train_ids = np.where(y_train == l)[0]
            l_test_ids = np.where(y_test == l)[0]
            inf_of_l_samples = train_test_inf_mat_tmp[l_train_ids, :]
            inf_of_l_samples_on_test_l_samples = inf_of_l_samples[:, l_test_ids]
            inf_of_l_samples_on_test_l_samples[
                inf_of_l_samples_on_test_l_samples > 0
            ] = 0
            nil_values[l_train_ids] = inf_of_l_samples_on_test_l_samples.sum(axis=1)
            nil_values_var[l_train_ids] = inf_of_l_samples_on_test_l_samples.var(axis=1)
        return -1 * nil_values, nil_values_var

    def nil_opt_avg_fast(self, train_test_inf_mat: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
        nil_opt, _ = self.nil_opt_fast(train_test_inf_mat, y_train=y_train, y_test=y_test)
        slin_opt, _ = self.num_slin_opt_fast(train_test_inf_mat=train_test_inf_mat, y_train=y_train, y_test=y_test)
        return nil_opt/slin_opt, None

    def gd_class_fast(
        self, train_test_inf_mat: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
    ):
        train_test_inf_mat_tmp = train_test_inf_mat.copy()
        htil_values = None
        for l in np.unique(y_train):
            l_test_ids = np.where(y_test == l)[0]
            inf_of_l_samples_on_test_l_samples = train_test_inf_mat_tmp[:, l_test_ids]
            htil_values_tmp = inf_of_l_samples_on_test_l_samples.sum(axis=1)
            if htil_values is None:
                htil_values = htil_values_tmp.copy()
            else:
                htil_values = np.vstack((htil_values, htil_values_tmp))
        return -htil_values.min(axis=0), htil_values.argmin(axis=0)

    def nil_opt_fast(
        self, train_test_inf_mat: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
    ):
        train_test_inf_mat_tmp = train_test_inf_mat.copy()
        nil_values = None
        for l in np.unique(y_train):
            l_test_ids = np.where(y_test == l)[0]
            inf_of_l_samples_on_test_l_samples = train_test_inf_mat_tmp[:, l_test_ids]
            inf_of_l_samples_on_test_l_samples[
                inf_of_l_samples_on_test_l_samples > 0
            ] = 0
            nil_values_tmp = inf_of_l_samples_on_test_l_samples.sum(axis=1)
            if nil_values is None:
                nil_values = nil_values_tmp.copy()
            else:
                nil_values = np.vstack((nil_values, nil_values_tmp))
        return -1 * nil_values.min(axis=0), nil_values.argmin(axis=0)

    def pil_opt_fast(
        self, train_test_inf_mat: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
    ):
        train_test_inf_mat_tmp = train_test_inf_mat.copy()
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
        return pil_values.max(axis=0), None

    def num_slin_opt_fast(
            self, train_test_inf_mat: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
    ):
        train_test_inf_mat_tmp = train_test_inf_mat.copy()
        slin_values = None
        for l in np.unique(y_train):
            l_test_ids = np.where(y_test == l)[0]
            inf_of_l_samples_on_test_l_samples = train_test_inf_mat_tmp[:, l_test_ids]
            inf_of_l_samples_on_test_l_samples[
                inf_of_l_samples_on_test_l_samples > 0
                ] = 0
            slin_values_tmp = np.sum(
                np.array(inf_of_l_samples_on_test_l_samples) < 0, axis=1
            )
            if slin_values is None:
                slin_values = slin_values_tmp.copy()
            else:
                slin_values = np.vstack((slin_values, slin_values_tmp))
        return slin_values.max(axis=0), None

    def ail_fast(
        self, train_test_inf_mat: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
    ):
        pf, _ = self.pil_fast(train_test_inf_mat, y_train, y_test)
        nf, _ = self.nil_fast(train_test_inf_mat, y_train, y_test)
        return pf + nf, None

    def mpi_fast(self, train_test_inf_mat: np.ndarray):
        train_test_inf_mat_tmp = train_test_inf_mat.copy()
        train_test_inf_mat_tmp[train_test_inf_mat_tmp < 0] = 0
        return train_test_inf_mat_tmp.sum(axis=1), train_test_inf_mat_tmp.var(axis=1)

    def mni_fast(self, train_test_inf_mat: np.ndarray):
        train_test_inf_mat_tmp = train_test_inf_mat.copy()
        train_test_inf_mat_tmp[train_test_inf_mat_tmp > 0] = 0
        return -1 * train_test_inf_mat_tmp.sum(axis=1), train_test_inf_mat_tmp.var(
            axis=1
        )

    def mnic_fast(self, train_test_inf_mat: np.ndarray):
        train_test_inf_mat_tmp = train_test_inf_mat.copy()
        train_test_inf_mat_tmp[train_test_inf_mat_tmp > 0] = 0
        train_test_inf_mat_tmp[train_test_inf_mat_tmp < 0] = 1
        return train_test_inf_mat_tmp.sum(axis=1), train_test_inf_mat_tmp.var(
            axis=1
        )

    def mai_fast(self, train_test_inf_mat: np.ndarray):
        return np.absolute(train_test_inf_mat).sum(axis=1), train_test_inf_mat.var(
            axis=1
        )

    def aai_fast(self, train_test_inf_mat: np.ndarray):
        train_test_inf_mat_tmp = train_test_inf_mat.copy()
        return np.absolute(train_test_inf_mat_tmp).mean(axis=1), train_test_inf_mat_tmp.var(
            axis=1
        )

    def mi_fast(self, train_test_inf_mat: np.ndarray):
        return train_test_inf_mat.sum(axis=1), train_test_inf_mat.var(
            axis=1
        )

    def num_slip_fast(
        self, train_test_inf_mat: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
    ):
        train_test_inf_mat_tmp = train_test_inf_mat.copy()
        slip_values = np.zeros(len(y_train))
        slip_values_var = np.zeros(len(y_train))
        for l in np.unique(y_train):
            l_train_ids = np.where(y_train == l)[0]
            l_test_ids = np.where(y_test == l)[0]
            inf_of_l_samples = train_test_inf_mat_tmp[l_train_ids, :]
            inf_of_l_samples_on_test_l_samples = inf_of_l_samples[:, l_test_ids]
            slip_values[l_train_ids] = np.sum(
                np.array(inf_of_l_samples_on_test_l_samples) > 0, axis=1
            )
            slip_values_var[l_train_ids] = np.var(
                np.array(inf_of_l_samples_on_test_l_samples) > 0, axis=1
            )
        return slip_values, slip_values_var

    def num_slin_fast(
        self, train_test_inf_mat: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
    ):
        train_test_inf_mat_tmp = train_test_inf_mat.copy()
        slin_values = np.zeros(len(y_train))
        slin_values_var = np.zeros(len(y_train))
        for l in np.unique(y_train):
            l_train_ids = np.where(y_train == l)[0]
            l_test_ids = np.where(y_test == l)[0]
            inf_of_l_samples = train_test_inf_mat_tmp[l_train_ids, :]
            inf_of_l_samples_on_test_l_samples = inf_of_l_samples[:, l_test_ids]
            slin_values[l_train_ids] = np.sum(
                np.array(inf_of_l_samples_on_test_l_samples) < 0, axis=1
            )
            slin_values_var[l_train_ids] = np.var(
                np.array(inf_of_l_samples_on_test_l_samples) < 0, axis=1
            )
        return slin_values, slin_values_var

    ### Test Signals ###

    # proponents influence
    def pi(self, inf_mat, ids=None):
        tmp_mat = inf_mat.copy()
        if ids is not None:
            tmp_mat = tmp_mat[ids, :]
        tmp_mat[tmp_mat < 0] = 0
        return np.sum(tmp_mat, axis=0)

    # opponents influence
    def oi(self, inf_mat, ids=None):
        tmp_mat = inf_mat.copy()
        if ids is not None:
            tmp_mat = tmp_mat[ids, :]
        tmp_mat[tmp_mat > 0] = 0
        return np.sum(tmp_mat, axis=0)

    # proponents count
    def pc(self, inf_mat, ids=None):
        tmp_mat = inf_mat.copy()
        if ids is not None:
            tmp_mat = inf_mat[ids, :]
        return np.sum(np.greater(tmp_mat, 0), axis=0)

    # opponents count
    def oc(self, inf_mat, ids=None):
        tmp_mat = inf_mat.copy()
        if ids is not None:
            tmp_mat = inf_mat[ids, :]
        return np.sum(np.less(tmp_mat, 0), axis=0)

    def ppc(self, inf_mat, train_labels, y_preds, count=True):
        tmp_mat = inf_mat.copy()
        unq_pred_labels = np.unique(y_preds)
        cpp_arr = np.zeros(len(y_preds))
        fn = self.pc if count else self.pi
        for l in unq_pred_labels:
            train_l_ids = np.where(train_labels == l)[0]
            pred_l_ids = np.where(y_preds == l)[0]
            pc_arr = fn(tmp_mat, train_l_ids)
            cpp_arr[pred_l_ids] = pc_arr[pred_l_ids]
        return cpp_arr

    # opponents of predicted class
    def ntopc(self, inf_mat, train_labels, y_preds):
        tmp_mat = inf_mat.copy()
        unq_pred_labels = np.unique(y_preds)
        cpo_arr = np.zeros(len(y_preds))
        for l in unq_pred_labels:
            train_l_ids = np.where(train_labels == l)[0]
            pred_l_ids = np.where(y_preds == l)[0]
            oc_arr = self.oc(tmp_mat, train_l_ids) / len(train_l_ids)
            cpo_arr[pred_l_ids] = oc_arr[pred_l_ids]
        return cpo_arr