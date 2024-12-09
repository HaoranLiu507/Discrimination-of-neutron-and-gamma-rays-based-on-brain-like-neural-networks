import optuna
import numpy as np
from CCNN_MODEL import CCNN_exam
import os
from Double_guass import histogram_fitting_and_compute_fom


def objective(trial):
    op_af = trial.suggest_float("op_af", 0.1, 0.9, step=0.05)
    op_ae = trial.suggest_float("op_ae", 0.05, 0.9, step=0.01)
    op_al = trial.suggest_float("op_al", 0.1, 0.9, step=0.05)
    op_ve = trial.suggest_float("op_ve", 5, 20, step=1.5)
    op_beta = trial.suggest_float("op_beta", 0.1, 0.9, step=0.05)
    op_K = trial.suggest_int("op_K", 10, 200, step=5)

    file_path = "your_path"
    signal = np.loadtxt(file_path, delimiter=',')


    R = np.zeros(signal.shape[0])
    for i in range(signal.shape[0]):
        DATA = signal[i, :]
        DATA = DATA.reshape(-1, 1)
        maxposition = np.argmax(DATA)  # Find the maximum position of the signal
        n0 = 7  # Number of samples to the left of the maximum position
        m0 = maxposition - n0  # Start index of the integration window
        n2 = 200  # Number of samples to the right of the maximum position
        ignition = CCNN_exam(DATA, CCNN_AF=op_af, CCNN_AE=op_ae, CCNN_AL=op_al, CCNN_VE=op_ve, CCNN_BETA=op_beta,CCNN_K=op_K)
        # ignition = PCNN_exam(DATA, CCNN_AF=op_af, CCNN_AE=op_ae, CCNN_AL=op_al, CCNN_VE=op_ve, CCNN_BETA=op_beta,
        #                      CCNN_VL=op_vl,
        #                      CCNN_VF=op_vf, CCNN_K=op_K)
        SUM = np.sum(ignition[m0:maxposition + n2 + 1])
        R[i] = SUM
    if np.isnan(R).any():
        R = R[~np.isnan(R)]  # Remove NaNs
        if R.size == 0:  # Check if R is empty after removing NaNs
            return np.nan

    fom = histogram_fitting_and_compute_fom(R)
    return abs(fom)


if __name__ == "__main__":
    storage_name = "sqlite:///optuna_ccnn7.db"
    if os.path.exists("pcnn_optuna_file/optuna_ccnn7.db"):
        print("study resumed")
        study_load = optuna.load_study(storage=storage_name, study_name="optuna_ccnn7.db")
        study_load.optimize(objective, n_trials=1000, timeout=None)
    else:
        print("study created")
        study = optuna.create_study(direction="maximize",
                                    study_name="optuna_ccnn7.db",
                                    storage=storage_name,
                                    load_if_exists=True,
                                    sampler=optuna.samplers.RandomSampler()
                                    )
        study.optimize(objective, n_trials=500, timeout=None)
