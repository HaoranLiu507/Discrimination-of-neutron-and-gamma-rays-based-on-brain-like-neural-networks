import optuna
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from RCNN_MODEL import rcnn_exam
# from pcnn import PCNN_exam
from Double_guass import histogram_fitting_and_compute_fom


def objective(trial):
    af = trial.suggest_float("af", 0.1, 0.9, step=0.05)
    at = trial.suggest_float("at", 0.05, 0.9, step=0.01)
    vt = trial.suggest_float("vt", 0.1, 0.9, step=0.01)
    beta = trial.suggest_float("beta", 0.1, 0.9, step=0.05)
    iterop = trial.suggest_int("iterop", 10, 200, step=1)

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
        ignition = rcnn_exam(DATA, op_beta=beta, op_at=at, op_vt=vt, op_af=af, op_it=iterop)
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
    storage_name = "sqlite:///optuna_rcnn_fom2.db"
    if os.path.exists("H:\\optuna_rcnn\\rcnn_optuna_file\\optuna_rcnn_fom2.db"):
        print("study resumed")
        study_load = optuna.load_study(storage=storage_name, study_name="optuna_rcnn_fom2")
        study_load.optimize(objective, n_trials=1000, timeout=None)
    else:
        print("study created")
        study = optuna.create_study(direction="maximize",
                                    study_name="optuna_rcnn_fom2",
                                    storage=storage_name,
                                    load_if_exists=True,
                                    sampler=optuna.samplers.RandomSampler()
                                    )
        study.optimize(objective, n_trials=500, timeout=None)
