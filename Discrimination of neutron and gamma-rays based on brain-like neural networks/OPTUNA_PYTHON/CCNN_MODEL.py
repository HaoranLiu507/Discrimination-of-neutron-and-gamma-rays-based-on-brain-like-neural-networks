import numpy as np
from scipy.signal import convolve2d

def CCNN_exam(I, CCNN_AF,CCNN_AE,CCNN_AL,CCNN_VE,CCNN_BETA,CCNN_K):
    S = np.double(I)
    m, n = I.shape
    TS = 0
    Y = np.zeros((m, n))
    E = Y + 1
    U = Y
    F = Y
    L = Y
    af = CCNN_AF
    ae = CCNN_AE
    al = CCNN_AL
    Ve = CCNN_VE
    beta = CCNN_BETA
    Vf = 0.63
    Vl = Vf
    M = np.array([[0.5, 1, 0.5],
                  [1, 0, 1],
                  [0.5, 1, 0.5]])  # Define a 3x3 matrix M
    W = M
    for t in range(1, CCNN_K):
        F = np.exp(-af) * F + Vf * convolve2d(Y, M, mode='same') + S
        L = np.exp(-al) * L + Vl * convolve2d(Y, W, mode='same')
        U = F * (1 + beta * L)
        Y = 1 / (1 + np.exp(E - U))
        E = np.exp(-ae) * E + Ve * Y
        TS = TS + Y
    return TS
