import numpy as np
from scipy.signal import convolve2d
import cv2


def rand_matrix(dimension, P, flag, sigma):
    """
    This function generates a random deactivation matrix
    in a single image to control the firing state of domain neurons.
    It is composed of 0 and 1, and generates corresponding Boolean values.
    1 represents that the connection input between the central nerve and the neuron at that location is turned on,
    while 0 represents that the connection input is turned off, also known as peripheral neuron inactivation.
    The probability of inactivation is proportional to the distance from the central neuron.
    """
    # establish an all one matrix with dimension*dimension
    matrix_one = np.ones((dimension, dimension))

    if flag == 'norm':
        # generate a 2D Gaussian kernel of size dimension and normalize it
        kernel_1d = cv2.getGaussianKernel(dimension, sigma)
        matrix = np.outer(kernel_1d, kernel_1d)
        # take the reciprocal of the center point of the generated Gaussian matrix
        pixel_coef = 1 / matrix[dimension // 2, dimension // 2]
        # normalize the values in the matrix
        matrix *= pixel_coef
        # Establish a matrix with values randomly distributed between zero and one
        matrix_random = np.random.rand(dimension, dimension)
        # save the size comparison results of the corresponding position values
        matrix = (matrix_random < matrix).astype(np.float32)  # Convert to float32 for Boolean values
    else:
        matrix_new = np.random.rand(dimension, dimension)
        matrix = (matrix_new < (matrix_one * P)).astype(np.float32)

    return matrix


def rcnn_exam(signal, op_beta, op_at, op_vt, op_af, op_it):
    m, n = signal.shape
    S = signal.astype(float)
    YY = np.zeros((m, n))
    U = np.zeros((m, n))
    E = np.ones((m, n))
    Y = np.zeros((m, n))

    B = op_beta
    V = 1
    aT = op_at
    vT = op_vt
    aF = op_af
    t = op_it
    d = 9
    sigma1 = 4
    sigma2 = 6

    # Generate default Gaussian kernel
    kernel_1d = cv2.getGaussianKernel(d, sigma1)
    W_default = np.outer(kernel_1d, kernel_1d)

    # Set the center value to 0
    W_default[d // 2, d // 2] = 0

    for _ in range(t):
        mask = rand_matrix(d, 0.1, 'norm', sigma2)
        W = np.where(mask, W_default, 0)
        L = convolve2d(Y, W, mode='same')
        U = U * np.exp(-aF) + S * (1 + V * B * L)
        Y = (U > E).astype(np.float64)
        E = np.exp(-aT) * E + vT * Y
        YY += Y

    return YY
