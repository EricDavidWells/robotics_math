import numpy as np

def skew_symmetric_matrix(vector):
    if vector.shape != (3,):
        raise ValueError("Input vector must be a 1x3 NumPy array.")

    x, y, z = vector
    skew_matrix = np.array([[0, -z, y],
                            [z, 0, -x],
                            [-y, x, 0]])
    return skew_matrix
