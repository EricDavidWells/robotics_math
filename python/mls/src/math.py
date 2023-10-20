import numpy as np


def skew_symmetric_matrix(vector):
    if vector.shape != (3,):
        raise ValueError("Input vector must be a 1x3 NumPy array.")

    x, y, z = vector
    skew_matrix = np.array([[0, -z, y],
                            [z, 0, -x],
                            [-y, x, 0]])
    return skew_matrix

def create_homogeneous_xform(rotation, translation):
    # Check if the input matrices are NumPy arrays
    if not isinstance(rotation, np.ndarray) or not isinstance(translation, np.ndarray):
        raise ValueError("Both rotation and translation must be NumPy arrays.")

    # Check if the rotation matrix is 3x3
    if rotation.shape != (3, 3):
        raise ValueError("Rotation matrix must be a 3x3 NumPy array.")

    # Check if the translation vector is 3x1
    if translation.shape != (3,):
        raise ValueError("Translation vector must be a 3x1 NumPy array.")

    # Create a 4x4 identity matrix
    homogeneous_matrix = np.eye(4)

    # Populate the upper-left 3x3 submatrix with the rotation matrix
    homogeneous_matrix[:3, :3] = rotation

    # Populate the first three elements of the last column with the translation vector
    homogeneous_matrix[:3, 3] = translation

    return homogeneous_matrix

