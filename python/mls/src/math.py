import numpy as np


def normalize_matrix(matrix):
    """
    Normalize a matrix by scaling its values so that the magnitude of the matrix is 1.

    Args:
        matrix (numpy.ndarray): Input matrix.

    Returns:
        (numpy.ndarray, float): A tuple containing the normalized matrix and the scaling factor.

    Raises:
        ValueError: If the input matrix has a magnitude of 0 (i.e., all-zero matrix).

    """
    # Calculate the magnitude of the input matrix
    magnitude = np.linalg.norm(matrix)

    # Check if the magnitude is zero
    if np.isclose(magnitude, 0):
        raise ValueError("Input matrix has a magnitude of 0 (all-zero matrix).")

    # Calculate the scaling factor
    scaling_factor = 1 / magnitude

    # Normalize the matrix by scaling its values
    normalized_matrix = matrix * scaling_factor

    return normalized_matrix, scaling_factor


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

