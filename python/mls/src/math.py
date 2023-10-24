import numpy as np
from math import cos, sin, tan, acos

class Twist:
    def __init__(self, w, v, theta=0.0):
        """
        Initialize a Twist object with angular velocity (w), linear velocity (v), and an optional theta.

        Args:
            w (numpy.ndarray): 1x3 angular velocity vector.
            v (numpy.ndarray): 1x3 linear velocity vector.
            theta (float, optional): Rotation angle (default is 0).

        Raises:
            ValueError: If w and v are not 1x3 vectors.
        """
        if w.shape != (3,) or v.shape != (3,):
            raise ValueError("Both w and v must be 1x3 vectors.")
        
        self.w = w
        self.v = v
        self.theta = theta

    def __str__(self):
        return f"Twist (w: {self.w}, v: {self.v}, theta: {self.theta})"

def skew_symmetric_matrix(vector):
    if vector.shape != (3,):
        raise ValueError("Input vector must be a 1x3 NumPy array.")

    x, y, z = vector
    skew_matrix = np.array([[0, -z, y],
                            [z, 0, -x],
                            [-y, x, 0]])
    return skew_matrix

def rotation_from_canonical_coordinates(vector, theta):
    if vector.shape != (3,):
        raise ValueError("Input vector must be a 1x3 NumPy array.")
    
    vector = vector / np.linalg.norm(vector)
    matrix = skew_symmetric_matrix(vector)
    return np.eye(3) + matrix * sin(theta) + matrix @ matrix * (1 - cos(theta))

def rotation_to_canonical_coordinates(matrix):
    if matrix.shape != (3, 3):
        raise ValueError("Input matrix must be a 3x3 NumPy array.")

    theta = acos((matrix_trace(matrix) - 1)/2)
    w = 1/(2*sin(theta)) * np.array([matrix[2,1] - matrix[1,2], matrix[0,2] - matrix[2,0], matrix[1,0] - matrix[0,1]])
    return w, theta

def matrix_trace(matrix):
    """
    Compute the trace of a matrix.

    Args:
        matrix (numpy.ndarray): Input matrix.

    Returns:
        float: The trace of the matrix.

    Raises:
        ValueError: If the input is not a square matrix.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square for trace computation.")

    trace = np.trace(matrix)
    return trace

def homogenous_from_rotation_translation(rotation, translation):
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

def homogenous_from_exponential_coordinates(twist):
    if not isinstance(twist, Twist):
        raise ValueError("Input twist must be an instance of the Twist class.")

    w = twist.w
    v = twist.v
    theta = twist.theta

    p = np.zeros((3,))
    R = np.eye(3)
    if np.allclose(w, np.zeros(3)):
        p = v * theta
    else:
        R = rotation_from_canonical_coordinates(w, theta)
        p = (np.eye(3) - R) @ skew_symmetric_matrix(w) @ v + np.outer(w, w) @ v * theta

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p

    return T

def homogenous_to_exponential_coordinates(T):
    if T.shape != (4, 4):
        raise ValueError("Input matrix must be a 4x4 homogeneous transformation matrix.")

    R = T[:3, :3]
    p = T[:3, 3]

    w = np.zeros((3,))
    v = np.zeros((3,))
    if np.allclose(R, np.eye(3)):
        theta = np.linalg.norm(p)
        if np.isclose(theta, 0): v = np.zeros((3,))
        else: v = p / theta
    else:
        w, theta = rotation_to_canonical_coordinates(R)
        A = (np.eye(3) - R) @ skew_symmetric_matrix(w) + np.outer(w, w) * theta
        v = np.linalg.inv(A) @ p

    return Twist(w, v, theta)
