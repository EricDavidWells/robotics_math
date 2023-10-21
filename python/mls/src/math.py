import numpy as np
from math import cos, sin, tan, acos

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

def create_twist(w, v):
    """
    Create a 6D twist vector from angular velocity (w) and linear velocity (v).

    Args:
        w (numpy.ndarray): 3D angular velocity vector.
        v (numpy.ndarray): 3D linear velocity vector.

    Returns:
        numpy.ndarray: 6D twist vector [angular_velocity, linear_velocity].

    Raises:
        ValueError: If w and v are not both 3D vectors.
    """
    if w.shape != (3,) or v.shape != (3,):
        raise ValueError("Both w and v must be 3D vectors.")

    return np.concatenate((w, v))

def homogenous_from_exponential_coordinates(twist, theta):
    """
    Convert a twist to a 4x4 homogeneous transformation matrix.

    Args:
        twist (numpy.ndarray): 6D twist vector [angular_velocity, linear_velocity].
        theta (float): Rotation angle (for the magnitude of the twist).

    Returns:
        numpy.ndarray: 4x4 homogeneous transformation matrix.

    Raises:
        ValueError: If the twist vector is not a 6D vector.
    """
    if twist.shape != (6,):
        raise ValueError("Input twist vector must be a 6D vector.")

    # Split the twist vector into angular velocity and linear velocity components
    w = twist[:3]
    v = twist[3:]

    # Calculate the rotation matrix using the Rodrigues' formula
    p = np.zeros((3,))
    R = np.eye(3)
    if np.allclose(w, 0):
        p = v * theta
    else:
        R = rotation_from_canonical_coordinates(w, theta)
        p = (np.eye(3) - R)@skew_symmetric_matrix(w)@v + w.reshape(3,1)@w.reshape(1,3)@v*theta

    # Create the 4x4 homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p

    return T

def homogenous_to_exponential_coordinates(T):
    """
    Convert a homogeneous transformation matrix to a twist.

    Args:
        T (numpy.ndarray): 4x4 homogeneous transformation matrix.

    Returns:
        numpy.ndarray: 6D twist vector [angular_velocity, linear_velocity].

    Raises:
        ValueError: If the input matrix is not a 4x4 matrix.
    """
    if T.shape != (4, 4):
        raise ValueError("Input matrix must be a 4x4 homogeneous transformation matrix.")

    # Extract the rotation matrix and translation vector
    R = T[:3, :3]
    p = T[:3, 3]        

    w = np.zeros((3,))
    v = np.zeros((3,))
    if np.allclose(R, np.eye(3)):
        theta = np.linalg.norm(p)
        v = p/theta
    else:
        
      # Calculate the angular velocity (skew-symmetric matrix)
      w, theta = rotation_to_canonical_coordinates(R)
      A = (np.eye(3) - R)@skew_symmetric_matrix(w) + w.reshape(3,1)@w.reshape(1,3) * theta

      # Calculate the linear velocity
      v = np.linalg.inv(A) @ p

    return np.concatenate((w, v)), theta
