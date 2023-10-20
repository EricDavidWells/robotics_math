import pytest

import os
import sys

from mls.src.math import *

# Test cases for skew_symmetric_matrix
def test_skew_symmetric_matrix():
    # Test a valid input
    input_vector = np.array([1.0, 2.0, 3.0])
    expected_output = np.array([[0, -3, 2],
                               [3, 0, -1],
                               [-2, 1, 0]])
    result = skew_symmetric_matrix(input_vector)
    assert np.allclose(result, expected_output)

    # Test an invalid input (not a 1x3 vector)
    invalid_input_vector = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        skew_symmetric_matrix(invalid_input_vector)

def test_create_homogeneous_xform_with_valid_input():
    # Test with valid input
    rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    translation = np.array([1, 2, 3])
    result = create_homogeneous_xform(rotation, translation)

    expected_result = np.array([[1, 0, 0, 1],
                                [0, 1, 0, 2],
                                [0, 0, 1, 3],
                                [0, 0, 0, 1]])

    assert np.array_equal(result, expected_result)

    # Test with invalid input (neither rotation nor translation is a NumPy array)
    rotation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    translation = [1, 2, 3]

    with pytest.raises(ValueError):
        create_homogeneous_xform(rotation, translation)

    # Test with invalid input (translation vector not 3x1)
    rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    translation = np.array([1, 2])

    with pytest.raises(ValueError):
        create_homogeneous_xform(rotation, translation)

    # Test with invalid input (rotation matrix not 3x3)
    rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
    translation = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        create_homogeneous_xform(rotation, translation)

def test_normalize_matrix_with_valid_input():
    # Create an input matrix
    input_matrix = np.array([[2, 0, 0],
                            [0, 3, 0],
                            [0, 0, 4]])

    # Call the function to get the normalized matrix and the scaling factor
    normalized_matrix, scaling_factor = normalize_matrix(input_matrix)

    # Check if the magnitude of the normalized matrix is close to 1
    magnitude = np.linalg.norm(normalized_matrix)
    assert np.isclose(magnitude, 1.0)

    # Check if the scaling factor is the reciprocal of the magnitude of the input matrix
    input_magnitude = np.linalg.norm(input_matrix)
    assert np.isclose(scaling_factor, 1 / input_magnitude)

    # Test with an input matrix that has zero magnitude (all-zero matrix)
    input_matrix = np.zeros((3, 3))

    with pytest.raises(ValueError):
        normalize_matrix(input_matrix)

if __name__ == "__main__":
    pytest.main()
