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

def test_create_homogeneous_xform_with_invalid_rotation():
    # Test with invalid input (rotation matrix not 3x3)
    rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
    translation = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        create_homogeneous_xform(rotation, translation)

def test_create_homogeneous_xform_with_invalid_translation():
    # Test with invalid input (translation vector not 3x1)
    rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    translation = np.array([1, 2])

    with pytest.raises(ValueError):
        create_homogeneous_xform(rotation, translation)

def test_create_homogeneous_xform_with_invalid_input_type():
    # Test with invalid input (neither rotation nor translation is a NumPy array)
    rotation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    translation = [1, 2, 3]

    with pytest.raises(ValueError):
        create_homogeneous_xform(rotation, translation)

if __name__ == "__main__":
    pytest.main()
