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

    # Additional test cases
    # You can add more test cases here to further validate the function

if __name__ == "__main__":
    pytest.main()
