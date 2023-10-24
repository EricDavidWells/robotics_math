import pytest

import os
import sys

from mls.src.math import *
from mls.src.robot import *
from math import sin, cos, tan, pi

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

def test_compute_trace_with_valid_input():
    # Test with a valid square matrix
    matrix = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

    trace = matrix_trace(matrix)
    expected_trace = 15  # Sum of the diagonal elements
    assert trace == expected_trace

    # Test with a non-square matrix
    non_square_matrix = np.array([[1, 2, 3],
                                 [4, 5, 6]])

    with pytest.raises(ValueError):
        matrix_trace(non_square_matrix)

def test_homogenous_from_rotation_translation_with_valid_input():
    # Test with valid input
    rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    translation = np.array([1, 2, 3])
    result = homogenous_from_rotation_translation(rotation, translation)

    expected_result = np.array([[1, 0, 0, 1],
                                [0, 1, 0, 2],
                                [0, 0, 1, 3],
                                [0, 0, 0, 1]])

    assert np.array_equal(result, expected_result)

    # Test with invalid input (neither rotation nor translation is a NumPy array)
    rotation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    translation = [1, 2, 3]

    with pytest.raises(ValueError):
        homogenous_from_rotation_translation(rotation, translation)

    # Test with invalid input (translation vector not 3x1)
    rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    translation = np.array([1, 2])

    with pytest.raises(ValueError):
        homogenous_from_rotation_translation(rotation, translation)

    # Test with invalid input (rotation matrix not 3x3)
    rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
    translation = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        homogenous_from_rotation_translation(rotation, translation)

def test_rotation_from_canonical_coordinates():
    # Valid input
    vector = np.array([1, 0, 0])
    theta = np.pi / 2
    result = rotation_from_canonical_coordinates(vector, theta)
    expected_result = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    assert np.allclose(result, expected_result)

    # Invalid input: vector shape is not 1x3
    with pytest.raises(ValueError):
        rotation_from_canonical_coordinates(np.zeros(4), theta)

    # compare with https://www.andre-gaschler.com/rotationconverter/
    vector = np.array([1, 2, 3])
    theta = np.linalg.norm(vector)
    result = rotation_from_canonical_coordinates(vector, theta)
    expected_result = np.array([[-0.6949205,  0.7135210,  0.0892929], [-0.1920070, -0.3037851,  0.9331924], [0.6929781,  0.6313497,  0.3481075]])
    assert np.allclose(result, expected_result)

    # work backwards, note that expected values can be negative (two solutions)
    matrix = np.array([[-0.6949205,  0.7135210,  0.0892929], [-0.1920070, -0.3037851,  0.9331924], [0.6929781,  0.6313497,  0.3481075]])
    expected_theta = np.linalg.norm(np.array([1, 2, 3]))
    expected_w = np.array([1, 2, 3]) / expected_theta

    w, theta = rotation_to_canonical_coordinates(matrix)

    assert np.isclose(theta, expected_theta) or np.isclose(theta, 2*np.pi - expected_theta)
    assert np.allclose(w, expected_w) or np.allclose(w, -expected_w)

def test_homogenous_twist():
    l1 = 3
    theta = 1
    v = np.array([l1, 0, 0])
    w = np.array([0, 0, 1])
    twist = Twist(w, v, theta)
    expected_result = np.array([[cos(theta), -sin(theta), 0, l1 * sin(theta)],
                                [sin(theta), cos(theta), 0, l1 * (1 - cos(theta))],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])

    g = homogenous_from_exponential_coordinates(twist)
    assert np.allclose(g, expected_result)

    new_twist = homogenous_to_exponential_coordinates(g)
    assert np.allclose(twist.w, new_twist.w)
    assert np.allclose(twist.v, new_twist.v)
    assert np.isclose(twist.theta, new_twist.theta)

    # test pure translation
    l1 = 3
    theta = 1
    v = np.array([1, 0, 0])
    w = np.array([0, 0, 0])
    twist = Twist(w, v, theta)
    expected_result = np.array([[1, 0, 0, theta],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
    g = homogenous_from_exponential_coordinates(twist)
    assert np.allclose(g, expected_result)

    new_twist = homogenous_to_exponential_coordinates(g)
    assert np.allclose(twist.w, new_twist.w)
    assert np.allclose(twist.v, new_twist.v)
    assert np.isclose(twist.theta, new_twist.theta)

def test_kinematic_tree_creation():
    # Create some Joint objects
    joint0 = Joint("origin", Twist(np.array([0.1, 0.2, 0.3]), np.array([1.0, 2.0, 3.0]), theta=0.1))
    joint1 = Joint("Joint1", Twist(np.array([0.1, 0.2, 0.3]), np.array([1.0, 2.0, 3.0]), theta=0.1))
    joint2 = Joint("Joint2", Twist(np.array([0.2, 0.3, 0.4]), np.array([2.0, 3.0, 4.0]), theta=0.2))

    link1 = Link("link1")
    link2 = Link("link2")

    kinematicTreeNode0 = KinematicTreeNode(joint0)
    kinematicTreeNode1 = KinematicTreeNode(joint1)
    kinematicTreeNode2 = KinematicTreeNode(joint2)

    tree = KinematicTree()

    tree.add_node(kinematicTreeNode0)
    tree.add_node(kinematicTreeNode1)
    tree.add_node(kinematicTreeNode2)

    tree.add_edge(kinematicTreeNode0, kinematicTreeNode1, link1)
    tree.add_edge(kinematicTreeNode1, kinematicTreeNode2, link2)

    tree.print_tree()

def test_kinematic_tree_forward_kinematics():
    # example 4.1.2 in Modern Robotics

    xform = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
    origin = Joint("origin", 
                   default_twist=homogenous_to_exponential_coordinates(xform))
    xform = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
    active_twist = Twist(np.array([0, 0, 1]), np.array([0, 0, 0]))
    joint0 = Joint("joint_0", 
                   default_twist=homogenous_to_exponential_coordinates(xform),
                   active_twist=active_twist)
    xform = np.array([
        [1, 0, 0, 10],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
    active_twist = Twist(np.array([0, 1, 0]), np.array([0, 0, 0]))
    joint1 = Joint("joint_1", 
                   default_twist=homogenous_to_exponential_coordinates(xform),
                   active_twist=active_twist)

    xform = np.array([
        [1, 0, 0, 5],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
    active_twist = Twist(np.array([0, 0, 1]), np.array([0, 0, 0]))
    joint2 = Joint("joint_2", 
                   default_twist=homogenous_to_exponential_coordinates(xform),
                   active_twist=active_twist)

    link1 = Link("link1")
    link1 = Link("link2")

    kinematicTreeNodeorigin = KinematicTreeNode(origin)
    kinematicTreeNode0 = KinematicTreeNode(joint0)
    kinematicTreeNode1 = KinematicTreeNode(joint1)
    kinematicTreeNode2 = KinematicTreeNode(joint2)

    tree = KinematicTree()
    tree.add_node(kinematicTreeNodeorigin)
    tree.add_node(kinematicTreeNode0)
    tree.add_node(kinematicTreeNode1)
    tree.add_node(kinematicTreeNode2)
    tree.add_edge(kinematicTreeNode0, kinematicTreeNode1, link1)
    tree.add_edge(kinematicTreeNode1, kinematicTreeNode2, link1)
    tree.print_tree()

    tree.update_thetas(["joint_0", "joint_1", "joint_2"], [pi, pi, pi/6])
    final_xform = tree.forward_kinematics(tree.get_node_by_joint_name("joint_2"))
    print(np.around(final_xform, 2))
    tree.update_thetas(["joint_0", "joint_1", "joint_2"], [pi, pi, pi/6+0.1])
    final_xform = tree.forward_kinematics(tree.get_node_by_joint_name("joint_2"))
    print(np.around(final_xform, 2))

    
# def test_load_robot_from_urdf():
    
#     urdf_file_path = os.path.join(os.path.dirname(__file__), "../urdfs/base_finger.urdf")

#     print("wtf")
#     print(urdf_file_path)
#     # Load the Robot from the URDF file
#     robot = Robot.load_from_urdf(urdf_file_path)

#     # Assert that the Robot is created successfully
#     assert robot.name == "base_finger"  # Replace with the actual robot name
  

if __name__ == "__main__":
    pytest.main()
