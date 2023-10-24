from mls.src.math import *
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R

class Geometry:
    def __init__(self, mesh):
        self.mesh = mesh

class Link:
    def __init__(self, name, geometry=None):
        self.name = name
        self.geometry = geometry
  
class Joint:
    def __init__(self, name, default_twist=None, active_twist=None):
        """
        Initialize a Joint with a name, default twist, and optional active twist, parent link, and child links.

        Args:
            name (str): Name of the joint.
            default_twist (Twist): Default Twist object for the joint.
            active_twist (Twist, optional): Active Twist object for the joint (default is None).
            parent_link (Link, optional): Reference to the parent link (default is None).
            child_links (List[Link], optional): References to child links (default is an empty list).
        """
        self.name = name
        self.default_twist = default_twist
        self.active_twist = active_twist

    def __str__(self):
        return f"Joint (Name: {self.name}, Default Twist: {self.default_twist}, Active Twist: {self.active_twist})"

class KinematicTreeEdge:
    def __init__(self, parent_node=None, child_node=None, link=None):
        """
        Initialize a KinematicTreeEdge.

        Args:
            parent_node (KinematicTreeNode): The parent node associated with the edge.
            child_node (KinematicTreeNode): The child node associated with the edge.
            link (Link): The link associated with the edge.
        """
        self.parent_node = parent_node
        self.child_node = child_node
        self.link = link

class KinematicTreeNode:
    def __init__(self, joint):
        """
        Initialize a KinematicTreeNode.

        Args:
            joint (Joint): The joint associated with the node.
        """
        self.joint = joint
        self.parent_edge = None  # Incoming edge (from parent joint)
        self.child_edges = []    # List of outgoing edges (to child joints)
    
class KinematicTree:
    def __init__(self):
        self.nodes = []  # List of KinematicTreeNodes
        self.edges = []  # List of KinematicTreeEdges

    def add_node(self, node):
        """
        Add a KinematicTreeNode to the tree.

        Args:
            node (KinematicTreeNode): The node to be added.
        """
        self.nodes.append(node)

    def add_edge(self, parent_node, child_node, link):
        """
        Add an edge to the tree between two nodes.

        Args:
            parent_node (KinematicTreeNode): The parent node.
            child_node (KinematicTreeNode): The child node.
            link (Link): The link associated with the edge.
        """
        # Check if the child node already has a parent edge
        if child_node.parent_edge is not None:
            raise ValueError("Child node already has a parent edge.")
        
        # Create the edge and add it to the list of edges
        edge = KinematicTreeEdge(parent_node, child_node, link)
        self.edges.append(edge)

        # Add the edge to the parent and child nodes
        parent_node.child_edges.append(edge)
        child_node.parent_edge = edge

    def print_tree(self):
        """
        Print the kinematic tree graphically to the command line.
        """
        print('\n')
        def print_subtree(node, indent=""):
            if node:
                print(f"{indent}({node.joint.name})")
                for edge in node.child_edges:
                    print(f"{indent}  |")
                    print(f"{indent}  |--- [{edge.link.name}]")

                    next_node = edge.child_node
                    print_subtree(next_node, indent + "  |")
        
        for node in self.nodes:
            if node.parent_edge is None:
                print(node.joint.name)
                for edge in node.child_edges:
                    print(f"  |--- [{edge.link.name}]")
                    child_node = edge.child_node
                    print_subtree(child_node, "  |")

    def update_thetas(self, joint_names, thetas):
        if len(joint_names) != len(thetas):
            raise ValueError("Joint names and theta values must have the same length.")

        for node in self.nodes:
            if node.joint.name in joint_names:
                node.joint.active_twist.theta = thetas[joint_names.index(node.joint.name)]

    def forward_kinematics(self, node):
        if node not in self.nodes:
            raise ValueError("The specified node is not in the list of nodes.")

        if not isinstance(node, KinematicTreeNode):
            raise ValueError("Node should be an instance of KinematicTreeNode.")

        # Initialize an identity matrix
        T = np.eye(4)

        # Traverse up the tree starting from the specified node
        current_node = node
        while current_node is not None:
            default_twist = current_node.joint.default_twist
            active_twist = current_node.joint.active_twist

            # Calculate the transformation matrix for the current joint
            T_joint = homogenous_from_exponential_coordinates(default_twist)
            if active_twist is not None:
                T_joint = T_joint @ homogenous_from_exponential_coordinates(active_twist)

            # Pre-multiply the transformation matrix with the accumulated transformation
            T = T_joint @ T

            # Move to the parent node
            if current_node.parent_edge is None or current_node.parent_edge.parent_node is None: break

        return T

    def get_node_by_joint_name(self, joint_name):
        """
        Get a reference to the internal node by joint name.

        Args:
            joint_name (str): The name of the joint.

        Returns:
            KinematicTreeNode: The internal node with the specified joint name, or None if not found.
        """
        for node in self.nodes:
            if node.joint.name == joint_name:
                return node
        return None

    @classmethod
    def load_from_urdf(cls, urdf_file):
        
        kinematic_tree = KinematicTree()

        xform = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
        tree.add_node(KinematicTreeNode(Joint("origin", 
                   default_twist=homogenous_to_exponential_coordinates(xform))))

        tree = ET.parse(urdf_file)
        root = tree.getroot()

        if root.tag != "robot":
            raise ValueError("Invalid URDF file. 'robot' element is not root.")

        robot_element = root
        robot_name = robot_element.get("name")

        parent_to_child = {}
        child_to_parent = {}

        # joints = []
        # joints.append(Joint("origin", child_link))
        links = []

        # Iterate through all link elements
        for link_element in robot_element.findall("link"):
            link_name = link_element.get("name")
            link = Link(link_name)
            links.append(link)
        

        # Iterate through all joint elements
        for joint_element in robot_element.findall("joint"):
            joint_name = joint_element.get("name")

            # Extract parent and child link names from the URDF
            parent_link_name = joint_element.find("parent").get("link")
            child_link_name = joint_element.find("child").get("link")

            # # Find the corresponding link objects
            parent_link = next((link for link in links if link.name == parent_link_name), None)
            child_link = next((link for link in links if link.name == child_link_name), None)

            if child_link is None:
                raise ValueError(f"link not found in urdf: {child_link}")
            if parent_link is None:
                raise ValueError(f"link not found in urdf: {parent_link}")


            joint_type = joint_element.get("type")
            if joint_type == "revolute":
                axis = float(joint_element.get("axis"))
                active_twist = Twist(np.array(axis), np.array([0, 0, 0]))
            else:
                raise ValueError(f"only revolute joints currently supported")

            # Extract the xyz and rpy attributes
            xyz = joint_element.find("origin").get("xyz").split()
            rpy = joint_element.find("origin").get("rpy").split()

            # Convert the extracted values to float
            x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
            r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
            rotation = R.from_euler("xyz", [r, p, y]).as_matrix()
            xform = homogenous_from_rotation_translation(rotation, np.array([x, y, z]))
            default_twist = homogenous_to_exponential_coordinates(xform)

            kinematic_tree.add_node(Joint(joint_name, default_twist=default_twist, active_twist=active_twist))
            kinematic_Tree.add_edge()



            # joint = Joint(joint_name, twist, active_twist, parent_joint, child_joint)
            # joints.append(joint)



# class Robot:
#     def __init__(self, name, joints=[], links=[]):
#         """
#         Initialize a Robot with a name and a list of Joint objects.

#         Args:
#             name (str): Name of the robot.
#             joints (list, optional): List of Joint objects (default is an empty list).
#         """
#         self.name = name
#         self.joints = joints
#         self.links = links

#     def __str__(self):
#         return f"Robot (Name: {self.name}, Joints: {self.joints})"




#     # @classmethod
#     # def load_from_urdf(cls, urdf_file):
#         tree = ET.parse(urdf_file)
#         root = tree.getroot()

#         if root.tag != "robot":
#             raise ValueError("Invalid URDF file. 'robot' element is not root.")

#         robot_element = root
#         robot_name = robot_element.get("name")

#         joints = []
#         joints.append(Joint("origin", child_link))
#         links = []

#         # Iterate through all link elements
#         for link_element in robot_element.findall("link"):
#             link_name = link_element.get("name")
#             # Extract link information from the URDF
#             # You can customize this part based on the structure of your URDF
#             # For example, you may need to extract link properties, visual, collision, etc.
#             link = Link(link_name)
#             links.append(link)
        

#         # Iterate through all joint elements
#         for joint_element in robot_element.findall("joint"):
#             joint_name = joint_element.get("name")

#             # Extract parent and child link names from the URDF
#             parent_link_name = joint_element.find("parent").get("link")
#             child_link_name = joint_element.find("child").get("link")

#             # # Find the corresponding link objects
#             parent_link = next((link for link in links if link.name == parent_link_name), None)
#             child_link = next((link for link in links if link.name == child_link_name), None)

#             joints.append(Joint(joint_name, parent_link=parent_link, child_links=[child_link]))


#             # if child_link is None:
#             #     raise ValueError(f"link not found in urdf: {child_link}")
#             # if parent_link is None:
#             #     raise ValueError(f"link not found in urdf: {parent_link}")

#             # # Extract the xyz and rpy attributes
#             # xyz = joint_element.find("origin").get("xyz").split()
#             # rpy = joint_element.find("origin").get("rpy").split()

#             # # Convert the extracted values to float
#             # x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
#             # r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
#             # rotation = R.from_euler("xyz", [r, p, y]).as_matrix()
#             # xform = homogenous_from_rotation_translation(rotation, np.array([x, y, z]))
#             # twist = homogenous_to_exponential_coordinates(xform)

#             # joint_type = joint_element.get("type")
#             # if joint_type == "revolute":
#             #     axis = joint_element.get("axis")
#             #     # active_twist = Twist()

#             # joint = Joint(joint_name, twist, active_twist, parent_joint, child_joint)
#             # joints.append(joint)

#         # Usage example
#         # Define your list of joints and links, then call the function
#         root_links = find_root_links(joints, links)
#         if len(root_links) == 0:
#             raise ValueError("No root link found.")
#         elif len(root_links) > 1:
#             raise ValueError("Multiple root links found")
#         else:
#             print("Root link of {root_link[0]} found")

#         root_link = root_links[0]
#         joint_tree = create_tree_structure(links, joints, root_link)

#         return cls(robot_name, joints, links)

# def find_root_links(joints, links):
#     # Create a set of link names that are children in joints
#     child_links = set(joint.child_links[0] for joint in joints if joint.child_links)

#     root_links = []

#     # Find the links that is not a child of any other link
#     for link in links:
#         if link not in child_links:
#             root_links.append(link)

#     return root_links

# def create_tree_structure(links, joints, root_link):
#     def build_tree(link):
#         child_nodes = []
#         for joint in joints:
#             if joint.parent_link == link:
#                 child_link = joint.child_links[0]
#                 child_node = build_tree(child_link)
#                 child_nodes.append(child_node)
#         return TreeNode(link, child_nodes)

#     return build_tree(root_link)
