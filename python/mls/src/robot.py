from mls.src.math import *
import numpy as np
import lxml.etree as ET

class Geometry:
    def __init__(self, mesh):
        self.mesh = mesh

class Link:
    def __init__(self, name, geometry=None):
        self.name = name
        self.geometry = geometry
  
class Joint:
    def __init__(self, name, default_twist=None, active_twist=None, limit=None, joint_type=None):
        """
        Initialize a Joint with a name, default twist, and optional active twist, parent link, and child links.

        Args:
            name (str): Name of the joint.
            default_twist (Twist): Default Twist object for the joint.
            active_twist (Twist, optional): Active Twist object for the joint (default is None).
            parent_link (Link, optional): Reference to the parent link (default is None).
            child_links (List[Link], optional): References to child links (default is an empty list).
            joint_type (str): used to write tree back to urdf format
        """
        self.name = name
        self.default_twist = default_twist
        self.active_twist = active_twist
        self.joint_type = joint_type
        self.limit = limit


    def __str__(self):
        return f"Joint (Name: {self.name}, Default Twist: {self.default_twist}, Active Twist: {self.active_twist})"

class KinematicTreeEdge:
    def __init__(self, parent_node=None, child_node=None, joint=None):
        self.parent_node = parent_node
        self.child_node = child_node
        self.joint = joint

class KinematicTreeNode:
    def __init__(self, link):
        self.link = link
        self.parent_edge = None
        self.child_edges = []
    
class KinematicTree:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, parent_node, child_node, joint):
        if child_node.parent_edge:
            raise ValueError("Child node already has a parent edge.")
        edge = KinematicTreeEdge(parent_node, child_node, joint)
        self.edges.append(edge)
        parent_node.child_edges.append(edge)
        child_node.parent_edge = edge

    def print_tree(self):
        def print_subtree(node, indent=""):
            if node:
                print(f"{indent}({node.link.name})")
                for edge in node.child_edges:
                    print(f"{indent}  |")
                    print(f"{indent}  |--- [{edge.joint.name}]")
                    next_node = edge.child_node
                    print_subtree(next_node, indent + "  |")

        print('\n')
        for node in self.nodes:
            if not node.parent_edge:
                print_subtree(node, "")

    def update_thetas(self, joint_names, thetas):
        if len(joint_names) != len(thetas):
            raise ValueError("Joint names and theta values must have the same length.")

        for edge in self.edges:
            if edge.joint.name in joint_names:
                edge.joint.active_twist.theta = thetas[joint_names.index(edge.joint.name)]

    def forward_kinematics(self, edge):
        if edge not in self.edges:
            raise ValueError("The specified node is not in the list of nodes.")

        if not isinstance(edge, KinematicTreeEdge):
            raise ValueError("Node should be an instance of KinematicTreeEdge.")

        # Initialize an identity matrix
        T = np.eye(4)

        # Traverse up the tree starting from the specified node
        current_edge = edge
        while current_edge is not None:
            default_twist = current_edge.joint.default_twist
            active_twist = current_edge.joint.active_twist

            # Calculate the transformation matrix for the current joint
            T_joint = homogenous_from_exponential_coordinates(default_twist)
            if active_twist is not None:
                T_joint = T_joint @ homogenous_from_exponential_coordinates(active_twist)

            # Pre-multiply the transformation matrix with the accumulated transformation
            T = T_joint @ T

            # Move to the parent edge
            if current_edge.parent_node is None or current_edge.parent_node.parent_edge is None: break
            current_edge = current_edge.parent_node.parent_edge

        return T
    
    def get_edge_by_joint_name(self, joint_name):
        """
        Get a reference to the internal edge by joint name.

        Args:
            joint_name (str): The name of the joint.

        Returns:
            KinematicTreeEdge: The internal edge with the specified joint name, or None if not found.
        """
        for edge in self.edges:
            if edge.joint.name == joint_name:
                return edge
        return None
    
    def get_node_by_link_name(self, link_name):
        """
        Get a reference to the internal node by link name.

        Args:
            link_name (str): The name of the link.

        Returns:
            KinematicTreeNode: The internal node with the specified link name, or None if not found.
        """
        for node in self.nodes:
            if node.link.name == link_name:
                return node
        return None

    @classmethod
    def load_from_urdf(cls, urdf_file):
        
        tree = ET.parse(urdf_file)
        root = tree.getroot()

        if root.tag != "robot":
            raise ValueError("Invalid URDF file. 'robot' element is not root.")

        robot_element = root
        robot_name = robot_element.get("name")

        kinematic_tree = KinematicTree()
        # Iterate through all link elements
        for link_element in robot_element.findall("link"):
            link_name = link_element.get("name")
            link = Link(link_name)

            treenode = KinematicTreeNode(link)
            kinematic_tree.add_node(treenode)

        # Iterate through all joint elements
        for joint_element in robot_element.findall("joint"):
            joint_name = joint_element.get("name")

            # Extract parent and child link names from the URDF
            child_link_name = joint_element.find("child").get("link")
            parent_link_name = joint_element.find("parent").get("link")

            # # Find the corresponding link objects
            parent_node = kinematic_tree.get_node_by_link_name(parent_link_name)
            child_node = kinematic_tree.get_node_by_link_name(child_link_name)

            if parent_node is None:
                raise ValueError(f"link not found in urdf: {parent_link_name}")
            if child_node is None:
                raise ValueError(f"link not found in urdf: {child_link_name}")

            joint_type = joint_element.get("type")
            if joint_type == "revolute":
                axis = [float(value) for value in joint_element.find("axis").get("xyz").split()]
                active_twist = Twist(np.array(axis), np.array([0, 0, 0]))
                limit = [float(joint_element.find("limit").get("lower")), float(joint_element.find("limit").get("upper"))]
            elif joint_type == "fixed":
                active_twist = Twist(np.array([0, 0, 0]), np.array([0, 0, 0]))
                limit = None
            elif joint_type == "continuous":
                axis = [float(value) for value in joint_element.find("axis").get("xyz").split()]
                active_twist = Twist(np.array(axis), np.array([0, 0, 0]))
                limit = None
            elif joint_type == "prismatic":
                active_twist = Twist(np.array([0, 0, 0], np.array(axis)))
                limit = [float(joint_element.find("limit").get("lower")), float(joint_element.find("limit").get("upper"))]
            else:
                raise ValueError(f"joint type of {joint_type} not supported")

            # Extract the xyz and rpy attributes
            xyz = joint_element.find("origin").get("xyz").split()
            rpy = joint_element.find("origin").get("rpy").split()

            # Convert the extracted values to float
            x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
            print(y)
            r = float(rpy[0])
            p = float(rpy[1])
            yaw = float(rpy[2])
            r, p, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
            rotation = extrinsic_rpy_to_rotation(r, p, yaw)
            xform = homogenous_from_rotation_translation(rotation, np.array([x, y, z]))
            default_twist = homogenous_to_exponential_coordinates(xform)

            joint = Joint(joint_name, 
                          default_twist=default_twist,
                          active_twist=active_twist,
                          joint_type=joint_type,
                          limit=limit)

            kinematic_tree.add_edge(parent_node, child_node, joint)

        return kinematic_tree

    def to_urdf(self, urdf_file):
        # Create the root URDF element
        robot_element = ET.Element("robot", name="your_robot_name")

        for node in self.nodes:
            # Create a link element for each node
            link_element = ET.Element("link", name=node.link.name)
            robot_element.append(link_element)        

        for edge in self.edges:
            
            xform = homogenous_from_exponential_coordinates(edge.joint.default_twist)
            rotation, translation = homogenous_to_rotation_translation(xform)

            joint_element = ET.Element("joint")
            joint_element.set("name", edge.joint.name)
            joint_element.set("type", edge.joint.joint_type)
            origin_element = ET.Element("origin")
            r, p, yaw = extrinsic_rpy_from_rotation(rotation)
            x, y, z = translation
            origin_element.set("xyz", f"{x} {y} {z}")
            origin_element.set("rpy", f"{r} {p} {yaw}")
            joint_element.append(origin_element)

            parent_element = ET.Element("parent")
            parent_element.set("link", edge.parent_node.link.name)
            joint_element.append(parent_element)
            child_element = ET.Element("child")
            child_element.set("link", edge.child_node.link.name)
            joint_element.append(child_element)
            axis_element = ET.Element("axis")
            if edge.joint.joint_type == "revolute":
              w = edge.joint.active_twist.w
              axis_element.set("xyz", f"{w[0]} {w[1]} {w[2]}")
            elif edge.joint.joint_type == "continuous":
              w = edge.joint.active_twist.w
              axis_element.set("xyz", f"{w[0]} {w[1]} {w[2]}")
            elif edge.joint.joint_type == "prismatic":
              v = edge.joint.active_twist.v
              axis_element.set("xyz", f"{v[0]} {v[1]} {v[2]}")
            joint_element.append(axis_element)

            if (edge.joint.limit):
              limit_element = ET.Element("limit")
              limit_element.set("effort", "0")
              limit_element.set("velocity", "0")
              limit_element.set("lower", f"{edge.joint.limit[0]}")
              limit_element.set("upper", f"{edge.joint.limit[1]}")
              joint_element.append(limit_element)

            robot_element.append(joint_element)
        # for edge in self.edges:
        #     joint_element = ET.E

        # Create the XML tree
        tree = ET.ElementTree(robot_element)

        # Save the tree to the URDF file
        tree.write(urdf_file, pretty_print=True)
