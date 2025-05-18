from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Set the path to your URDF/Xacro file
    urdf_file = os.path.join(
        get_package_share_directory('rr_mechanism'),
        'resources',
        'rr_mechanism.urdf'
    )

    rviz_config_path = os.path.join(
        get_package_share_directory('rr_mechanism'),
        'resources',
        'rr_display.rviz'  # You can create and save a custom config here
    )

    return LaunchDescription([
        # Joint State Publisher GUI
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            output='screen'
        ),

        # Robot State Publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': open(urdf_file).read()}]
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_path]
        ),
        Node(
            package='rr_mechanism',
            executable='rr_mechanism',
            name='rr_mechanism',
            parameters=[{'urdf_path': urdf_file},
                        {'tip_frame': 'link_tip'}]
        )
    ])
