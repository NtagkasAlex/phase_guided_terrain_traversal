from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory('grid_map_filter_node'),
        'config',
        'filters_demo.yaml'
    )

    return LaunchDescription([
        Node(
            package='grid_map_filter_node',
            executable='filters_demo_node',
            name='grid_map_filters_node',
            parameters=[config_file]
            #arguments=['--ros-args', '--log-level', 'DEBUG']
        )
    ])

