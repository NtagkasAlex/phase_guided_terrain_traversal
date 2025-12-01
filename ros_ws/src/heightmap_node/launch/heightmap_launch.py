from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('heightmap_node')  # your package name
    param_file = os.path.join(pkg_share, 'config', 'heightmap_params.yaml')

    return LaunchDescription([
        Node(
            package='heightmap_node',
            executable='heightmap_node',
            name='heightmap_node',
            parameters=[param_file],
            output='screen'
        )
    ])
