import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    config_dir = get_package_share_directory('elevation_mapping')
    fused_config = os.path.join(config_dir, 'config', 'visualization', 'fused.yaml')
    raw_config = os.path.join(config_dir, 'config', 'visualization', 'raw.yaml')

    return LaunchDescription([
        Node(
            package='grid_map_visualization',
            executable='grid_map_visualization',
            name='elevation_map_fused_visualization',
            output='screen',
            parameters=[fused_config]
        ),
        Node(
            package='grid_map_visualization',
            executable='grid_map_visualization',
            name='elevation_map_raw_visualization',
            output='screen',
            parameters=[raw_config]
        ),
    ])
