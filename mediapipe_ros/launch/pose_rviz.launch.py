import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mediapipe_ros',
            executable='pose_estimation_node',
            name='pose_estimation',
        ),
        Node(
            package='mediapipe_ros',
            executable='visualization_node',
            name='visualization',
        ),
        ExecuteProcess(
            cmd=['rviz2', '-d', os.path.join(get_package_share_directory('mediapipe_ros'), 'config', 'pose_rviz.rviz')],
            output='screen'
        )
    ])

def get_package_share_directory(package_name):
    from ament_index_python.packages import get_package_share_directory
    return get_package_share_directory(package_name)

