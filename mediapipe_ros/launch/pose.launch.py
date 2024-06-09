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
        )
    ])

