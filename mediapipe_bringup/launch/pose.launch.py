from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="mediapipe_ros",
            executable="pose_estimation_node",
            namespace="mediapipe",
            name="pose_estimation_node",
            ),
        Node(
            package="mediapipe_ros",
            executable="visulization_node",
            namespace="mediapipe",
            name="visulization_node",
            ),
        ])

