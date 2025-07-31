from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    mediapipe_ros_dir = get_package_share_directory('mediapipe_ros')

    rgb_msg_topic = DeclareLaunchArgument('rgb_topic',
                                          default_value='/mediapipe/rgb/image_raw',
                                          description='RGB image topic')
    depth_msg_topic = DeclareLaunchArgument('depth_topic',
                                            default_value='/mediapipe/depth_to_rgb/image_raw',
                                            description='Depth to rgb topic')
    depth_msg_info = DeclareLaunchArgument('depth_info_topic',
                                           default_value='/mediapipe/depth_to_rgb/camera_info',
                                           description='Depth camera info')
    min_detection_conf_arg = DeclareLaunchArgument('min_detection_confidence',
                                                   default_value='0.5',
                                                   description='Minimum detection confidence')
    min_tracking_conf_arg = DeclareLaunchArgument('min_tracking_confidence',
                                                  default_value='0.5',
                                                  description='Minimum tracking confidence')
    use_gpu = DeclareLaunchArgument('use_gpu',
                                    default_value=True,
                                    description='Switch between GPU and CPU usage')

    pose_estimation_node = Node(
            package='mediapipe_ros',
            executable='pose_estimation_node',
            name='pose_estimation',
            parameters=[{
                'rgb_topic':LaunchConfiguration('rgb_topic'),
                'depth_topic':LaunchConfiguration('depth_topic'),
                'depth_info_topic':LaunchConfiguration('depth_info_topic'),
                'min_detection_confidence':LaunchConfiguration('min_detection_confidence'),
                'min_tracking_confidence':LaunchConfiguration('min_tracking_confidence'),
                'use_gpu':LaunchConfiguration('use_gpu'),
                'enable':True,
                'rgb_image_reliability':1,
                'depth_image_reliability':1,
                'depth_info_reliability':1,
                }],
            output='screen'
            )
    return LaunchDescription([
        rgb_msg_topic,
        depth_msg_topic,
        depth_msg_info,
        min_detection_conf_arg,
        min_tracking_conf_arg,
        pose_estimation_node
        ])



