from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Declare all launch arguments
    enable_arg = DeclareLaunchArgument(
        'enable',
        default_value='true',
        description='Enable pose estimation'
    )
    
    rgb_reliability_arg = DeclareLaunchArgument(
        'rgb_image_reliability',
        default_value='1',  # 1 corresponds to BEST_EFFORT
        description='QoS reliability setting for RGB image subscription'
    )
    
    depth_reliability_arg = DeclareLaunchArgument(
        'depth_image_reliability',
        default_value='1',  # 1 corresponds to BEST_EFFORT
        description='QoS reliability setting for depth image subscription'
    )
    
    depth_info_reliability_arg = DeclareLaunchArgument(
        'depth_info_reliability',
        default_value='1',  # 1 corresponds to BEST_EFFORT
        description='QoS reliability setting for depth camera info subscription'
    )
    
    rgb_topic_arg = DeclareLaunchArgument(
        'rgb_topic',
        default_value='/mediapipe/rgb/image_raw',
        description='RGB image topic'
    )
    
    depth_topic_arg = DeclareLaunchArgument(
        'depth_topic',
        default_value='/mediapipe/depth_to_rgb/image_raw',
        description='Depth image topic'
    )
    
    depth_info_topic_arg = DeclareLaunchArgument(
        'depth_info_topic',
        default_value='/mediapipe/depth_to_rgb/camera_info',
        description='Depth camera info topic'
    )
    
    # Get the LaunchConfiguration objects
    enable = LaunchConfiguration('enable')
    rgb_reliability = LaunchConfiguration('rgb_image_reliability')
    depth_reliability = LaunchConfiguration('depth_image_reliability')
    depth_info_reliability = LaunchConfiguration('depth_info_reliability')
    rgb_topic = LaunchConfiguration('rgb_topic')
    depth_topic = LaunchConfiguration('depth_topic')
    depth_info_topic = LaunchConfiguration('depth_info_topic')
    
    # Define the static transform publisher
    static_transform_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_to_world_broadcaster',
        arguments=['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', 'world', 'rgb_camera_link'],
        output='screen'
    )
    mediapipe_share_dir = get_package_share_directory('mediapipe_ros')
    
    # Define the pose estimation node
    pose_estimation_node = Node(
        package='mediapipe_ros',
        executable='pose_estimation_node',
        name='pose_estimation_node',
        output='screen',
        parameters=[{
            'enable': enable,
            'rgb_image_reliability': rgb_reliability,
            'depth_image_reliability': depth_reliability,
            'depth_info_reliability': depth_info_reliability,
            'rgb_topic': rgb_topic,
            'depth_topic': depth_topic,
            'depth_info_topic': depth_info_topic,
            'num_poses': 1,
            'min_pose_detection_confidence': 0.5,
            'min_pose_presence_confidence': 0.5,
            'min_tracking_confidence': 0.5
        }]
    )
    
    # Return the LaunchDescription
    return LaunchDescription([
        # Launch arguments
        enable_arg,
        rgb_reliability_arg,
        depth_reliability_arg,
        depth_info_reliability_arg,
        rgb_topic_arg,
        depth_topic_arg,
        depth_info_topic_arg,
        
        # Nodes
        static_transform_publisher,
        pose_estimation_node
    ])
