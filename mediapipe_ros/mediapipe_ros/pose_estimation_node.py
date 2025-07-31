import rclpy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState

import message_filters
import cv2
import os
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from mediapipe_msg.msg import Point2D, Keypoint, PoseArray
from geometry_msgs.msg import Point

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from ament_index_python.packages import get_package_share_directory

import tf2_ros
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import Header, ColorRGBA



class PosePublisher(LifecycleNode):
    def __init__(self) -> None:

        super().__init__("pose_estimation_node")

        self.declare_parameter("rgb_image_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        self.declare_parameter("depth_image_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        self.declare_parameter("depth_info_reliability", QoSReliabilityPolicy.BEST_EFFORT)

        self.declare_parameter("rgb_topic", "/rgb/image_raw")
        self.declare_parameter("depth_topic", "/depth_to_rgb/image_raw")
        self.declare_parameter("depth_info_topic", "/depth_to_rgb/camera_info")

        self.declare_parameter("min_detection_confidence", 0.5)
        self.declare_parameter("min_tracking_confidence", 0.5)
        self.declare_parameter("min_pose_presence_confidence", 0.5)
        self.declare_parameter("num_poses", 1)
        self.declare_parameter("use_gpu", True)

        self.declare_parameter("enable", True)
        self.source_frame = "rgb_camera_link"
        self.cv_bridge = CvBridge()
        self.get_logger().info("Pose estimation Node created")

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:

        try:
            self.get_logger().info(f"Configuring {self.get_name()}")
            self.enable = self.get_parameter(
                "enable").get_parameter_value().bool_value

            self.rgb_topic = self.get_parameter("rgb_topic").get_parameter_value().string_value
            self.depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
            self.depth_info_topic = self.get_parameter("depth_info_topic").get_parameter_value().string_value

            self.min_detection_conf = self.get_parameter("min_detection_confidence").get_parameter_value().double_value
            self.min_tracking_conf = self.get_parameter("min_tracking_confidence").get_parameter_value().double_value
            self.min_pose_presence_conf = self.get_parameter("min_pose_presence_confidence").get_parameter_value().double_value
            self.num_poses = self.get_parameter("num_poses").get_parameter_value().integer_value
            self.use_gpu = self.get_parameter("use_gpu").get_parameter_value().bool_value
            
            # Setup model path
            package_dir = get_package_share_directory('mediapipe_ros')
            self.model_path = os.path.join(package_dir, "model", "pose_landmarker_full.task")
            
            # Check if model file exists
            if not os.path.exists(self.model_path):
                self.get_logger().error(f"Model file not found: {self.model_path}")
                return TransitionCallbackReturn.FAILURE

            rgb_reliability = self.get_parameter("rgb_image_reliability").get_parameter_value().integer_value
            self.rgb_image_qos_profile = QoSProfile(reliability=rgb_reliability,
                                                history=QoSHistoryPolicy.KEEP_LAST,
                                                durability=QoSDurabilityPolicy.VOLATILE,
                                                depth=1)
            depth_reliability = self.get_parameter("depth_image_reliability").get_parameter_value().integer_value
            self.depth_image_qos_profile = QoSProfile(reliability=depth_reliability,
                                                history=QoSHistoryPolicy.KEEP_LAST,
                                                durability=QoSDurabilityPolicy.VOLATILE,
                                                depth=1)
            depth_info_reliability = self.get_parameter("depth_info_reliability").get_parameter_value().integer_value
            self.depth_info_qos_profile = QoSProfile(reliability=depth_info_reliability,
                                                history=QoSHistoryPolicy.KEEP_LAST,
                                                durability=QoSDurabilityPolicy.VOLATILE,
                                                depth=1)

            #publishers
            self._pose_pub = self.create_publisher(PoseArray, "pose_list", 10)
            self._im_pub = self.create_publisher(Image, "processed", 10)

            return TransitionCallbackReturn.SUCCESS
            
        except Exception as e:
            self.get_logger().error(f"Configuration failed: {e}")
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:

        self.get_logger().info(f"Activating {self.get_name()}")
        self.pose_name = [
            "NOSE", "LEFT_EYE_INNER",
            "LEFT_EYE", "LEFT_EYE_OUTER",
            "RIGHT_EYE_INNER", "RIGHT_EYE",
            "RIGHT_EYE_OUTER", "LEFT_EAR",
            "RIGHT_EAR", "MOUTH_LEFT",
            "MOUTH_RIGHT",  "LEFT_SHOULDER",
            "RIGHT_SHOULDER","LEFT_ELBOW",
            "RIGHT_ELBOW", "LEFT_WRIST",
            "RIGHT_WRIST", "LEFT_PINKY",
            "RIGHT_PINKY", "LEFT_INDEX",
            "RIGHT_INDEX", "LEFT_THUMB",
            "RIGHT_THUMB", "LEFT_HIP",
            "RIGHT_HIP",  "LEFT_KNEE",
            "RIGHT_KNEE", "LEFT_ANKLE",
            "RIGHT_ANKLE", "LEFT_HEEL",
            "RIGHT_HEEL", "LEFT_FOOT_INDEX",
            "RIGHT_FOOT_INDEX"
        ]
        
        # Setup MediaPipe pose landmarker with GPU/CPU selection
        delegate = python.BaseOptions.Delegate.GPU if self.use_gpu else python.BaseOptions.Delegate.CPU
        base_options = python.BaseOptions(model_asset_path=self.model_path, delegate=delegate)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=self.num_poses,
            min_pose_detection_confidence=self.min_detection_conf,
            min_pose_presence_confidence=self.min_pose_presence_conf,
            min_tracking_confidence=self.min_tracking_conf,
            output_segmentation_masks=False
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.broadcaster = tf2_ros.TransformBroadcaster(self)
        self.get_logger().info("Pose object intialized")

        #suscribers
        try:
            self._rgb_sub = message_filters.Subscriber(self, 
                Image, self.rgb_topic, qos_profile=self.rgb_image_qos_profile
            )
            self.get_logger().info("rgb sub created")

            self._depth_sub = message_filters.Subscriber(self, 
                Image, self.depth_topic, qos_profile=self.depth_image_qos_profile
            )
            self.get_logger().info("depth sub created")

            self._depth_info_sub = message_filters.Subscriber(self,
                CameraInfo, self.depth_info_topic, qos_profile=self.depth_info_qos_profile)
            self.get_logger().info("depth info sub created")

            self._synchronizer = message_filters.ApproximateTimeSynchronizer(
                    (self._rgb_sub, self._depth_sub, self._depth_info_sub), 10, 0.5)
            self.get_logger().info("sync sub created")

            self._synchronizer.registerCallback(self.pose_cb)
            self.get_logger().info("sync callback created")
        except Exception as e:
            self.get_logger().error(f"Failed to create subscribers: {e}")
            return TransitionCallbackReturn.FAILURE

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:

        self.get_logger().info(f"Deactivating {self.get_name()}")
        
        if hasattr(self, 'landmarker') and self.landmarker is not None:
            self.landmarker.close()
            del self.landmarker
        if hasattr(self, '_rgb_sub'):
            del self._rgb_sub
        if hasattr(self, '_depth_sub'):
            del self._depth_sub
        if hasattr(self, '_depth_info_sub'):
            del self._depth_info_sub

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:

        self.get_logger().info(f"Cleaning {self.get_name()}")
        self.destroy_publisher(self._im_pub)
        self.destroy_publisher(self._pose_pub)
        del self.rgb_image_qos_profile
        del self.depth_image_qos_profile
        del self.depth_info_qos_profile

        return TransitionCallbackReturn.SUCCESS

    #Pose estimation callback 
    def pose_cb(self, rgb_msg: Image, depth_msg: Image, depth_info: CameraInfo) :

        posearray_msg = PoseArray()
        rgb_image_msg = self.cv_bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth_image_msg = self.cv_bridge.imgmsg_to_cv2(depth_msg, "16UC1")

        #flip image (camera mounted upside down)
        rgb_image = cv2.cvtColor(cv2.flip(rgb_image_msg, 0), cv2.COLOR_BGR2RGB)

        # Convert to MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Detect pose landmarks
        results = self.landmarker.detect(mp_image)
        
        # Draw the pose annotation on the image
        image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        h, w, c = image.shape

        if results.pose_landmarks:
            keypoints = self.process_results(results, depth_image_msg, depth_info, h, w) 
            posearray_msg.header = rgb_msg.header 
            posearray_msg.pose = keypoints
            
            # Draw landmarks on the image
            self.draw_landmarks_on_image(image, results)

        self._pose_pub.publish(posearray_msg)
        img_msg = self.cv_bridge.cv2_to_imgmsg(image, "bgr8")
        self._im_pub.publish(img_msg)
                
    def process_results(self, results, depth_image_msg, depth_info, h, w):
        keypoint_list = []
        # Use the first pose (assuming num_poses=1)
        landmarks = results.pose_landmarks[0]
        for ids, pose_landmarks in enumerate(landmarks):
            #check for wrist
            if 15 <= ids < 23:
                keypoint_msg = Keypoint()
                keypoint_msg.name = self.pose_name[ids]
                
                if pose_landmarks:
                    det_x, det_y = pose_landmarks.x*w, pose_landmarks.y*h
                    keypoint_msg.point2d.x = det_x
                    keypoint_msg.point2d.y = det_y
                    point = self.image_to_world(depth_image_msg, depth_info, int(det_x), int(det_y))
                    keypoint_msg.point3d.x = point["x"]
                    keypoint_msg.point3d.y = point["y"]
                    keypoint_msg.point3d.z = point["z"]
                else:
                    keypoint_msg.point2d = Point2D()
                    keypoint_msg.point3d = Point()

                self.transform(keypoint_msg)
                keypoint_list.append(keypoint_msg)

        return  keypoint_list

    def draw_landmarks_on_image(self, image, results):
        """Draw pose landmarks on the image using the new MediaPipe API"""
        if not results.pose_landmarks:
            return
            
        # Convert landmarks to the format expected by the drawing utilities
        landmarks = results.pose_landmarks[0]
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z) for landmark in landmarks
        ])
        
        # Draw landmarks
        mp.solutions.drawing_utils.draw_landmarks(
            image,
            pose_landmarks_proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style())

    def image_to_world(self,depth_image_msg: np.ndarray,  depth_info: CameraInfo, center_x , center_y ) :
        depth_image = cv2.flip(depth_image_msg, 0)
        
        # Check bounds to prevent crashes
        if center_y < 0 or center_y >= depth_image.shape[0] or center_x < 0 or center_x >= depth_image.shape[1]:
            return {"x": 0.0, "y": 0.0, "z": 0.0}
            
        z = float(depth_image[center_y, center_x])
        #Get camera intrinsic parameters
        # Camera matrix K = [fx 0 cx]
        #                   [0 fy cy]
        #                   [0 0  1 ]

        # project image to camera cordinates
        k = depth_info.k
        cx, cy, fx, fy = k[2], k[5], k[0], k[4]
        x = z * (center_x - cx) / fx
        y = z * (center_y - cy) / fy
        point_dict = {"x": x, "y": y, "z": z }

        return point_dict 

    def transform(self, keypoints: Keypoint):

        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.source_frame
        transform.child_frame_id = keypoints.name

        transform.transform.translation.x = keypoints.point3d.x/1000
        transform.transform.translation.y = keypoints.point3d.y/1000
        transform.transform.translation.z = keypoints.point3d.z/1000
        transform.transform.rotation.w = 1.0
        self.broadcaster.sendTransform(transform)




def main():
    rclpy.init()
    node = PosePublisher()
    
    # Configure the node
    if node.trigger_configure() != TransitionCallbackReturn.SUCCESS:
        node.get_logger().error("Failed to configure node")
        node.destroy_node()
        rclpy.shutdown()
        return
    
    # Activate the node
    if node.trigger_activate() != TransitionCallbackReturn.SUCCESS:
        node.get_logger().error("Failed to activate node")
        node.destroy_node()
        rclpy.shutdown()
        return
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.trigger_deactivate()
        node.trigger_cleanup()
        node.destroy_node()
        rclpy.shutdown()

