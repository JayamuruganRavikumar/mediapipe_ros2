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

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from ament_index_python.packages import get_package_share_directory

import tf2_ros
from geometry_msgs.msg import Point, TransformStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import Header, ColorRGBA


class PosePublisher(LifecycleNode):
    def __init__(self) -> None:
        super().__init__("pose_estimation_node")

        self.declare_parameter("rgb_image_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        self.declare_parameter("depth_image_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        self.declare_parameter("depth_info_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        self.declare_parameter("enable", True)
        self.source_frame = "rgb_camera_link"
        self.cv_bridge = CvBridge()
        self.depth_image = None
        self.rgb_image = None
        self.depth_info_msg = None
        self.get_logger().info("Pose estimattion Node created")

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:

        self.get_logger().info(f"Configuring {self.get_name()}")
        package_dir = get_package_share_directory('mediapipe_ros')
        # model parameters (need to change to ros params)
        self.model_path = os.path.join(package_dir,"model","pose_landmarker_full.task")
        self.num_poses = 1
        self.min_pose_detection_confidence = 0.5
        self.min_pose_presence_confidence = 0.5
        self.min_tracking_confidence = 0.5
        self.last_timestamp_ms = 0
        self.pose_name = (
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
        )
        self.base_options = python.BaseOptions(model_asset_path=self.model_path,
                                       delegate=python.BaseOptions.Delegate.GPU)
        self.options = vision.PoseLandmarkerOptions(
            base_options=self.base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_poses=self.num_poses,
            min_pose_detection_confidence=self.min_pose_detection_confidence,
            min_pose_presence_confidence=self.min_pose_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_segmentation_masks=False,
            result_callback=self.mediapipe_callback
        )
        self.enable = self.get_parameter(
            "enable").get_parameter_value().bool_value

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

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:

        self.get_logger().info(f"Activating {self.get_name()}")
        self.landmarker = vision.PoseLandmarker.create_from_options(self.options) 
        self.broadcaster = tf2_ros.TransformBroadcaster(self)
        self.get_logger().info("Pose object intialized")

        #suscribers
        self._rgb_sub = message_filters.Subscriber(self, 
            Image, "/mediapipe/rgb/image_raw", qos_profile=self.rgb_image_qos_profile
        )
        self.get_logger().info("rgb sub created")

        self._depth_sub = message_filters.Subscriber(self, 
            Image, "/mediapipe/depth_to_rgb/image_raw", qos_profile=self.depth_image_qos_profile
        )
        self.get_logger().info("depth sub created")

        self._depth_info_sub = message_filters.Subscriber(self,
            CameraInfo, "/mediapipe/depth_to_rgb/camera_info", qos_profile=self.depth_info_qos_profile)
        self.get_logger().info("message sub created")

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
                (self._rgb_sub, self._depth_sub, self._depth_info_sub), 10, 0.5)
        self.get_logger().info("sync sub created")

        self._synchronizer.registerCallback(self.pose_cb)
        self.get_logger().info("sync callback created")

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:

        self.get_logger().info(f"Deactivating {self.get_name()}")
        
        del self.mp_pose

        self.destroy_subscription(self._rgb_sub)
        self._rgb_sub = None
        self._depth_sub = None
        self._depth_info_sub = None

        super().on_deactivate(state)

    def on_cleanup(self, state: LifecycleNode) -> TransitionCallbackReturn:

        self.get_logger().info(f"Cleaning {self.get_name()}")
        self.destroy_publisher(self._im_pub)
        self.destroy_publisher(self._pose_pub)
        del self.image_qos_profile

        return TransitionCallbackReturn.SUCCESS

    def pose_cb(self, rgb_msg: Image, depth_msg: Image, depth_info: CameraInfo):
        self.rgb_image_msg = rgb_msg
        self.depth_image_msg = depth_msg
        self.depth_info_msg = depth_info
        rgb_image_msg = self.cv_bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        rgb_image = cv2.cvtColor(cv2.flip(rgb_image_msg, 0), cv2.COLOR_BGR2RGB)

        mp_pose = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image)
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        self.landmarker.detect_async(mp_pose, timestamp_ms)

    def mediapipe_callback(self, detection_result: vision.PoseLandmarkerResult, output_image: mp.Image,
                     timestamp_ms: int):
        if timestamp_ms < self.last_timestamp_ms:
            return
        self.last_timestamp_ms = timestamp_ms

        if self.rgb_image_msg is None or self.depth_image_msg is None or self.depth_info_msg is None:
            self.get_logger().warn("No Synchronized images")

        self.process_results(detection_result, output_image, self.rgb_image_msg, self.depth_image_msg, self.depth_info_msg)

    def process_results(self, detection_result, output_image, rgb_msg, depth_msg, depth_info):
        depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        posearray_msg = PoseArray()
        keypoints, annotated_image = self.draw_landmarks_on_image(
            output_image.numpy_view(), 
            detection_result,
            depth_image,
            depth_info
        )
        posearray_msg.header = rgb_msg.header
        posearray_msg.pose = keypoints
        img_msg = self.cv_bridge.cv2_to_imgmsg(annotated_image, "bgr8")
        self._pose_pub.publish(posearray_msg)
        self._im_pub.publish(img_msg)


    def draw_landmarks_on_image(self, rgb_image, detection_result, depth_image, depth_info):
        annotated_image = np.copy(rgb_image)
        h,w,c = rgb_image.shape

        if not detection_result.pose_landmarks:
            return annotated_image

        # Here the num of poses = 1, can be modified for more than 1 person
        landmarks = detection_result.pose_landmarks[0]
        keypoint_list = []

        # Loop through the detected poses to visualize.
        for idx, pose_landmarks in enumerate(landmarks):
            if 15 <= idx < 23:
                if pose_landmarks:
                    keypoint_msg = Keypoint()
                    det_x, det_y = pose_landmarks.x*w, pose_landmarks.y*h
                    keypoint_msg.name = self.pose_name[idx]
                    keypoint_msg.point2d.x = det_x 
                    keypoint_msg.point2d.y = det_y

                    point3D = self.image_to_world(depth_image,
                                                      depth_info,
                                                      int(det_x),
                                                      int(det_y))

                    keypoint_msg.point3d.x = point3D["x"]
                    keypoint_msg.point3d.y = point3D["y"]
                    keypoint_msg.point3d.z = point3D["z"]
                else:
                    keypoint_msg = Keypoint()
                    keypoint_msg.name = self.pose_name[idx]
                    keypoint_msg.point2d = Point2D
                    keypoint_msg.point3d = Point

                self.transform(keypoint_msg)
                keypoint_list.append(keypoint_msg)

        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z) for landmark in landmarks
        ])
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style())

        return keypoint_list, annotated_image

    def image_to_world(self,depth_image_msg: np.ndarray,  depth_info: CameraInfo, center_x , center_y ) :
        depth_image = cv2.flip(depth_image_msg, 0)
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
        point_list = {"x": x, "y": y, "z": z }

        return point_list 

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
    node=PosePublisher()
    node.trigger_configure()
    node.trigger_activate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

