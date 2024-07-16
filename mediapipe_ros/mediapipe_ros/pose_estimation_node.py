
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
import numpy as np

from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from mediapipe_msg.msg import PoseList

import mediapipe as mp
from mediapipe.python.solutions.pose import PoseLandmark



class PosePublisher(LifecycleNode):
    def __init__(self) -> None:

        super().__init__("pose_estimation_node")

        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        self.enable = self.get_parameter("enable").get_parameter_value().bool_value
        self.model_path = "model/pose_landmarker_full.task"
        self.bridge = CvBridge()
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
        self.get_logger().info("Pose estimattion Node created")

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:

        self.get_logger().info(f"Configuring {self.get_name()}")
        self.enable = self.get_parameter(
            "enable").get_parameter_value().bool_value
        self.reliability = self.get_parameter("image_reliability").get_parameter_value().integer_value

        self.image_qos_profile = QoSProfile(reliability=self.reliability,
                                            history=QoSHistoryPolicy.KEEP_LAST,
                                            durability=QoSDurabilityPolicy.VOLATILE,
                                            depth=1)

        #publishers
        self._pose_pub = self.create_lifecycle_publisher(PoseList, "pose_list", 10)
        self._im_pub = self.create_lifecycle_publisher(Image, "processed", 10)
        self.cv_bridge = CvBridge()

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:

        self.get_logger().info(f"Activating {self.get_name()}")
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

        #suscribers
        self._rgb_sub = self.create_subscription(
            Image,
            "image_raw",
            self.rgb_cb,
            self.image_qos_profile
        )
        super().on_activate(state)

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:

        self.get_logger().info(f"Deactivating {self.get_name()}")
        
        del self.mp_pose

        self.destroy_subscription(self._rgb_sub)
        self._rgb_sub = None

        super().on_deactivate(state)

    def on_cleanup(self, state: LifecycleNode) -> TransitionCallbackReturn:

        self.get_logger().info(f"Cleaning {self.get_name()}")
        self.destroy_publisher(self._im_pub)
        self.destroy_publisher(self._pose_pub)
        del self.image_qos_profile

        return TransitionCallbackReturn.SUCCESS

    #rgb callback
    def rgb_cb(self, msg: image) -> None:

        poselist = PoseList() 
        image_msg = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.rgb = cv2.cvtColor(cv2.flip(image_msg, 0), cv2.COLOR_BGR2RGB)                
        
        with self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:
                image.flags.writeable = False
                results = pose.process(image)
                # Draw the pose annotation on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                self.mp_drawing.draw_landmarks( image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                h, w, c = image.shape

                if results.pose_landmarks != None:
                    index = 0
                    for ids, pose_landmarks in enumerate(results.pose_landmarks.landmark):
                        #check for wrist
                        if 15 <= ids < 23:
                            cx,cy = pose_landmarks.x*w, pose_landmarks.y*h
                            poselist.human_pose[index].name = self.pose_name[ids]
                            poselist.human_pose[index].x = cx
                            poselist.human_pose[index].y = cy
                            #poselist.human_pose[index].z = float(depth[int(cy),int(cx)])
                            index+=1
                    self._pose_pub.publish(poselist)

                else: 
                    index = 0
                    for ids, pose_landmarks in enumerate(results.pose_landmarks.landmark):
                        if 15 <= ids < 23:
                            poselist.human_pose[index].name = self.pose_name[ids]
                            poselist.human_pose[index].x = 0.0
                            poselist.human_pose[index].y = 0.0
                            #poselist.human_pose[index].z = 0.0
                            index+=1
                    self._pose_pub.publish(poselist)

                img_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
                self._im_pub.publish(img_msg)

def main(args=None):

    rclpy.init(args=args)
    node=PosePublisher()
    node.trigger_configure()
    node.trigger_activate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

