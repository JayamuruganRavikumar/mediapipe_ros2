import rclpy
import cv2
import mediapipe as mp
import numpy as np
from rclpy.node import Node
from media_pipe_ros2_msg.msg import  MediaPipeHumanPoseList                            
from mediapipe.python.solutions.pose import PoseLandmark
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
NAME_POSE = [
    (PoseLandmark.NOSE), (PoseLandmark.LEFT_EYE_INNER),
    (PoseLandmark.LEFT_EYE), (PoseLandmark.LEFT_EYE_OUTER),
    (PoseLandmark.RIGHT_EYE_INNER), ( PoseLandmark.RIGHT_EYE),
    (PoseLandmark.RIGHT_EYE_OUTER), ( PoseLandmark.LEFT_EAR),
    (PoseLandmark.RIGHT_EAR), ( PoseLandmark.MOUTH_LEFT),
    (PoseLandmark.MOUTH_RIGHT), ( PoseLandmark.LEFT_SHOULDER),
    (PoseLandmark.RIGHT_SHOULDER), ( PoseLandmark.LEFT_ELBOW),
    (PoseLandmark.RIGHT_ELBOW), ( PoseLandmark.LEFT_WRIST),
    (PoseLandmark.RIGHT_WRIST), ( PoseLandmark.LEFT_PINKY),
    (PoseLandmark.RIGHT_PINKY), ( PoseLandmark.LEFT_INDEX),
    (PoseLandmark.RIGHT_INDEX), ( PoseLandmark.LEFT_THUMB),
    (PoseLandmark.RIGHT_THUMB), ( PoseLandmark.LEFT_HIP),
    (PoseLandmark.RIGHT_HIP), ( PoseLandmark.LEFT_KNEE),
    (PoseLandmark.RIGHT_KNEE), ( PoseLandmark.LEFT_ANKLE),
    (PoseLandmark.RIGHT_ANKLE), ( PoseLandmark.LEFT_HEEL),
    (PoseLandmark.RIGHT_HEEL), ( PoseLandmark.LEFT_FOOT_INDEX),
    (PoseLandmark.RIGHT_FOOT_INDEX)
]

class PosePublisher(Node):

    def __init__(self):
        super().__init__('mediapipe_pose_publisher')
        self.publisher_ = self.create_publisher(MediaPipeHumanPoseList, '/mediapipe/human_pose_list', 10)
        self.image_publisher=self.create_publisher(Image, '/processed/image', 10)
        self.subscription = self.create_subscription(Image, '/rgb_to_depth/image_raw', self.getrgb_callback, 10)
        self.subscription = self.create_subscription(Image, '/depth/image_raw', self.getdepth_callback, 10)
        self.bridge = CvBridge()


    def compare_depth(self):
        print("called")

        mediapipehumanposelist = MediaPipeHumanPoseList() 
        
        with mp_pose.Pose(
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5) as pose:
            results = pose.process(self.rgb)

            # Draw the pose annotation on the image.
            self.rgb.flags.writeable = True
            image = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks( image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            
            if results.pose_landmarks != None:
                ids = 0 
                #for ids, pose_landmarks enumerate(results.pose_landmarks.landmark):
                for pose_landmarks in(results.pose_landmarks.landmark):
                    #if 16 <= ids < 23:
                    mediapipehumanposelist.human_pose_list[ids].name = str(NAME_POSE[ids])
                    mediapipehumanposelist.human_pose_list[ids].x = pose_landmarks.x
                    mediapipehumanposelist.human_pose_list[ids].y = pose_landmarks.y
                    mediapipehumanposelist.human_pose_list[ids].visibility = pose_landmarks.visibility
                    ids = ids+1

                mediapipehumanposelist.num_humans = 1
                self.publisher_.publish(mediapipehumanposelist)
            else: 
                ids = 0
                for point in mp_pose.PoseLandmark:                          
                                                                                      
                    mediapipehumanposelist.human_pose_list[ids].name = str(NAME_POSE[ids])
                    mediapipehumanposelist.human_pose_list[ids].x = 0.0
                    mediapipehumanposelist.human_pose_list[ids].y = 0.0
                    mediapipehumanposelist.human_pose_list[ids].visibility = 0.0
                    ids = ids+1
            
                mediapipehumanposelist.num_humans = 1
                self.publisher_.publish(mediapipehumanposelist)

            img_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
            self.image_publisher.publish(img_msg)
        
    def getrgb_callback(self, msg):
        image_msg = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.rgb = cv2.cvtColor(cv2.flip(image_msg, 0), cv2.COLOR_BGR2RGB)                

    def getdepth_callback(self, msg):
        #conver form 32FC1 to np array
        self.depth = np.frombuffer(msg.data, dtype=np.float32).reshape(512, msg.width)                 
        self.compare_depth()


def main(args=None):

    rclpy.init(args=args)
    pose_publisher=PosePublisher()
    rclpy.spin(pose_publisher)
    pose_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
