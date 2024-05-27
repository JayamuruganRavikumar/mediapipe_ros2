import rclpy
import cv2
import mediapipe as mp
import numpy as np
from rclpy.node import Node
from mediapipe_msg.msg import PoseList
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
        self.publisher_ = self.create_publisher(PoseList, '/mediapipe/pose_list', 10)
        self.image_publisher=self.create_publisher(Image, '/processed/image', 10)
        self.subscription = self.create_subscription(Image, '/rgb/image_raw', self.getrgb_callback, 10)
        self.subscription = self.create_subscription(Image, '/depth_to_rgb/image_raw', self.getdepth_callback, 10)
        self.bridge = CvBridge()

    #callback function for depth camera    
    def getdepth_callback(self, msg):
        try:
            #conver form 32FC1 to np array
            #depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height,msg.width)                 
            depth = self.bridge.imgmsg_to_cv2(msg, "16UC1") 
            self.depth = depth[::-1,:]
            if hasattr(self, 'rgb'):
                self.compare_depth(self.rgb,self.depth)
        except CvBridgeError as e:
            self.get_logger().error(f"Error converting from depth camera: {e}")

    #callback function for rgb camera
    def getrgb_callback(self, msg):
        try:
            image_msg = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.rgb = cv2.cvtColor(cv2.flip(image_msg, 0), cv2.COLOR_BGR2RGB)                
            if hasattr(self, 'depth'):
                self.compare_depth(self.rgb,self.depth)
        except CvBridgeError as e:
            self.get_logger().error(f"Error converting from rgb camera: {e}")

    #compare depth and rgb image
    def compare_depth(self, image, depth):

        poselist = PoseList() 
        
        with mp_pose.Pose(
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5) as pose:
            image.flags.writeable = False
            results = pose.process(image)
            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks( image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            h, w, c = image.shape

            if results.pose_landmarks != None:
                index = 0
                for ids, pose_landmarks in enumerate(results.pose_landmarks.landmark):
                    #check for wrist
                    if 15 <= ids < 23:
                        cx,cy = int(pose_landmarks.x*w), int(pose_landmarks.y*h)
                        poselist.human_pose[index].name = str(NAME_POSE[ids])
                        poselist.human_pose[index].x = cx
                        poselist.human_pose[index].y = cy
                        poselist.human_pose[index].z = int(depth[cy,cx])
                        index+=1
                self.publisher_.publish(poselist)

            else: 
                index = 0
                for point in mp_pose.PoseLandmark:                          
                    if 15 <= ids < 23:
                        poselist.human_pose[ids].name = str(NAME_POSE[ids])
                        poselist.human_pose[ids].x = 0
                        poselist.human_pose[ids].y = 0
                        poselist.human_pose[ids].z = 0 
                        index +=1
                self.publisher_.publish(poselist)

            img_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
            self.image_publisher.publish(img_msg)

def main(args=None):

    rclpy.init(args=args)
    pose_publisher=PosePublisher()
    rclpy.spin(pose_publisher)
    pose_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

