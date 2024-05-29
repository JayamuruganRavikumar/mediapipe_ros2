import rclpy
import cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from std_msgs.msg import Header, ColorRGBA
from mediapipe_msg.msg import VisualPose, PoseList
from cv_bridge import CvBridge

class Visualization(Node):
    def __init__(self):
        super().__init__('visualization')
        self.point_array_pub = self.create_publisher(VisualPose, '/mediapipe/points3D', 10)
        self.subscription=self.create_subscription(CameraInfo, '/rgb/camera_info', self.getcamerainfo_callback, 10)
        self.subscription=self.create_subscription(PoseList, 'mediapipe/pose_list', self.pixel_to_3d, 10)
        self.bridge = CvBridge()
        self.caminfo = None


    def getcamerainfo_callback(self, msg):

        #self.get_logger().info("Received Cam info")
        try:
            self.caminfo = msg
        except Exception as e:
            self.getlogget().info(f"Camera Info error{e}")


    def pixel_to_3d(self, msg):

        if self.caminfo is None:
            return

        points = VisualPose()

        #Get camera intrinsic parameters
        # Camera matrix K = [fx 0 cx]
        #                   [0 fy cy]
        #                   [0 0  1 ]

        fx = self.caminfo.k[0]
        fy = self.caminfo.k[4]
        cx = self.caminfo.k[2]
        cy = self.caminfo.k[5]

        for i, val in enumerate(msg.human_pose):
            depth = val.z
            print(depth)
            points.act_position[i].name =val.name
            points.act_position[i].x = float((val.x - cx) * depth / fx)
            points.act_position[i].y = float((val.y - cy) * depth / fy)
            points.act_position[i].z = float(depth)

        print(marker_points)

        self.point_array_pub.publish(points)


def main(args=None):

    rclpy.init(args=args)
    visualization = Visualization()
    rclpy.spin(visualization)
    visualization.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
