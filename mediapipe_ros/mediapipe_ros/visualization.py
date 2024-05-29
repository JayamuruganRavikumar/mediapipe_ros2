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
        self.marker_pub = self.create_publisher(Marker, '/mediapipe/points_marker3D', 10)
        self.subscription=self.create_subscription(CameraInfo, '/depth_to_rgb/camera_info', self.getcamerainfo_callback, 10)
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
        marker_points=[]
        marker_names=[]


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
            marker_points.append(Point(x=float((val.x - cx) * depth / fx), y=float((val.y - cy) * depth / fy), z=float(depth)))
            marker_names.append(str(val.name))

        print(marker_points)

        self.point_array_pub.publish(points)
        self.point_marker(marker_points,marker_names)

    def point_marker(self, points, names):

        marker = Marker()
        marker.header = Header()
        marker.header.frame_id = "rgb_camera_link"  # Set the frame ID 
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "points"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0

        # Set the scale of the points
        marker.scale.x = 0.1  # Width of the points
        marker.scale.y = 0.1  # Height of the points

        # Set the color
        marker.color = ColorRGBA()
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0  # 1 for fully opaque
        marker.points = points

        self.marker_pub.publish(marker)
        self.get_logger().info('Publishing PointArray and Marker')

        for i, point in enumerate(points):
            text_marker = Marker()
            text_marker.header = Header()
            text_marker.header.frame_id = "rgb_camera_link"  # Set the frame ID
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.ns = names[i]
            text_marker.id = i + 1
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = point.x 
            text_marker.pose.position.y = point.y 
            text_marker.pose.position.z = point.z 
            text_marker.pose.position.z += 0.2  # Slightly above the point
            text_marker.pose.orientation.w = 1.0

            # Set the scale of the text
            text_marker.scale.z = 0.2  # Height of the text

            # Set the color
            text_marker.color = ColorRGBA()
            text_marker.color.r = 0.0
            text_marker.color.g = 1.0
            text_marker.color.b = 0.0
            text_marker.color.a = 1.0  # 1 for fully opaque

            text_marker.text = f"Point {i + 1}"

            self.marker_pub.publish(text_marker)
        self.get_logger().info('Publishing Label and Marker')

def main(args=None):

    rclpy.init(args=args)
    visualization = Visualization()
    rclpy.spin(visualization)
    visualization.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
