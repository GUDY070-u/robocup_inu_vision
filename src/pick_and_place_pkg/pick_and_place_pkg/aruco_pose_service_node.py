import rclpy
from rclpy.node import Node

import numpy as np
import cv2
import time
from collections import deque

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from msgs_pkg.srv import GetObjectPose


MARKER_SIZE = 0.03

STABLE_TIME_SEC = 1.0
THRESH_MM = 10.0
THRESH_RZ_DEG = 5.0


# ===============================
# Euler 기반 rz 계산 (Standalone 동일)
# ===============================
def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rz = 0.0

    return np.degrees(rz)


def wrap_to_180(angle):
    return (angle + 180) % 360 - 180


# ===============================
class ArucoPoseService(Node):

    def __init__(self):
        super().__init__("aruco_pose_service")

        self.get_logger().info("✅ ArUco Pose Service Started")

        self.bridge = CvBridge()

        # 최신 프레임 저장
        self.latest_image = None
        self.camera_ready = False

        self.camera_matrix = None
        self.dist_coeffs = None

        # Subscribers
        self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.image_callback,
            10
        )

        self.create_subscription(
            CameraInfo,
            "/camera/color/camera_info",
            self.camera_info_callback,
            10
        )

        # ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )
        self.params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(
            self.aruco_dict,
            self.params
        )

        half = MARKER_SIZE / 2.0
        self.obj_points = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0]
        ], dtype=np.float32)

        # 안정화용 pose buffer
        self.pose_hist = deque(maxlen=200)

        # ✅ Pose Service 서버
        self.srv = self.create_service(
            GetObjectPose,
            "/vision/get_object_pose",
            self.handle_pose_request
        )


    # ===============================
    def camera_info_callback(self, msg):

        if self.camera_ready:
            return

        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)
        self.camera_ready = True

        self.get_logger().info("✅ CameraInfo received")


    def image_callback(self, msg):

        self.latest_image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="bgr8"
        )


    # ===============================
    # Service 요청 처리
    # ===============================
    def handle_pose_request(self, request, response):

        self.get_logger().info("📡 Pose Request Received")

        if not self.camera_ready or self.latest_image is None:
            response.success = False
            return response

        self.pose_hist.clear()
        start = time.time()

        # ✅ 1초 동안 안정화 측정
        while time.time() - start < STABLE_TIME_SEC:

            img = self.latest_image.copy()
            corners, ids, _ = self.detector.detectMarkers(img)

            if ids is None:
                continue

            img_points = corners[0].reshape(4, 2)

            success, rvec, tvec = cv2.solvePnP(
                self.obj_points,
                img_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            if not success:
                continue

            # Pose 계산
            R, _ = cv2.Rodrigues(rvec)
            rz_deg = wrap_to_180(rotationMatrixToEulerAngles(R))

            x, y, z = tvec.flatten()

            self.pose_hist.append((x, y, z, rz_deg))

            time.sleep(0.05)

        if len(self.pose_hist) < 2:
            response.success = False
            return response

        # ✅ 평균값 반환
        poses = np.array(self.pose_hist)
        mean_pose = np.mean(poses, axis=0)

        response.x = float(mean_pose[0])
        response.y = float(mean_pose[1])
        response.z = float(mean_pose[2])
        response.rz = float(mean_pose[3])
        response.success = True

        self.get_logger().info(
            f"✅ Pose Response: x={response.x:.3f}, y={response.y:.3f}, rz={response.rz:.1f}"
        )

        return response


# ===============================
def main():
    rclpy.init()
    node = ArucoPoseService()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()