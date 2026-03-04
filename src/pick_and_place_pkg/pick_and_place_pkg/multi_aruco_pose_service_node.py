import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

import numpy as np
import cv2
import time
from collections import deque

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from msgs_pkg.srv import GetObjectPose

MARKER_SIZE = 0.03
STABLE_TIME_SEC = 1.0

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

class ArucoPoseService(Node):

    def __init__(self):
        super().__init__("aruco_pose_service")
        self.get_logger().info("✅ ArUco Pose Service Started (Target: Max X-axis)")
        
        self.bridge = CvBridge()
        self.callback_group = ReentrantCallbackGroup()

        self.latest_image = None
        self.camera_ready = False
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # ArUco 설정
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.params)

        half = MARKER_SIZE / 2.0
        self.obj_points = np.array([
            [-half,  half, 0], [ half,  half, 0],
            [ half, -half, 0], [-half, -half, 0]
        ], dtype=np.float32)

        self.pose_hist = deque(maxlen=200)

        # Subscribers
        self.create_subscription(Image, "/camera/camera/color/image_raw", self.image_callback, 10, callback_group=self.callback_group)
        self.create_subscription(CameraInfo, "/camera/camera/color/camera_info", self.camera_info_callback, 10, callback_group=self.callback_group)

        # Service
        self.srv = self.create_service(GetObjectPose, "/vision/get_object_pose", self.handle_pose_request, callback_group=self.callback_group)

    def camera_info_callback(self, msg):
        if self.camera_ready: return
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)
        self.camera_ready = True
        self.get_logger().info("✅ CameraInfo received")

    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Image decode error: {e}")

    # ========================================================
    # 🔍 핵심 로직: 현재 화면에서 X값이 가장 큰 마커 ID 찾기
    # ========================================================
    def find_target_with_max_x(self):
        """
        현재 이미지에서 모든 마커를 감지하고,
        카메라 좌표계 기준 X값(tvec[0])이 가장 큰 마커의 ID를 반환합니다.
        """
        if self.latest_image is None: return None

        img = self.latest_image.copy()
        corners, ids, _ = self.detector.detectMarkers(img)

        if ids is None: return None

        max_x_val = -99999.0  # 아주 작은 값으로 초기화
        target_id = None

        # 감지된 모든 마커를 순회하며 X값 비교
        for i, marker_id in enumerate(ids.flatten()):
            # PnP 계산
            img_points = corners[i].reshape(4, 2)
            success, _, tvec = cv2.solvePnP(
                self.obj_points, img_points, self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            
            if success:
                current_x = tvec[0][0] # tvec은 (3,1) 벡터이므로 [0][0]이 x값
                if current_x > max_x_val:
                    max_x_val = current_x
                    target_id = int(marker_id)

        return target_id

    # ========================================================
    # 서비스 핸들러
    # ========================================================
    def handle_pose_request(self, request, response):
        self.get_logger().info("📡 Pose Request Received (Auto-Selection Mode)")

        # 1. 카메라 준비 상태 확인
        if not self.camera_ready or self.latest_image is None:
            time.sleep(1.0) # 혹시 시작 직후라면 1초 대기
            if self.latest_image is None:
                self.get_logger().warn("No image received")
                response.success = False
                return response

        # 2. 🎯 타겟 선정 (X축이 가장 큰 마커)
        # 안정화를 시작하기 전에, "누구를 잡을지" 먼저 결정합니다.
        # 루프 안에서 매번 결정하면 노이즈 때문에 타겟이 왔다갔다 할 수 있으므로, 처음에 'Lock'을 겁니다.
        target_id = self.find_target_with_max_x()

        if target_id is None:
            self.get_logger().warn("❌ No markers found in the scene.")
            response.success = False
            return response
        
        self.get_logger().info(f"🎯 Target Locked! ID: {target_id} (Has Max X value)")

        # 3. 데이터 수집 및 안정화 (선정된 target_id만 추적)
        self.pose_hist.clear()
        start = time.time()
        
        while time.time() - start < STABLE_TIME_SEC:
            if self.latest_image is None: continue
            img = self.latest_image.copy()

            corners, ids, _ = self.detector.detectMarkers(img)
            if ids is None: continue

            # 선정된 ID가 현재 프레임에 있는지 확인
            flat_ids = ids.flatten()
            if target_id not in flat_ids:
                continue # 이번 프레임엔 타겟이 안 보임 (잠시 가려짐 등)

            # 타겟 ID의 인덱스 찾기
            target_index = np.where(flat_ids == target_id)[0][0]
            img_points = corners[target_index].reshape(4, 2)

            success_pnp, rvec, tvec = cv2.solvePnP(
                self.obj_points, img_points, self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            if success_pnp:
                R, _ = cv2.Rodrigues(rvec)
                rz_deg = wrap_to_180(rotationMatrixToEulerAngles(R))
                x, y, z = tvec.flatten()
                self.pose_hist.append((x, y, z, rz_deg))
            
            time.sleep(0.05)

        # 4. 결과 반환
        if len(self.pose_hist) < 5: # 최소 5 프레임 이상 확보 확인
            self.get_logger().warn(f"Failed to stabilize pose for ID {target_id}")
            response.success = False
            return response

        poses = np.array(self.pose_hist)
        mean_pose = np.mean(poses, axis=0)

        response.x = float(mean_pose[0])
        response.y = float(mean_pose[1])
        response.z = float(mean_pose[2])
        response.rz = float(mean_pose[3])
        response.success = True
        
        # 만약 서비스 메시지에 id 필드가 있다면 채워줄 수도 있음 (선택사항)
        response.detected_id = int(target_id)

        self.get_logger().info(
            f"✅ Success (ID {target_id}): x={response.x:.3f}, y={response.y:.3f}, z={response.z:.3f}"
        )
        return response

def main():
    rclpy.init()
    node = ArucoPoseService()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()