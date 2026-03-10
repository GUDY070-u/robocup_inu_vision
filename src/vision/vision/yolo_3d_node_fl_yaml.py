#!/usr/bin/env python3
import rclpy
import time
import os
import math
import numpy as np
import cv2
import open3d as o3d
import yaml
from collections import deque

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters
from ultralytics import YOLO

from msgs_pkg.srv import GetObjectPose

class Yolo3DNode(Node):
    def __init__(self):
        super().__init__('yolo_3d_node')

        # --- [변경] 경로 및 설정 관리 ---
        # 1. YAML 파일 경로 직접 지정
        self.yaml_path = '/home/user/ros2_ws/src/vision/config/models.yaml'
        
        # 2. YOLO 및 기본 파라미터 설정
        self.declare_parameter('model_path', '/home/user/ros2_ws/src/vision/yolo_models/0128_train/weights/best.pt')
        self.declare_parameter('model_device', 'cuda:0')
        self.declare_parameter('ground_dist_thresh', 0.01)
        self.declare_parameter('window_size', 15)
        self.declare_parameter('freshness_timeout', 0.5)

        model_path = self.get_parameter('model_path').value
        self.device = self.get_parameter('model_device').value
        self.ground_dist_thresh = self.get_parameter('ground_dist_thresh').value
        self.window_size = self.get_parameter('window_size').value
        self.timeout = self.get_parameter('freshness_timeout').value

        # 3. [핵심 변경] YAML로부터 물체 데이터베이스 구축
        self.obj_db = {}
        self.load_models_from_yaml()

        # 4. YOLO 모델 로드
        self.model = YOLO(model_path)

        # 5. 상태 관리 및 필터 변수
        self.pos_queues = {k: deque(maxlen=self.window_size) for k in ['x', 'y', 'z']}
        self.angle_queues = {k: {'sin': deque(maxlen=self.window_size), 'cos': deque(maxlen=self.window_size)} for k in ['roll', 'pitch', 'yaw']}
        self.prev_angles = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        self.intrinsics = None
        
        # 최신 결과 저장용 (타임스탬프 포함)
        self.latest_result = {"success": False, "timestamp": 0.0}

        # 6. ROS 통신 설정 (RealSense 카메라 토픽 기준)
        self.cv_bridge = CvBridge()
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.color_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_raw', qos_profile=qos)
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw', qos_profile=qos)
        self.sync = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=10, slop=0.5)
        self.sync.registerCallback(self.process_callback)
        
        self.info_sub = self.create_subscription(CameraInfo, '/camera/camera/aligned_depth_to_color/camera_info', self.info_callback, 10)
        self.srv = self.create_service(GetObjectPose, '/vision/get_object_pose', self.handle_get_pose)

        self.get_logger().info(f"YOLO 3D Node Started. YAML Loaded: {len(self.obj_db)} objects.")

    # --- [신규 함수] YAML 로드 로직 ---
    def load_models_from_yaml(self):
        """지정된 YAML 파일을 읽어 각 물체별 설정과 STL 모델을 메모리에 올립니다."""
        if not os.path.exists(self.yaml_path):
            self.get_logger().error(f"YAML 파일을 찾을 수 없습니다: {self.yaml_path}")
            return

        try:
            with open(self.yaml_path, 'r') as f:
                config = yaml.safe_load(f)
                
            for item in config['objects']:
                obj_id = int(item['id'])
                # 각 ID별로 파라미터와 미리 로드된 PCD 저장
                self.obj_db[obj_id] = {
                    "name": item['name'],
                    "pcd": self.load_stl_model(item['stl_path']),
                    "thickness": float(item['thickness']),
                    "symmetry": float(item['symmetry']),
                    "icp_dist": float(item['icp_dist']),
                    "voxel": float(item['voxel']),
                    "min_pts": int(item['min_pts'])
                }
                self.get_logger().info(f"모델 등록: [{obj_id}] {item['name']}")
        except Exception as e:
            self.get_logger().error(f"YAML 로드 중 에러 발생: {e}")

    def load_stl_model(self, path):
        """STL 파일을 읽어 ICP 매칭용 점구름으로 변환합니다."""
        if not os.path.exists(path):
            self.get_logger().warn(f"STL 파일 경로 오류: {path}")
            return None
        mesh = o3d.io.read_triangle_mesh(path)
        pcd = mesh.sample_points_uniformly(number_of_points=2000)
        pcd.translate(-pcd.get_center()) # 모델 중심화
        pcd.estimate_normals()
        return pcd

    def reset_queues(self):
        """데이터 필터를 초기화합니다 (잔상 제거용)."""
        for k in self.pos_queues: self.pos_queues[k].clear()
        for k in self.angle_queues:
            self.angle_queues[k]['sin'].clear()
            self.angle_queues[k]['cos'].clear()
        self.prev_angles = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        self.get_logger().info("필터 큐가 초기화되었습니다.")

    def info_callback(self, msg):
        if self.intrinsics is None:
            self.intrinsics = {'fx': msg.k[0], 'fy': msg.k[4], 'ppx': msg.k[2], 'ppy': msg.k[5], 'width': msg.width, 'height': msg.height}

    def process_callback(self, color_msg, depth_msg):
        if self.intrinsics is None: return

        color_image = self.cv_bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, 'passthrough')

        # 1. YOLO 예측
        results = self.model.predict(source=color_image, device=self.device, conf=0.7, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        cls_ids = results[0].boxes.cls.cpu().numpy()

        if len(boxes) == 0:
            self.latest_result["success"] = False
            cv2.imshow("Pose Estimation", color_image)
            cv2.waitKey(1)
            return

        # 2. 점구름 전처리 및 지면 제거
        pcd_all, u_map, v_map = self.generate_pointcloud(depth_image)
        valid_idx = pcd_all[:,2] > 0
        pcd_valid = pcd_all[valid_idx]
        u_valid, v_valid = u_map.reshape(-1)[valid_idx], v_map.reshape(-1)[valid_idx]
        
        try:
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(pcd_valid[::5]) # 샘플링
            plane_model, _ = pcd_o3d.segment_plane(distance_threshold=self.ground_dist_thresh, ransac_n=3, num_iterations=50)
            non_ground_mask = self.point_to_plane_distance(pcd_valid, plane_model) > self.ground_dist_thresh
            ground_removed_pts, u_ng, v_ng = pcd_valid[non_ground_mask], u_valid[non_ground_mask], v_valid[non_ground_mask]
        except:
            ground_removed_pts, u_ng, v_ng = pcd_valid, u_valid, v_valid

        found_at_least_one = False

        # 3. 객체별 루프 (YAML 설정을 기반으로 매칭)
        for idx, (x1, y1, x2, y2) in enumerate(boxes.astype(int)):
            cid = int(cls_ids[idx])
            
            # [변경] YAML 데이터베이스에 해당 ID가 있는지 확인
            if cid not in self.obj_db:
                continue
            cfg = self.obj_db[cid]

            roi_mask = (u_ng >= x1) & (u_ng < x2) & (v_ng >= y1) & (v_ng < y2)
            roi_pts = ground_removed_pts[roi_mask]
            
            # [변경] 물체별 최소 포인트 기준 적용
            if len(roi_pts) < cfg['min_pts']: 
                continue

            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(roi_pts)
            target_pcd = target_pcd.voxel_down_sample(cfg['voxel']) # [변경] 물체별 복셀 크기 적용
            target_pcd.estimate_normals()

            try:
                # ICP 초기값 설정 (OBB 중심 기반)
                obb = target_pcd.get_oriented_bounding_box()
                initial_trans = np.eye(4)
                initial_trans[:3, :3] = obb.R
                initial_trans[:3, 3] = np.array(obb.center) + [0, 0, cfg['thickness']/2.0] # [변경] 물체 두께 반영

                # [변경] 물체별 파라미터를 사용하여 ICP 매칭
                reg_p2p = o3d.pipelines.registration.registration_icp(
                    cfg['pcd'], target_pcd, cfg['icp_dist'], initial_trans,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane()
                )
                final_trans = reg_p2p.transformation

                # 데이터 필터링 및 안정화
                curr_xyz = final_trans[:3, 3] * 1000.0
                for i, k in enumerate(['x', 'y', 'z']): self.pos_queues[k].append(curr_xyz[i])
                
                r_raw, p_raw, y_raw = self.rotation_matrix_to_euler_angles(final_trans[:3, :3])
                # [변경] 물체별 대칭 각도(90도, 180도 등) 반영
                r_s = self.stabilize_angle_symmetry(r_raw, self.prev_angles['roll'], cfg['symmetry'])
                p_s = self.stabilize_angle_symmetry(p_raw, self.prev_angles['pitch'], cfg['symmetry'])
                y_s = self.stabilize_angle_symmetry(y_raw, self.prev_angles['yaw'], cfg['symmetry'])

                avgs_deg = {}
                for k, v in zip(['roll', 'pitch', 'yaw'], [r_s, p_s, y_s]):
                    self.angle_queues[k]['sin'].append(math.sin(v))
                    self.angle_queues[k]['cos'].append(math.cos(v))
                    res_rad = math.atan2(np.median(self.angle_queues[k]['sin']), np.median(self.angle_queues[k]['cos']))
                    avgs_deg[k] = np.degrees(res_rad)
                    self.prev_angles[k] = res_rad

                # 결과 업데이트
                self.latest_result = {
                    "success": True, 
                    "id": cid,
                    "name": cfg['name'],
                    "x_mm": np.median(self.pos_queues['x']), 
                    "y_mm": np.median(self.pos_queues['y']), 
                    "z_mm": np.median(self.pos_queues['z']),
                    "roll_deg": avgs_deg['roll'], "pitch_deg": avgs_deg['pitch'], "yaw_deg": avgs_deg['yaw'],
                    "timestamp": time.time()
                }
                found_at_least_one = True

                # 시각화 (ID 및 이름 표시)
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                txt = f"{cfg['name']} (ID:{cid})"
                cv2.putText(color_image, txt, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            except Exception:
                continue

        if not found_at_least_one:
            self.latest_result["success"] = False

        cv2.imshow("Pose Estimation", color_image)
        cv2.waitKey(1)

    def handle_get_pose(self, request, response):
        """서비스 요청 시 최신 데이터를 반환하고 잔상 방지를 위해 큐를 초기화합니다."""
        current_time = time.time()
        data_time = self.latest_result.get("timestamp", 0.0)
        
        if self.latest_result["success"] and (current_time - data_time < self.timeout):
            d = self.latest_result
            response.success = True
            response.detected_id = int(d["id"])
            response.x, response.y, response.z = d["x_mm"]/1000.0, d["y_mm"]/1000.0, d["z_mm"]/1000.0
            response.rz = float(d["yaw_deg"])
            
            # [중요] 응답 후 큐 초기화 (로봇이 집어간 물체의 잔상 제거)
            self.reset_queues()
            self.latest_result["success"] = False 
            self.get_logger().info(f"[응답] {d['name']} 정보를 전송하고 큐를 비웠습니다.")
        else:
            response.success = False
            self.get_logger().warn("[거부] 유효한 최신 정보가 없습니다.")
        return response

    # --- 유틸리티 함수들 ---
    def stabilize_angle_symmetry(self, current, reference, symmetry_deg):
        """물체의 대칭성(예: 90도 마다 같은 모양)을 고려하여 각도를 보정합니다."""
        step = math.radians(symmetry_deg)
        n = round((reference - current) / step)
        return current + n * step

    def rotation_matrix_to_euler_angles(self, R):
        sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
        if sy > 1e-6: return math.atan2(R[2,1], R[2,2]), math.atan2(-R[2,0], sy), math.atan2(R[1,0], R[0,0])
        else: return math.atan2(-R[1,2], R[1,1]), math.atan2(-R[2,0], sy), 0

    def generate_pointcloud(self, depth_image):
        h, w = depth_image.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        depth_m = depth_image * 0.001
        X = (u - self.intrinsics['ppx']) * depth_m / self.intrinsics['fx']
        Y = (v - self.intrinsics['ppy']) * depth_m / self.intrinsics['fy']
        return np.dstack((X, Y, depth_m)).reshape(-1, 3), u, v

    def point_to_plane_distance(self, pts, plane):
        a, b, c, d = plane
        return np.abs(a*pts[:,0] + b*pts[:,1] + c*pts[:,2] + d) / math.sqrt(a**2 + b**2 + c**2)

def main():
    rclpy.init()
    node = Yolo3DNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()