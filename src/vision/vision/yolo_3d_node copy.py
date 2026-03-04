#!/usr/bin/env python3
import rclpy
import time
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters
import numpy as np
import open3d as o3d
import cv2
from ultralytics import YOLO
import os
import math
from collections import deque

from msgs_pkg.srv import GetObjectPose

class Yolo3DNode(Node):
    def __init__(self):
        super().__init__('yolo_3d_node')

        # 1. 파라미터 설정
        self.declare_parameter('model_path', '/home/user/ros2_ws/src/vision/yolo_models/0128_train/weights/best.pt')
        self.declare_parameter('stl_path', '/home/user/ros2_ws/src/vision/config/Aluminum4040.stl')
        self.declare_parameter('model_device', 'cuda:0')
        self.declare_parameter('ground_dist_thresh', 0.01)
        self.declare_parameter('object_thickness_m', 0.04) 

        model_path = self.get_parameter('model_path').value
        self.stl_path = self.get_parameter('stl_path').value
        self.device = self.get_parameter('model_device').value
        self.ground_dist_thresh = self.get_parameter('ground_dist_thresh').value
        self.half_thickness = self.get_parameter('object_thickness_m').value / 2.0 

        # 모델 로드
        self.model = YOLO(model_path)
        self.source_pcd = self.load_stl_model(self.stl_path)

        # 데이터 큐 및 상태
        self.window_size = 15
        self.pos_queues = {k: deque(maxlen=self.window_size) for k in ['x', 'y', 'z']}
        self.angle_queues = {
            'roll':  {'sin': deque(maxlen=self.window_size), 'cos': deque(maxlen=self.window_size)},
            'pitch': {'sin': deque(maxlen=self.window_size), 'cos': deque(maxlen=self.window_size)},
            'yaw':   {'sin': deque(maxlen=self.window_size), 'cos': deque(maxlen=self.window_size)}
        }
        self.prev_angles = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        self.intrinsics = None
        
        # 최신 결과 저장 변수 (타임스탬프 포함)
        self.latest_result = {
            "success": False, 
            "x_mm": 0.0, "y_mm": 0.0, "z_mm": 0.0, 
            "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0, 
            "id": -1,
            "timestamp": 0.0
        }

        # ROS 통신 설정
        self.cv_bridge = CvBridge()
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.color_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_raw', qos_profile=qos)
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw', qos_profile=qos)
        self.sync = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=10, slop=0.5)
        self.sync.registerCallback(self.process_callback)
        
        self.info_sub = self.create_subscription(CameraInfo, '/camera/camera/aligned_depth_to_color/camera_info', self.info_callback, 10)
        self.srv = self.create_service(GetObjectPose, '/vision/get_object_pose', self.handle_get_pose)

        self.get_logger().info("✅ YOLO 3D Node Started. (Ghosting Prevention Active)")

    def load_stl_model(self, path):
        if not os.path.exists(path): return None
        mesh = o3d.io.read_triangle_mesh(path)
        pcd = mesh.sample_points_uniformly(number_of_points=2000)
        pcd.translate(-pcd.get_center())
        pcd.estimate_normals()
        return pcd

    def reset_queues(self):
        """데이터 큐를 비워 이전 프레임의 잔상을 제거합니다."""
        for k in self.pos_queues: self.pos_queues[k].clear()
        for k in self.angle_queues:
            self.angle_queues[k]['sin'].clear()
            self.angle_queues[k]['cos'].clear()
        self.prev_angles = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        self.get_logger().info("🧹 필터 큐가 초기화되었습니다.")

    def info_callback(self, msg):
        if self.intrinsics is None:
            self.intrinsics = {'fx': msg.k[0], 'fy': msg.k[4], 'ppx': msg.k[2], 'ppy': msg.k[5], 'width': msg.width, 'height': msg.height}

    def process_callback(self, color_msg, depth_msg):
        if self.intrinsics is None or self.source_pcd is None: return

        color_image = self.cv_bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, 'passthrough')

        # 1. YOLO 예측
        results = self.model.predict(source=color_image, device=self.device, conf=0.7, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        cls_ids = results[0].boxes.cls.cpu().numpy()

        # 물체가 하나도 없으면 상태를 False로 만들고 리턴
        if len(boxes) == 0:
            self.latest_result["success"] = False
            cv2.imshow("Pose Estimation", color_image)
            cv2.waitKey(1)
            return

        # 2. 점구름 생성 및 지면 제거
        pcd_all, u_map, v_map = self.generate_pointcloud(depth_image)
        valid_idx = pcd_all[:,2] > 0
        pcd_valid = pcd_all[valid_idx]
        u_valid, v_valid = u_map.reshape(-1)[valid_idx], v_map.reshape(-1)[valid_idx]
        
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_valid)
        down = pcd_o3d.voxel_down_sample(0.005)
        
        try:
            plane_model, _ = down.segment_plane(distance_threshold=self.ground_dist_thresh, ransac_n=3, num_iterations=50)
            non_ground_mask = self.point_to_plane_distance(pcd_valid, plane_model) > self.ground_dist_thresh
            ground_removed_pts = pcd_valid[non_ground_mask]
            u_ng, v_ng = u_valid[non_ground_mask], v_valid[non_ground_mask]
        except:
            ground_removed_pts, u_ng, v_ng = pcd_valid, u_valid, v_valid

        found_at_least_one = False

        # 3. 객체별 루프
        for idx, (x1, y1, x2, y2) in enumerate(boxes.astype(int)):
            # 일단 박스는 무조건 그림
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(color_image, f"ID: {int(cls_ids[idx])}", (x1, y1 - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            roi_mask = (u_ng >= x1) & (u_ng < x2) & (v_ng >= y1) & (v_ng < y2)
            roi_pts = ground_removed_pts[roi_mask]
            
            if len(roi_pts) < 50: 
                cv2.putText(color_image, "3D Error: Low Points", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                continue

            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(roi_pts)
            target_pcd.estimate_normals()

            try:
                # ICP 매칭
                obb = target_pcd.get_oriented_bounding_box()
                initial_trans = np.eye(4)
                initial_trans[:3, :3] = obb.R
                initial_trans[:3, 3] = np.array(obb.center) + [0, 0, self.half_thickness]

                reg_p2p = o3d.pipelines.registration.registration_icp(
                    self.source_pcd, target_pcd, 0.05, initial_trans,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane()
                )
                final_trans = reg_p2p.transformation

                # 데이터 필터링 (Median)
                curr_xyz = final_trans[:3, 3] * 1000.0
                for i, k in enumerate(['x', 'y', 'z']): self.pos_queues[k].append(curr_xyz[i])
                
                r_raw, p_raw, y_raw = self.rotation_matrix_to_euler_angles(final_trans[:3, :3])
                r_s = self.stabilize_angle_symmetry(r_raw, self.prev_angles['roll'])
                p_s = self.stabilize_angle_symmetry(p_raw, self.prev_angles['pitch'])
                y_s = self.stabilize_angle_symmetry(y_raw, self.prev_angles['yaw'])

                avgs_deg = {}
                for k, v in zip(['roll', 'pitch', 'yaw'], [r_s, p_s, y_s]):
                    self.angle_queues[k]['sin'].append(math.sin(v))
                    self.angle_queues[k]['cos'].append(math.cos(v))
                    res_rad = math.atan2(np.median(self.angle_queues[k]['sin']), np.median(self.angle_queues[k]['cos']))
                    avgs_deg[k] = np.degrees(res_rad)
                    self.prev_angles[k] = res_rad

                # 결과 업데이트 (타임스탬프 포함)
                avg_x, avg_y, avg_z = np.median(self.pos_queues['x']), np.median(self.pos_queues['y']), np.median(self.pos_queues['z'])
                self.latest_result = {
                    "success": True, 
                    "id": int(cls_ids[idx]),
                    "x_mm": avg_x, "y_mm": avg_y, "z_mm": avg_z,
                    "roll_deg": avgs_deg['roll'], "pitch_deg": avgs_deg['pitch'], "yaw_deg": avgs_deg['yaw'],
                    "timestamp": time.time()
                }
                found_at_least_one = True

                # 시각화 정보
                txt_xyz = f"XYZ: {avg_x:.1f}, {avg_y:.1f}, {avg_z:.1f}"
                txt_rpy = f"RPY: {avgs_deg['roll']:.1f}, {avgs_deg['pitch']:.1f}, {avgs_deg['yaw']:.1f}"
                cv2.putText(color_image, txt_xyz, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(color_image, txt_rpy, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            except Exception as e:
                cv2.putText(color_image, "3D Match Fail", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 이번 프레임에서 최종 매칭된 물체가 하나도 없다면 success = False
        if not found_at_least_one:
            self.latest_result["success"] = False

        cv2.imshow("Pose Estimation", color_image)
        cv2.waitKey(1)

    def handle_get_pose(self, request, response):
        """로봇의 서비스 요청을 처리합니다."""
        current_time = time.time()
        data_time = self.latest_result.get("timestamp", 0.0)
        
        # 데이터가 있고, 0.5초 이내에 업데이트된 싱싱한 데이터일 경우에만 응답
        if self.latest_result["success"] and (current_time - data_time < 0.5):
            d = self.latest_result
            response.success = True
            response.detected_id = int(d["id"])
            response.x, response.y, response.z = d["x_mm"]/1000.0, d["y_mm"]/1000.0, d["z_mm"]/1000.0
            response.rz = float(d["yaw_deg"])
            
            # [핵심] 응답 후 큐와 상태 초기화 (집은 물체 잔상 제거)
            self.reset_queues()
            self.latest_result["success"] = False 
            
            self.get_logger().info(f"📤 [응답] ID={response.detected_id} 정보를 보내고 큐를 초기화했습니다.")
        else:
            response.success = False
            self.get_logger().warn("⚠️ [거부] 유효한 최신 물체 정보가 없습니다.")
            
        return response

    # --- 유틸리티 함수들 ---
    def rotation_matrix_to_euler_angles(self, R):
        sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
        if sy > 1e-6: return math.atan2(R[2,1], R[2,2]), math.atan2(-R[2,0], sy), math.atan2(R[1,0], R[0,0])
        else: return math.atan2(-R[1,2], R[1,1]), math.atan2(-R[2,0], sy), 0

    def stabilize_angle_symmetry(self, current, reference, symmetry_step_deg=90.0):
        step = math.radians(symmetry_step_deg)
        n = round((reference - current) / step)
        return current + n * step

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