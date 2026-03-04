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

        # 12번 타겟 데이터 전용 큐 및 상태
        self.window_size = 15
        self.pos_queues = {k: deque(maxlen=self.window_size) for k in ['x', 'y', 'z']}
        self.angle_queues = {
            'roll':  {'sin': deque(maxlen=self.window_size), 'cos': deque(maxlen=self.window_size)},
            'pitch': {'sin': deque(maxlen=self.window_size), 'cos': deque(maxlen=self.window_size)},
            'yaw':   {'sin': deque(maxlen=self.window_size), 'cos': deque(maxlen=self.window_size)}
        }
        self.prev_angles = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        self.intrinsics = None
        
        # 최신 12번 결과 저장 변수
        self.latest_target_result = {
            "success": False, 
            "id": 12,
            "x_mm": 0.0, "y_mm": 0.0, "z_mm": 0.0, 
            "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0, 
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

        self.get_logger().info("✅ YOLO 3D Node Started. (Reg: ALL, Service: ID 12 only)")

    def load_stl_model(self, path):
        if not os.path.exists(path): return None
        mesh = o3d.io.read_triangle_mesh(path)
        pcd = mesh.sample_points_uniformly(number_of_points=2000)
        pcd.translate(-pcd.get_center())
        pcd.estimate_normals()
        return pcd

    def reset_target_queues(self):
        for k in self.pos_queues: self.pos_queues[k].clear()
        for k in self.angle_queues:
            self.angle_queues[k]['sin'].clear()
            self.angle_queues[k]['cos'].clear()
        self.prev_angles = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        self.get_logger().info("🧹 12번 타겟 큐가 초기화되었습니다.")

    def info_callback(self, msg):
        if self.intrinsics is None:
            self.intrinsics = {'fx': msg.k[0], 'fy': msg.k[4], 'ppx': msg.k[2], 'ppy': msg.k[5], 'width': msg.width, 'height': msg.height}

    def process_callback(self, color_msg, depth_msg):
        if self.intrinsics is None or self.source_pcd is None: return

        color_image = self.cv_bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, 'passthrough')

        results = self.model.predict(source=color_image, device=self.device, conf=0.7, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        cls_ids = results[0].boxes.cls.cpu().numpy()

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

        found_12 = False

        # 모든 물체에 대해 루프 수행 (정합 및 시각화)
        for idx, (x1, y1, x2, y2) in enumerate(boxes.astype(int)):
            obj_id = int(cls_ids[idx])
            
            roi_mask = (u_ng >= x1) & (u_ng < x2) & (v_ng >= y1) & (v_ng < y2)
            roi_pts = ground_removed_pts[roi_mask]
            
            if len(roi_pts) < 50: 
                continue

            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(roi_pts)
            target_pcd.estimate_normals()

            try:
                # 1. 모든 물체에 대해 ICP 수행
                obb = target_pcd.get_oriented_bounding_box()
                initial_trans = np.eye(4)
                initial_trans[:3, :3] = obb.R
                initial_trans[:3, 3] = np.array(obb.center) + [0, 0, self.half_thickness]

                reg_p2p = o3d.pipelines.registration.registration_icp(
                    self.source_pcd, target_pcd, 0.05, initial_trans,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane()
                )
                final_trans = reg_p2p.transformation
                
                curr_xyz = final_trans[:3, 3] * 1000.0
                r_raw, p_raw, y_raw = self.rotation_matrix_to_euler_angles(final_trans[:3, :3])

                # 2. 만약 12번 물체라면 필터링 적용 및 서비스 데이터 업데이트
                if obj_id == 12:
                    for i, k in enumerate(['x', 'y', 'z']): self.pos_queues[k].append(curr_xyz[i])
                    
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

                    avg_x, avg_y, avg_z = np.median(self.pos_queues['x']), np.median(self.pos_queues['y']), np.median(self.pos_queues['z'])
                    
                    self.latest_target_result = {
                        "success": True, "id": 12,
                        "x_mm": avg_x, "y_mm": avg_y, "z_mm": avg_z,
                        "roll_deg": avgs_deg['roll'], "pitch_deg": avgs_deg['pitch'], "yaw_deg": avgs_deg['yaw'],
                        "timestamp": time.time()
                    }
                    found_12 = True
                    color_box = (0, 255, 0) # 타겟은 녹색
                    label = f"TARGET ID: 12"
                    display_xyz = f"XYZ: {avg_x:.1f}, {avg_y:.1f}, {avg_z:.1f}"
                else:
                    # 12번이 아닌 물체는 필터링 없이 원본 값으로 시각화만 수행
                    color_box = (255, 0, 0) # 나머지는 파란색
                    label = f"ID: {obj_id}"
                    display_xyz = f"XYZ: {curr_xyz[0]:.1f}, {curr_xyz[1]:.1f}, {curr_xyz[2]:.1f}"

                # 공통 시각화 (박스 및 좌표)
                cv2.rectangle(color_image, (x1, y1), (x2, y2), color_box, 2)
                cv2.putText(color_image, label, (x1, y1 - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_box, 2)
                cv2.putText(color_image, display_xyz, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 1)

            except Exception:
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 0, 255), 1)

        # 현재 프레임에 12번이 없으면 success를 False로 (서비스 거부용)
        if not found_12:
            self.latest_target_result["success"] = False

        cv2.imshow("Pose Estimation", color_image)
        cv2.waitKey(1)

    def handle_get_pose(self, request, response):
        """로봇의 요청에 대해 12번 물체 정보만 응답"""
        current_time = time.time()
        data_time = self.latest_target_result.get("timestamp", 0.0)
        
        # 12번 데이터가 있고 0.5초 이내의 싱싱한 데이터일 때만 전송
        if self.latest_target_result["success"] and (current_time - data_time < 0.5):
            d = self.latest_target_result
            response.success = True
            response.detected_id = 12
            response.x, response.y, response.z = d["x_mm"]/1000.0, d["y_mm"]/1000.0, d["z_mm"]/1000.0
            response.rz = float(d["yaw_deg"])
            
            # 전송 후 큐 초기화 (잔상 방지)
            self.reset_target_queues()
            self.latest_target_result["success"] = False 
            
            self.get_logger().info(f"📤 [12번 전송] x={response.x:.3f}, y={response.y:.3f}, z={response.z:.3f}, rz={response.rz:.2f}")
        else:
            response.success = False
            self.get_logger().warn("⚠️ [전송 거부] 12번 물체가 감지되지 않았습니다.")
            
        return response

    # --- 유틸리티 함수 (기존과 동일) ---
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