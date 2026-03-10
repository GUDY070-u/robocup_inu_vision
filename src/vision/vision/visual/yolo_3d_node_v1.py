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
import threading

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
        self.get_logger().info(f"🔄 Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        
        self.get_logger().info(f"🔄 Loading STL model from {self.stl_path}...")
        self.source_pcd = self.load_stl_model(self.stl_path)
        if self.source_pcd is None:
            self.get_logger().error(f"❌ Failed to load STL file at {self.stl_path}!")

        # 2. Open3D 시각화 설정
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="3D View", width=640, height=480)
        self.pcd_vis = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd_vis)
        self.obj_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
        self.vis.add_geometry(self.obj_axis)
        self.obj_axis_applied_trans = np.eye(4)
        self.view_inited = False

        # 3. 데이터 큐 및 상태
        self.window_size = 15
        self.pos_queues = {k: deque(maxlen=self.window_size) for k in ['x', 'y', 'z']}
        self.angle_queues = {
            'roll':  {'sin': deque(maxlen=self.window_size), 'cos': deque(maxlen=self.window_size)},
            'pitch': {'sin': deque(maxlen=self.window_size), 'cos': deque(maxlen=self.window_size)},
            'yaw':   {'sin': deque(maxlen=self.window_size), 'cos': deque(maxlen=self.window_size)}
        }
        self.prev_angles = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        self.intrinsics = None
        self.latest_result = {"success": False, "x_mm": 0.0, "y_mm": 0.0, "z_mm": 0.0, "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0, "id": -1}

        # 4. ROS 통신 설정
        self.cv_bridge = CvBridge()
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        
        self.color_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_raw', qos_profile=qos)
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw', qos_profile=qos)
        
        # [적용] slop를 0.1에서 0.5로 변경하여 동기화 기준 완화
        self.sync = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=10, slop=0.5)
        self.sync.registerCallback(self.process_callback)
        
        self.info_sub = self.create_subscription(CameraInfo, '/camera/camera/aligned_depth_to_color/camera_info', self.info_callback, 10)
        self.srv = self.create_service(GetObjectPose, '/vision/get_object_pose', self.handle_get_pose)

        self.get_logger().info("✅ YOLO 3D Node Started. Waiting for camera topics...")

    def load_stl_model(self, path):
        if not os.path.exists(path): return None
        mesh = o3d.io.read_triangle_mesh(path)
        pcd = mesh.sample_points_uniformly(number_of_points=2000)
        pcd.translate(-pcd.get_center())
        pcd.estimate_normals()
        return pcd

    def info_callback(self, msg):
        if self.intrinsics is None:
            self.get_logger().info("📷 Camera Info (Intrinsics) Received!")
            self.intrinsics = {'fx': msg.k[0], 'fy': msg.k[4], 'ppx': msg.k[2], 'ppy': msg.k[5], 'width': msg.width, 'height': msg.height}

    def process_callback(self, color_msg, depth_msg):
        # [적용] 데이터 수신 로그 추가
        # self.get_logger().info("📩 Images received (Color & Depth synced)")

        if self.intrinsics is None:
            self.get_logger().warn("⚠️ Waiting for Camera Info (Intrinsics)...", throttle_duration_sec=5.0)
            return
        if self.source_pcd is None:
            self.get_logger().error("❌ STL model not loaded. Check model path.")
            return

        color_image = self.cv_bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, 'passthrough')

        # YOLO 예측
        results = self.model.predict(source=color_image, device=self.device, conf=0.7, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        cls_ids = results[0].boxes.cls.cpu().numpy()

        if len(boxes) == 0:
            # 물체가 발견되지 않을 때 로그 (너무 자주 찍히지 않게 cv2.imshow는 유지)
            cv2.imshow("Pose Estimation", color_image)
            cv2.waitKey(1)
            return

        # self.get_logger().info(f"🎯 YOLO found {len(boxes)} object(s).")

        pcd_all, u_map, v_map = self.generate_pointcloud(depth_image)
        valid_idx = pcd_all[:,2] > 0
        pcd_valid, u_valid, v_valid = pcd_all[valid_idx], u_map.reshape(-1)[valid_idx], v_map.reshape(-1)[valid_idx]
        
        if len(pcd_valid) < 100:
            self.get_logger().warn("⚠️ Not enough valid depth points.")
            return

        # 지면 제거
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_valid)
        down = pcd_o3d.voxel_down_sample(0.005)
        
        try:
            plane_model, _ = down.segment_plane(distance_threshold=self.ground_dist_thresh, ransac_n=3, num_iterations=50)
            non_ground_mask = self.point_to_plane_distance(pcd_valid, plane_model) > self.ground_dist_thresh
            ground_removed_pts = pcd_valid[non_ground_mask]
            u_ng, v_ng = u_valid[non_ground_mask], v_valid[non_ground_mask]
        except Exception as e:
            self.get_logger().error(f"Plane segmentation failed: {e}")
            return

        final_trans_to_viz = None

        for idx, (x1, y1, x2, y2) in enumerate(boxes.astype(int)):
            roi_mask = (u_ng >= x1) & (u_ng < x2) & (v_ng >= y1) & (v_ng < y2)
            roi_pts = ground_removed_pts[roi_mask]
            
            if len(roi_pts) < 50: 
                continue

            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(roi_pts)
            target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

            try:
                obb = target_pcd.get_oriented_bounding_box()
                initial_trans = np.eye(4)
                initial_trans[:3, :3] = obb.R
                center = np.array(obb.center)
                center[2] += self.half_thickness 
                initial_trans[:3, 3] = center

                reg_p2p = o3d.pipelines.registration.registration_icp(
                    self.source_pcd, target_pcd, 0.05, initial_trans,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane()
                )
                
                final_trans = reg_p2p.transformation
                final_trans_to_viz = final_trans

                # 필터링 및 결과 저장
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

                self.latest_result = {
                    "success": True, "id": int(cls_ids[idx]),
                    "x_mm": np.median(self.pos_queues['x']), "y_mm": np.median(self.pos_queues['y']), "z_mm": np.median(self.pos_queues['z']),
                    "roll_deg": avgs_deg['roll'], "pitch_deg": avgs_deg['pitch'], "yaw_deg": avgs_deg['yaw']
                }

                # 검출 성공 로그
                # self.get_logger().info(f"✅ Object Detected: ID={self.latest_result['id']} X={self.latest_result['x_mm']:.1f}")

                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            except Exception as e:
                pass

        self.update_visualization(color_image, ground_removed_pts, final_trans_to_viz)

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

    def handle_get_pose(self, request, response):
        d = self.latest_result
        if d["success"]:
            # 1. 성공 여부 및 ID
            response.success = True
            response.detected_id = int(d["id"])
            
            # 2. 위치 데이터 (XYZ)
            response.x = d["x_mm"] / 1000.0
            response.y = d["y_mm"] / 1000.0
            response.z = d["z_mm"] / 1000.0
            
            # 3. 회전 데이터 (rz만 존재하므로 rz에 Yaw 값을 넣음)
            # 서비스 파일에 rx, ry가 없으므로 아래 줄처럼 rz만 할당해야 합니다.
            response.rz = float(d["yaw_deg"])
            
            self.get_logger().info(f"📤 서비스 응답: ID={response.detected_id}, X={response.x:.3f}, Y={response.y:.3f}, Z={response.z:.3f}, RZ(Yaw)={response.rz:.2f}")
        else:
            response.success = False
            self.get_logger().warn("⚠️ 물체 검출 실패로 응답할 수 없습니다.")
            
        return response

    def update_visualization(self, color, points, final_trans=None):
        cv2.imshow("Pose Estimation", color)
        cv2.waitKey(1)
        if len(points) > 0:
            self.pcd_vis.points = o3d.utility.Vector3dVector(points)
            self.vis.update_geometry(self.pcd_vis)
            if final_trans is not None:
                self.obj_axis.transform(np.linalg.inv(self.obj_axis_applied_trans))
                self.obj_axis.transform(final_trans)
                self.obj_axis_applied_trans = final_trans
                self.vis.update_geometry(self.obj_axis)
            self.vis.poll_events()
            self.vis.update_renderer()

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