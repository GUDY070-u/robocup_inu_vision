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
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from msgs_pkg.srv import GetObjectPose

class Yolo3DNode(Node):
    def __init__(self):
        super().__init__('yolo_3d_node')

        # 1. 파라미터 및 설정
        self.declare_parameter('model_path', '/home/user/ros2_ws/src/vision/yolo_models/0128_train/weights/best.pt')
        self.declare_parameter('stl_path', '/home/user/ros2_ws/src/vision/config/Aluminum4040.stl')
        self.declare_parameter('model_device', 'cuda:0')
        self.declare_parameter('ground_dist_thresh', 0.01)

        model_path = self.get_parameter('model_path').value
        self.stl_path = self.get_parameter('stl_path').value
        self.device = self.get_parameter('model_device').value
        self.ground_dist_thresh = self.get_parameter('ground_dist_thresh').value

        # YOLO 모델 로드
        self.get_logger().info(f"YOLO 로드: {model_path}")
        self.model = YOLO(model_path)

        # STL 모델 로드 (ICP용)
        self.source_pcd = self.load_stl_model(self.stl_path)

        # 2. Open3D 시각화 설정
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="3D Registration View", width=640, height=480)
        self.pcd_vis = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd_vis)
        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
        self.view_inited = False

        # 3. ROS 통신 설정
        self.cv_bridge = CvBridge()
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

        self.color_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_raw', qos_profile=qos)
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw', qos_profile=qos)
        self.sync = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.process_callback)
        
        self.info_sub = self.create_subscription(CameraInfo, '/camera/camera/aligned_depth_to_color/camera_info', self.info_callback, 10)
        self.srv = self.create_service(GetObjectPose, '/vision/get_object_pose', self.handle_get_pose)

        # 데이터 저장 및 이동평균 큐
        self.intrinsics = None
        self.depth_scale = 0.001
        self.latest_result = {"success": False, "x_mm": 0.0, "y_mm": 0.0, "z_mm": 0.0, "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0, "id": -1}
        
        self.window_size = 5
        self.queues = {k: deque(maxlen=self.window_size) for k in ['x', 'y', 'z', 'roll', 'pitch', 'yaw']}
        
        self.plot_len = 50
        self.plot_data = {k: deque(maxlen=self.plot_len) for k in ['x', 'y', 'z', 'roll', 'pitch', 'yaw']}

        threading.Thread(target=self.init_plot_thread, daemon=True).start()

    def load_stl_model(self, path):
        """STL 파일을 읽어 ICP에 사용할 포인트 클라우드로 변환"""
        if not os.path.exists(path):
            self.get_logger().error(f"STL 파일을 찾을 수 없음: {path}")
            return None
        
        mesh = o3d.io.read_triangle_mesh(path)
        # 중요: 대부분의 STL은 mm 단위이나 Open3D는 m 단위이므로 스케일 조정 (필요 시)
        # mesh.scale(0.001, center=(0, 0, 0)) 
        
        # 메쉬 표면에서 점을 골고루 추출
        pcd = mesh.sample_points_uniformly(number_of_points=2000)
        # 모델의 중심을 0,0,0으로 맞춤 (정합 정확도 향상)
        pcd.translate(-pcd.get_center())
        self.get_logger().info(f"STL 모델 로드 완료: {path}")
        return pcd

    def info_callback(self, msg):
        if self.intrinsics is None:
            self.intrinsics = {'fx': msg.k[0], 'fy': msg.k[4], 'ppx': msg.k[2], 'ppy': msg.k[5], 'width': msg.width, 'height': msg.height}

    def handle_get_pose(self, request, response):
        self.get_logger().info("[Service] 요청 수신. 데이터 전송 중...")
        data = self.latest_result
        if data["success"]:
            response.success = True
            response.detected_id = int(data["id"])
            response.x, response.y, response.z = data["x_mm"]/1000.0, data["y_mm"]/1000.0, data["z_mm"]/1000.0
            response.rx, response.ry, response.rz = float(data["roll_deg"]), float(data["pitch_deg"]), float(data["yaw_deg"])
        else:
            response.success = False
        return response

    def process_callback(self, color_msg, depth_msg):
        if self.intrinsics is None or self.model is None or self.source_pcd is None:
            return

        color_image = self.cv_bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, 'passthrough')

        results = self.model.predict(source=color_image, device=self.device, conf=0.7, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        cls_ids = results[0].boxes.cls.cpu().numpy()

        pcd_all, u_map, v_map = self.generate_pointcloud(depth_image)
        valid_idx = pcd_all[:,2] > 0
        pcd_valid, u_valid, v_valid = pcd_all[valid_idx], u_map.reshape(-1)[valid_idx], v_map.reshape(-1)[valid_idx]

        if len(pcd_valid) < 100: return

        # 지면 제거
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_valid)
        down = pcd_o3d.voxel_down_sample(0.01)
        plane_model, _ = down.segment_plane(distance_threshold=self.ground_dist_thresh, ransac_n=3, num_iterations=50)
        non_ground_mask = self.point_to_plane_distance(pcd_valid, plane_model) > self.ground_dist_thresh
        
        ground_removed_pts = pcd_valid[non_ground_mask]
        u_ng, v_ng = u_valid[non_ground_mask], v_valid[non_ground_mask]

        for idx, (x1, y1, x2, y2) in enumerate(boxes.astype(int)):
            roi_mask = (u_ng >= x1) & (u_ng < x2) & (v_ng >= y1) & (v_ng < y2)
            roi_pts = ground_removed_pts[roi_mask]
            if len(roi_pts) < 50: continue

            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(roi_pts)

            try:
                # 1. 초기 추측값 (Initial Guess) 설정: OBB 활용
                obb = target_pcd.get_oriented_bounding_box()
                initial_trans = np.eye(4)
                initial_trans[:3, :3] = obb.R
                initial_trans[:3, 3] = obb.center

                # 2. ICP 정합 (STL 모델을 실제 포인트에 맞춤)
                reg_p2p = o3d.pipelines.registration.registration_icp(
                    self.source_pcd, target_pcd, 0.02, initial_trans,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint()
                )
                
                # 3. 정합된 결과 행렬 추출
                final_trans = reg_p2p.transformation
                final_xyz = final_trans[:3, 3] * 1000.0 # mm 변환
                final_R = final_trans[:3, :3]
                r, p, y = self.rotation_matrix_to_euler_angles(final_R)
                r_deg, p_deg, y_deg = np.degrees(r), np.degrees(p), np.degrees(y)

                # 이동평균 및 데이터 업데이트
                vals = [final_xyz[0], final_xyz[1], final_xyz[2], r_deg, p_deg, y_deg]
                keys = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
                avgs = {}
                for k, v in zip(keys, vals):
                    self.queues[k].append(v)
                    avgs[k] = np.mean(self.queues[k])
                    self.plot_data[k].append(avgs[k])

                self.latest_result = {
                    "success": True, "id": int(cls_ids[idx]),
                    "x_mm": avgs['x'], "y_mm": avgs['y'], "z_mm": avgs['z'],
                    "roll_deg": avgs['roll'], "pitch_deg": avgs['pitch'], "yaw_deg": avgs['yaw']
                }

                # 시각화 텍스트
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                info_RPY = f"RPY:({avgs['roll']:.2f},{avgs['pitch']:.2f},{avgs['yaw']:.2f})"
                info_XYZ = f"XYZ:({avgs['x']:.2f},{avgs['y']:.2f},{avgs['z']:.2f})"

                cv2.putText(color_image, info_RPY,(x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 50), 2)
                cv2.putText(color_image, info_XYZ,(x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 50), 2)

            except Exception as e:
                self.get_logger().warn(f"ICP 정합 실패: {e}")

        self.update_visualization(color_image, ground_removed_pts)

    def rotation_matrix_to_euler_angles(self, R):
        sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
        if sy > 1e-6:
            return math.atan2(R[2,1], R[2,2]), math.atan2(-R[2,0], sy), math.atan2(R[1,0], R[0,0])
        else:
            return math.atan2(-R[1,2], R[1,1]), math.atan2(-R[2,0], sy), 0

    def generate_pointcloud(self, depth_image):
        h, w = depth_image.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        depth_m = depth_image * self.depth_scale
        X = (u - self.intrinsics['ppx']) * depth_m / self.intrinsics['fx']
        Y = (v - self.intrinsics['ppy']) * depth_m / self.intrinsics['fy']
        return np.dstack((X, Y, depth_m)).reshape(-1, 3), u, v

    def point_to_plane_distance(self, pts, plane):
        a, b, c, d = plane
        return np.abs(a*pts[:,0] + b*pts[:,1] + c*pts[:,2] + d) / math.sqrt(a**2 + b**2 + c**2)

    def update_visualization(self, color, points):
        cv2.imshow("YOLO + STL Matching", color)
        cv2.waitKey(1)
        if len(points) > 0:
            self.pcd_vis.points = o3d.utility.Vector3dVector(points)
            if not self.view_inited:
                ctr = self.vis.get_view_control()
                ctr.set_lookat(self.pcd_vis.get_center()); ctr.set_zoom(0.5)
                self.view_inited = True
            self.vis.update_geometry(self.pcd_vis)
            self.vis.poll_events(); self.vis.update_renderer()

    def init_plot_thread(self):
        plt.ion()
        self.fig, axes = plt.subplots(3, 2, figsize=(10, 8))
        self.lines = {}
        titles = ['X', 'Roll', 'Y', 'Pitch', 'Z', 'Yaw']
        keys = ['x', 'roll', 'y', 'pitch', 'z', 'yaw']
        ylims = [(-200, 200), (-180, 180), (-200, 200), (-180, 180), (0, 1000), (-180, 180)]
        
        for i, (ax, title, key, ylim) in enumerate(zip(axes.flat, titles, keys, ylims)):
            self.lines[key], = ax.plot([], [], label=title)
            ax.set_title(title); ax.set_ylim(ylim); ax.set_xlim(0, self.plot_len)
        
        plt.tight_layout()
        while rclpy.ok():
            for k in keys:
                self.lines[k].set_data(range(len(self.plot_data[k])), list(self.plot_data[k]))
            self.fig.canvas.draw_idle()
            plt.pause(0.1)

def main():
    rclpy.init()
    node = Yolo3DNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()