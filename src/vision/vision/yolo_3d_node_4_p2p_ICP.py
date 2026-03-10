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
        self.declare_parameter('stl_path', '/home/user/ros2_ws/src/vision/config/M30.stl')
        self.declare_parameter('model_device', 'cuda:0')
        self.declare_parameter('ground_dist_thresh', 0.01)
        self.declare_parameter('object_thickness_m', 0.024) # 물체 두께 40mm

        model_path = self.get_parameter('model_path').value
        self.stl_path = self.get_parameter('stl_path').value
        self.device = self.get_parameter('model_device').value
        self.ground_dist_thresh = self.get_parameter('ground_dist_thresh').value
        self.half_thickness = self.get_parameter('object_thickness_m').value / 2.0 # 0.02m

        # YOLO 모델 로드
        self.get_logger().info(f"YOLO 로드: {model_path}")
        self.model = YOLO(model_path)

        # STL 모델 로드 (ICP용)
        self.source_pcd = self.load_stl_model(self.stl_path)

        # 2. Open3D 시각화 설정
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="3D Registration View", width=640, height=480)
        
        # [A] 배경 점군 (카메라 데이터)
        self.pcd_vis = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd_vis)
        
        # [B] 카메라 원점 좌표축 (고정)
        self.cam_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.vis.add_geometry(self.cam_axis)

        # [C] 물체 포즈 좌표축 (ICP 결과에 따라 움직임)
        self.obj_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08) # 약간 작게 설정
        self.vis.add_geometry(self.obj_axis)
        self.obj_axis_applied_trans = np.eye(4) # 이전 변환 저장용

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
        self.latest_result = {"success": False, "x_mm": 0.0, "y_mm": 0.0, "z_mm": 0.0, "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0, "id": -1, "trans_mat": np.eye(4)}
        
        self.window_size = 5
        self.queues = {k: deque(maxlen=self.window_size) for k in ['x', 'y', 'z', 'roll', 'pitch', 'yaw']}
        
        self.plot_len = 50
        self.plot_data = {k: deque(maxlen=self.plot_len) for k in ['x', 'y', 'z', 'roll', 'pitch', 'yaw']}

        threading.Thread(target=self.init_plot_thread, daemon=True).start()

    def load_stl_model(self, path):
        if not os.path.exists(path):
            self.get_logger().error(f"STL 파일을 찾을 수 없음: {path}")
            return None
        mesh = o3d.io.read_triangle_mesh(path)
        pcd = mesh.sample_points_uniformly(number_of_points=3000)
        pcd.translate(-pcd.get_center()) # 모델의 중심을 0,0,0으로
        pcd.estimate_normals()
        self.get_logger().info(f"STL 모델 로드 및 노멀 계산 완료")
        return pcd

    def info_callback(self, msg):
        if self.intrinsics is None:
            self.intrinsics = {'fx': msg.k[0], 'fy': msg.k[4], 'ppx': msg.k[2], 'ppy': msg.k[5], 'width': msg.width, 'height': msg.height}

    def handle_get_pose(self, request, response):
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

        # 지면 제거 로직
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_valid)
        down = pcd_o3d.voxel_down_sample(0.005)
        plane_model, _ = down.segment_plane(distance_threshold=self.ground_dist_thresh, ransac_n=3, num_iterations=50)
        non_ground_mask = self.point_to_plane_distance(pcd_valid, plane_model) > self.ground_dist_thresh
        ground_removed_pts = pcd_valid[non_ground_mask]
        u_ng, v_ng = u_valid[non_ground_mask], v_valid[non_ground_mask]

        final_trans_to_viz = None

        for idx, (x1, y1, x2, y2) in enumerate(boxes.astype(int)):
            roi_mask = (u_ng >= x1) & (u_ng < x2) & (v_ng >= y1) & (v_ng < y2)
            roi_pts = ground_removed_pts[roi_mask]
            if len(roi_pts) < 50: continue

            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(roi_pts)
            target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

            try:
                # 1. 초기 추측 (OBB + 20mm Z-offset)
                obb = target_pcd.get_oriented_bounding_box()
                initial_trans = np.eye(4)
                initial_trans[:3, :3] = obb.R
                corrected_center = np.array(obb.center)
                corrected_center[2] += self.half_thickness # 두께의 절반만큼 깊게 이동
                initial_trans[:3, 3] = corrected_center

                # 2. 정밀 ICP (Point-to-Plane)
                reg_p2p = o3d.pipelines.registration.registration_icp(
                    self.source_pcd, target_pcd, 0.05, initial_trans,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane()
                )
                
                final_trans = reg_p2p.transformation
                final_trans_to_viz = final_trans # 시각화용 저장

                # 3. 결과 후처리 (이동평균)
                r, p, y = self.rotation_matrix_to_euler_angles(final_trans[:3, :3])
                vals = [final_trans[0,3]*1000.0, final_trans[1,3]*1000.0, final_trans[2,3]*1000.0, np.degrees(r), np.degrees(p), np.degrees(y)]
                keys = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
                avgs = {}
                for k, v in zip(keys, vals):
                    self.queues[k].append(v)
                    avgs[k] = np.mean(self.queues[k])
                    self.plot_data[k].append(avgs[k])

                self.latest_result = {
                    "success": True, "id": int(cls_ids[idx]),
                    "x_mm": avgs['x'], "y_mm": avgs['y'], "z_mm": avgs['z'],
                    "roll_deg": avgs['roll'], "pitch_deg": avgs['pitch'], "yaw_deg": avgs['yaw'],
                    "trans_mat": final_trans
                }

                # 2D 텍스트 표시
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(color_image, f"XYZ:({avgs['x']:.1f},{avgs['y']:.1f},{avgs['z']:.1f})", (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 50), 2)
                cv2.putText(color_image, f"RPY:({avgs['roll']:.1f},{avgs['pitch']:.1f},{avgs['yaw']:.1f})", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 50), 2)

            except Exception as e:
                pass

        self.update_visualization(color_image, ground_removed_pts, final_trans_to_viz)

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

    def update_visualization(self, color, points, final_trans=None):
        cv2.imshow("YOLO + STL (40mm Offset Applied)", color)
        cv2.waitKey(1)
        
        if len(points) > 0:
            self.pcd_vis.points = o3d.utility.Vector3dVector(points)
            self.vis.update_geometry(self.pcd_vis)
            
            # [좌표축 시각화 업데이트]
            if final_trans is not None:
                # 1. 이전 변환을 되돌려 원점으로 보냄
                self.obj_axis.transform(np.linalg.inv(self.obj_axis_applied_trans))
                # 2. 새로운 변환 적용
                self.obj_axis.transform(final_trans)
                # 3. 현재 변환 저장
                self.obj_axis_applied_trans = final_trans
                self.vis.update_geometry(self.obj_axis)

            if not self.view_inited:
                self.vis.get_view_control().set_lookat(self.pcd_vis.get_center())
                self.vis.get_view_control().set_zoom(0.7)
                self.view_inited = True
            
            self.vis.poll_events()
            self.vis.update_renderer()

    def init_plot_thread(self):
        plt.ion()
        fig, axes = plt.subplots(3, 2, figsize=(8, 6))
        lines = {}
        keys = ['x', 'roll', 'y', 'pitch', 'z', 'yaw']
        ylims = [(-300, 300), (-180, 180), (-300, 300), (-180, 180), (0, 1200), (-180, 180)]
        for i, (ax, key, ylim) in enumerate(zip(axes.flat, keys, ylims)):
            lines[key], = ax.plot([], [], label=key.upper())
            ax.set_title(key.upper()); ax.set_ylim(ylim); ax.set_xlim(0, self.plot_len)
        plt.tight_layout()
        while rclpy.ok():
            for k in keys:
                lines[k].set_data(range(len(self.plot_data[k])), list(self.plot_data[k]))
            fig.canvas.draw_idle()
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