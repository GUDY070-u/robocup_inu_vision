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

        # 2. Open3D 시각화 설정
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="3D Pose Estimation View", width=640, height=480)
        self.pcd_vis = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd_vis)
        self.cam_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.vis.add_geometry(self.cam_axis)
        self.obj_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
        self.vis.add_geometry(self.obj_axis)
        self.obj_axis_applied_trans = np.eye(4)
        self.view_inited = False

        # 3. 필터 및 이전 상태 저장
        self.window_size = 15
        self.pos_queues = {k: deque(maxlen=self.window_size) for k in ['x', 'y', 'z']}
        self.angle_queues = {
            'roll':  {'sin': deque(maxlen=self.window_size), 'cos': deque(maxlen=self.window_size)},
            'pitch': {'sin': deque(maxlen=self.window_size), 'cos': deque(maxlen=self.window_size)},
            'yaw':   {'sin': deque(maxlen=self.window_size), 'cos': deque(maxlen=self.window_size)}
        }
        self.prev_angles = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}

        self.intrinsics = None
        self.depth_scale = 0.001
        self.latest_result = {"success": False, "x_mm": 0.0, "y_mm": 0.0, "z_mm": 0.0, "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0, "id": -1}
        self.plot_len = 50
        self.plot_data = {k: deque(maxlen=self.plot_len) for k in ['x', 'y', 'z', 'roll', 'pitch', 'yaw']}

        # ROS 통신
        self.cv_bridge = CvBridge()
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.color_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_raw', qos_profile=qos)
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw', qos_profile=qos)
        self.sync = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.process_callback)
        self.info_sub = self.create_subscription(CameraInfo, '/camera/camera/aligned_depth_to_color/camera_info', self.info_callback, 10)
        self.srv = self.create_service(GetObjectPose, '/vision/get_object_pose', self.handle_get_pose)

        threading.Thread(target=self.init_plot_thread, daemon=True).start()

    def load_stl_model(self, path):
        if not os.path.exists(path): return None
        mesh = o3d.io.read_triangle_mesh(path)
        pcd = mesh.sample_points_uniformly(number_of_points=3000)
        pcd.translate(-pcd.get_center())
        pcd.estimate_normals()
        return pcd

    def stabilize_angle_symmetry(self, current, reference, symmetry_step_deg=90.0):
        step = math.radians(symmetry_step_deg)
        n = round((reference - current) / step)
        return current + n * step

    def info_callback(self, msg):
        if self.intrinsics is None:
            self.intrinsics = {'fx': msg.k[0], 'fy': msg.k[4], 'ppx': msg.k[2], 'ppy': msg.k[5], 'width': msg.width, 'height': msg.height}

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
            
            # 3. 회전 데이터 (RPY)
            # 순서대로 rx=Roll, ry=Pitch, rz=Yaw 매핑
            response.rx = float(d["roll_deg"])
            response.ry = float(d["pitch_deg"])
            response.rz = float(d["yaw_deg"])
            
            self.get_logger().info(f"전송 완료: ID={response.detected_id},X={response.x:.3f},Y={response.y:.3f}, Z={response.z:.3f}, RX={response.rx:.2f}, RY={response.ry:.2f}, RZ={response.rz:.2f}")
        else:
            response.success = False
            self.get_logger().warn("물체 검출 실패로 응답할 수 없습니다.")
            
        return response

    def process_callback(self, color_msg, depth_msg):
        if self.intrinsics is None or self.source_pcd is None: return
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
        down = pcd_o3d.voxel_down_sample(0.005)
        plane_model, _ = down.segment_plane(distance_threshold=self.ground_dist_thresh, ransac_n=3, num_iterations=50)
        non_ground_mask = self.point_to_plane_distance(pcd_valid, plane_model) > self.ground_dist_thresh
        ground_removed_pts = pcd_valid[non_ground_mask]
        u_ng, v_ng = u_valid[non_ground_mask], v_valid[non_ground_mask]

        final_trans_to_viz = None

        for idx, (x1, y1, x2, y2) in enumerate(boxes.astype(int)):
            roi_mask = (u_ng >= x1) & (u_ng < x2) & (v_ng >= y1) & (v_ng < y2)
            roi_pts = ground_removed_pts[roi_mask]
            if len(roi_pts) < 100: continue

            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(roi_pts)
            target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

            try:
                # 1. ICP 정합 (Z-Offset 20mm 적용)
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

                # 2. 위치/각도 필터링
                curr_xyz = final_trans[:3, 3] * 1000.0
                for i, k in enumerate(['x', 'y', 'z']): self.pos_queues[k].append(curr_xyz[i])
                avg_x, avg_y, avg_z = [np.median(self.pos_queues[k]) for k in ['x', 'y', 'z']]

                r_raw, p_raw, y_raw = self.rotation_matrix_to_euler_angles(final_trans[:3, :3])
                r_stable = self.stabilize_angle_symmetry(r_raw, self.prev_angles['roll'], symmetry_step_deg=90.0)
                p_stable = self.stabilize_angle_symmetry(p_raw, self.prev_angles['pitch'], symmetry_step_deg=90.0)
                y_stable = self.stabilize_angle_symmetry(y_raw, self.prev_angles['yaw'], symmetry_step_deg=90.0)

                avgs_deg = {}
                for k in ['roll', 'pitch', 'yaw']:
                    self.angle_queues[k]['sin'].append(math.sin(curr_angles := {'roll': r_stable, 'pitch': p_stable, 'yaw': y_stable}[k]))
                    self.angle_queues[k]['cos'].append(math.cos(curr_angles))
                    res_rad = math.atan2(np.median(self.angle_queues[k]['sin']), np.median(self.angle_queues[k]['cos']))
                    avgs_deg[k] = np.degrees(res_rad)
                    self.prev_angles[k] = res_rad

                self.latest_result = {
                    "success": True, "id": int(cls_ids[idx]),
                    "x_mm": avg_x, "y_mm": avg_y, "z_mm": avg_z,
                    "roll_deg": avgs_deg['roll'], "pitch_deg": avgs_deg['pitch'], "yaw_deg": avgs_deg['yaw']
                }

                # 그래프 데이터
                for k in ['x', 'y', 'z']: self.plot_data[k].append(self.latest_result[f"{k}_mm"])
                for k in ['roll', 'pitch', 'yaw']: self.plot_data[k].append(self.latest_result[f"{k}_deg"])

                # OpenCV 시각화
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                txt_xyz = f"XYZ: {avg_x:.1f}, {avg_y:.1f}, {avg_z:.1f}"
                txt_rpy = f"RPY: {avgs_deg['roll']:.1f}, {avgs_deg['pitch']:.1f}, {avgs_deg['yaw']:.1f}"
                cv2.putText(color_image, txt_xyz, (x1, y1-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(color_image, txt_rpy, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            except Exception: pass

        self.update_visualization(color_image, ground_removed_pts, final_trans_to_viz)

    def rotation_matrix_to_euler_angles(self, R):
        sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
        if sy > 1e-6: return math.atan2(R[2,1], R[2,2]), math.atan2(-R[2,0], sy), math.atan2(R[1,0], R[0,0])
        else: return math.atan2(-R[1,2], R[1,1]), math.atan2(-R[2,0], sy), 0

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
        cv2.imshow("Final Pose Estimation", color)
        cv2.waitKey(1)
        if len(points) > 0:
            self.pcd_vis.points = o3d.utility.Vector3dVector(points)
            self.vis.update_geometry(self.pcd_vis)
            if final_trans is not None:
                self.obj_axis.transform(np.linalg.inv(self.obj_axis_applied_trans))
                self.obj_axis.transform(final_trans)
                self.obj_axis_applied_trans = final_trans
                self.vis.update_geometry(self.obj_axis)
            
            # [시점 초기화 로직: 카메라 뷰로 설정]
            if not self.view_inited:
                ctr = self.vis.get_view_control()
                ctr.set_lookat([0, 0, 0.5]) # 카메라 앞 0.5m 지점 응시
                ctr.set_front([0, 0, -1])   # 정면(Z축 방향)을 바라봄
                ctr.set_up([0, -1, 0])      # 카메라 Y축 특성 보정 (똑바로 서게 함)
                ctr.set_zoom(0.5)
                self.view_inited = True
                
            self.vis.poll_events(); self.vis.update_renderer()

    def init_plot_thread(self):
        plt.ion()
        fig, axes = plt.subplots(3, 2, figsize=(8, 6))
        lines = {k: axes.flat[i].plot([], [], label=k.upper())[0] for i, k in enumerate(['x', 'roll', 'y', 'pitch', 'z', 'yaw'])}
        ylims = [(-300,300), (-180,180), (-300,300), (-180,180), (0,1200), (-180,180)]
        for i, ax in enumerate(axes.flat):
            ax.set_ylim(ylims[i]); ax.set_xlim(0, self.plot_len); ax.grid(True)
        plt.tight_layout()
        while rclpy.ok():
            for k in ['x', 'roll', 'y', 'pitch', 'z', 'yaw']:
                lines[k].set_data(range(len(self.plot_data[k])), list(self.plot_data[k]))
            fig.canvas.draw_idle(); plt.pause(0.1)

def main():
    rclpy.init()
    node = Yolo3DNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()