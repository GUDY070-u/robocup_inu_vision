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

# # rb - sb - is
# alias rb='source ~/.bashrc'
# alias sb='source /opt/ros/humble/setup.bash'
# alias is='source ~/ros2_ws/install/local_setup.bash'
# alias cbc='rm -rf build install log && colcon build --symlink-install'
# alias cb='colcon build --symlink-install'

class Yolo3DNode(Node):
    def __init__(self):
        super().__init__('yolo_3d_node')

        # 1. 파라미터
        self.declare_parameter('model_path', '/home/user/ros2_ws/src/vision/yolo_models/0128_train/weights/best.pt')
        self.declare_parameter('model_device', 'cuda:0')
        self.declare_parameter('ground_dist_thresh', 0.01)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        if not os.path.isabs(model_path):
            model_path = os.path.abspath(model_path)
        device = self.get_parameter('model_device').get_parameter_value().string_value
        self.ground_dist_thresh = self.get_parameter('ground_dist_thresh').value

        self.get_logger().info(f"모델 로드: {model_path}, device: {device}")
        try:
            self.model = YOLO(model_path)
            self.device = device
        except Exception as e:
            self.get_logger().error(f"모델 로드 실패: {e}")
            self.model = None

        # 2. Open3D 시각화
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="3D View", width=640, height=480)
        self.pcd_vis = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd_vis)
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        self.vis.add_geometry(axis)
        self.view_inited = False

        # 3. ROS 구독 및 서비스 서버 생성
        self.cv_bridge = CvBridge()
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                 history=HistoryPolicy.KEEP_LAST, depth=1)

        # self.color_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw', qos_profile=qos_profile)
        # self.depth_sub = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw', qos_profile=qos_profile)
        # self.info_sub = self.create_subscription(CameraInfo, '/camera/aligned_depth_to_color/camera_info', self.info_callback, 10)

        self.color_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_raw', qos_profile=qos_profile)
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw', qos_profile=qos_profile)
        self.info_sub = self.create_subscription(CameraInfo, '/camera/camera/aligned_depth_to_color/camera_info', self.info_callback, 10)

        self.sync = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.process_callback)

        self.intrinsics = None
        self.depth_scale = 0.001

        self.srv = self.create_service(GetObjectPose, '/vision/get_object_pose', self.handle_get_pose)
        
        # 서비스 응답용 최신 데이터 저장 변수
        self.latest_result = {
            "success": False,
            "x_mm": 0.0, "y_mm": 0.0, "z_mm": 0.0, "yaw_deg": 0.0,
            "id": -1
        }

        # 4. 이동평균 및 그래프용 큐
        self.window_size = 5
        self.x_queue = deque(maxlen=self.window_size)
        self.y_queue = deque(maxlen=self.window_size)
        self.z_queue = deque(maxlen=self.window_size)
        self.yaw_queue = deque(maxlen=self.window_size)

        self.plot_len = 50
        self.data_x, self.data_y, self.data_z, self.data_yaw = deque(maxlen=self.plot_len), deque(maxlen=self.plot_len), deque(maxlen=self.plot_len), deque(maxlen=self.plot_len)

	# 그래프
        threading.Thread(target=self.init_plot_thread, daemon=True).start()

    def info_callback(self, msg):
        if self.intrinsics is None:
            self.intrinsics = {
                'fx': msg.k[0], 'fy': msg.k[4],
                'ppx': msg.k[2], 'ppy': msg.k[5],
                'width': msg.width, 'height': msg.height
            }

    # 서비스 콜백 함수: Degree 변환
    def handle_get_pose(self, request, response):

        self.get_logger().info("[Service] 요청 수신. 2초 대기 후 데이터 전송...")
        time.sleep(2.0)

        data = self.latest_result
        if data["success"]:
            response.success = True
            response.detected_id = int(data["id"])
            
            # 좌표: mm -> m 변환 (Client가 미터 단위를 받아 *1000 하므로 유지)
            response.x = data["x_mm"] / 1000.0
            response.y = data["y_mm"] / 1000.0
            response.z = data["z_mm"] / 1000.0
            
            # [중요] 각도: Degree 그대로 전송 (Client 코드: target_j[5] += pose.rz)
            response.rz = float(data["yaw_deg"]) 
            
            self.get_logger().info(f"[Service] 좌표 전송: X={response.x:.3f}m, Y={response.y:.3f}m, Yaw={data['yaw_deg']:.1f}deg")
        else:
            response.success = False
            response.detected_id = -1
            self.get_logger().warn("[Service] 요청받음, 그러나 감지된 물체 없음")
        return response

    def process_callback(self, color_msg, depth_msg):
        if self.intrinsics is None or self.model is None:
            return

        try:
            color_image = self.cv_bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except:
            return

        # YOLO 실행
        results = self.model.predict(source=color_image, device=self.device, imgsz=640, verbose=False, conf=0.7)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        cls_ids = results[0].boxes.cls.cpu().numpy()

        if len(boxes) == 0:
            self.latest_result["success"] = False

        # 포인트클라우드
        pcd_all, u_map, v_map = self.generate_pointcloud(depth_image)
        valid_idx = pcd_all[:,2] > 0
        pcd_valid = pcd_all[valid_idx]
        u_valid = u_map.reshape(-1)[valid_idx]
        v_valid = v_map.reshape(-1)[valid_idx]

        if len(pcd_valid) < 100:
            self.update_visualization(color_image, np.empty((0,3)))
            return

        # 바닥 제거
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_valid)
        down = pcd_o3d.voxel_down_sample(0.015)
        if len(down.points) < 10:
            self.update_visualization(color_image, np.empty((0,3)))
            return

        plane_model, inliers = down.segment_plane(distance_threshold=self.ground_dist_thresh, ransac_n=3, num_iterations=50)
        dist_all = self.point_to_plane_distance(pcd_valid, plane_model)
        non_ground_mask = dist_all > self.ground_dist_thresh

        ground_removed_pts = pcd_valid[non_ground_mask]
        u_ng = u_valid[non_ground_mask]
        v_ng = v_valid[non_ground_mask]

        # 객체 처리
        for idx, (x1, y1, x2, y2) in enumerate(boxes.astype(int)):
            roi_mask = (u_ng >= x1) & (u_ng < x2) & (v_ng >= y1) & (v_ng < y2)
            roi_pts = ground_removed_pts[roi_mask]
            if len(roi_pts) < 20:
                continue

            centroid = roi_pts.mean(axis=0) * 1000  # m -> mm
            conf_score = confs[idx]
            cls_id = int(cls_ids[idx])

            info_text = f"XYZ:({centroid[0]:7.2f},{centroid[1]:7.2f},{centroid[2]:7.2f}) Conf:{conf_score:.2f}"
            cv2.putText(color_image, info_text, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),2)
            cv2.rectangle(color_image, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.circle(color_image, (int((x1+x2)/2),(int((y1+y2)/2))),5,(0,0,255),-1)

            # 회전 및 데이터 업데이트
            try:
                roi_o3d = o3d.geometry.PointCloud()
                roi_o3d.points = o3d.utility.Vector3dVector(roi_pts)
                obb = roi_o3d.get_oriented_bounding_box()
                roll, pitch, yaw = self.rotation_matrix_to_euler_angles(obb.R)
                yaw_deg = np.degrees(yaw)

                # 이동평균 적용
                self.x_queue.append(centroid[0])
                self.y_queue.append(centroid[1])
                self.z_queue.append(centroid[2])
                self.yaw_queue.append(yaw_deg)

                avg_x = np.mean(self.x_queue)
                avg_y = np.mean(self.y_queue)
                avg_z = np.mean(self.z_queue)
                avg_yaw = np.mean(self.yaw_queue)

                rot_text = f"RPY:({int(np.degrees(roll))},{int(np.degrees(pitch))},{int(avg_yaw)})"
                cv2.putText(color_image, rot_text, (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,255,50),2)

                self.data_x.append(avg_x)
                self.data_y.append(avg_y)
                self.data_z.append(avg_z)
                self.data_yaw.append(avg_yaw)

                self.latest_result = {
                    "success": True,
                    "x_mm": avg_x,
                    "y_mm": avg_y,
                    "z_mm": avg_z,
                    "yaw_deg": avg_yaw,
                    "id": cls_id
                }

            except Exception:
                pass

        self.update_visualization(color_image, ground_removed_pts)

    def update_visualization(self, color_image, points_3d):
        cv2.imshow("YOLO 3D (XYZ + RPY)", color_image)
        cv2.waitKey(1)

        if len(points_3d) > 0:
            self.pcd_vis.points = o3d.utility.Vector3dVector(points_3d)
        else:
            self.pcd_vis.points = o3d.utility.Vector3dVector(np.empty((0,3),dtype=np.float64))

        if not self.view_inited and len(points_3d) > 100:
            self.init_view_to_cloud(self.vis, self.pcd_vis)
            self.view_inited = True

        self.vis.update_geometry(self.pcd_vis)
        self.vis.poll_events()
        self.vis.update_renderer()

    def rotation_matrix_to_euler_angles(self, R):
        sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(R[2,1], R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else:
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        return x,y,z

    def generate_pointcloud(self, depth_image):
        h,w = depth_image.shape
        u,v = np.meshgrid(np.arange(w), np.arange(h))
        depth_m = depth_image*self.depth_scale
        X = (u - self.intrinsics['ppx'])*depth_m/self.intrinsics['fx']
        Y = (v - self.intrinsics['ppy'])*depth_m/self.intrinsics['fy']
        Z = depth_m
        pts = np.dstack((X,Y,Z))
        return pts.reshape(-1,3), u, v

    def point_to_plane_distance(self, pts_xyz, plane_model):
        a,b,c,d = plane_model
        denom = math.sqrt(a*a+b*b+c*c)+1e-12
        return np.abs(a*pts_xyz[:,0]+b*pts_xyz[:,1]+c*pts_xyz[:,2]+d)/denom

    def init_view_to_cloud(self, vis, pcd):
        ctr = vis.get_view_control()
        ctr.set_lookat(pcd.get_center())
        ctr.set_front([0,-0.5,-1])
        ctr.set_up([0,-1,0])
        ctr.set_zoom(0.6)

    def init_plot_thread(self):
        plt.ion()
        self.fig, ((self.ax_x,self.ax_y),(self.ax_z,self.ax_yaw)) = plt.subplots(2,2, figsize=(10,6))
        self.fig.suptitle("Moving Average: X,Y,Z,Yaw")
        self.line_x, = self.ax_x.plot([],[],'r-'); self.ax_x.set_title('X'); self.ax_x.set_ylim(-170,170)
        self.line_y, = self.ax_y.plot([],[],'g-'); self.ax_y.set_title('Y'); self.ax_y.set_ylim(-170,170)
        self.line_z, = self.ax_z.plot([],[],'b-'); self.ax_z.set_title('Z'); self.ax_z.set_ylim(0,800)
        self.line_yaw, = self.ax_yaw.plot([],[],'m-'); self.ax_yaw.set_title('Yaw'); self.ax_yaw.set_ylim(-90,90)

        while rclpy.ok():
            self.update_plot()
            plt.pause(0.1)

    def update_plot(self):
        self.line_x.set_data(range(len(self.data_x)), [v for v in self.data_x])
        self.ax_x.set_xlim(0, self.plot_len)
        self.line_y.set_data(range(len(self.data_y)), [v for v in self.data_y])
        self.ax_y.set_xlim(0, self.plot_len)
        self.line_z.set_data(range(len(self.data_z)), [v for v in self.data_z])
        self.ax_z.set_xlim(0, self.plot_len)
        
        self.line_yaw.set_data(range(len(self.data_yaw)), list(self.data_yaw))
        self.ax_yaw.set_xlim(0, self.plot_len)
        
        self.ax_x.set_ylabel("X (mm)")
        self.ax_y.set_ylabel("Y (mm)")
        self.ax_z.set_ylabel("Z (mm)")
        self.ax_yaw.set_ylabel("Yaw (deg)")
        
        self.fig.canvas.draw_idle()

def main(args=None):
    rclpy.init(args=args)
    node = Yolo3DNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (Ctrl+C) Detected')
    except Exception as e:
        node.get_logger().error(f'Exception in main: {e}')
    finally:
        # 안전한 종료 처리
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        
        # 시각화 창 닫기
        try:
            node.vis.destroy_window()
            cv2.destroyAllWindows()
            plt.close('all') # 그래프 창 닫기
        except:
            pass

if __name__ == '__main__':
    main()