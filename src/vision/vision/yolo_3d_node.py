#!/usr/bin/env python3
import os
import time
import math
import yaml
from collections import deque

import cv2
import numpy as np
import open3d as o3d

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from ament_index_python.packages import get_package_share_directory

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters
from ultralytics import YOLO

from msgs_pkg.srv import GetObjectPose


class Yolo3DNode(Node):
    def __init__(self):
        super().__init__('yolo_3d_node')

        # --------------------------------------------------
        # 1. Parameters
        # --------------------------------------------------
        pkg_path = get_package_share_directory('vision')

        default_model_path = os.path.join(
            pkg_path, 'yolo_models', '0128_train', 'weights', 'best.pt'
        )
        default_yaml_path = os.path.join(
            pkg_path, 'config', 'models.yaml'
        )

        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('yaml_path', default_yaml_path)

        self.model_path = self.get_parameter('model_path').value
        self.yaml_path = self.get_parameter('yaml_path').value
        self.declare_parameter('model_device', 'cuda:0')
        self.declare_parameter('ground_dist_thresh', 0.01)
        self.declare_parameter('default_conf', 0.7)
        self.declare_parameter('track_match_px', 80.0)
        self.declare_parameter('track_timeout_sec', 1.0)
        self.declare_parameter('window_size', 15)
        self.declare_parameter('result_timeout_sec', 0.5)

        model_path = self.get_parameter('model_path').value
        yaml_path = self.get_parameter('yaml_path').value
        self.device = self.get_parameter('model_device').value
        self.ground_dist_thresh = float(self.get_parameter('ground_dist_thresh').value)
        self.default_conf = float(self.get_parameter('default_conf').value)
        self.track_match_px = float(self.get_parameter('track_match_px').value)
        self.track_timeout_sec = float(self.get_parameter('track_timeout_sec').value)
        self.window_size = int(self.get_parameter('window_size').value)
        self.result_timeout_sec = float(self.get_parameter('result_timeout_sec').value)

        # --------------------------------------------------
        # 2. Load model / configs
        # --------------------------------------------------
        self.model = YOLO(model_path)
        self.object_configs = self.load_models_from_yaml(yaml_path)

        if not self.object_configs:
            self.get_logger().warn("No valid object configs loaded from YAML.")

        # --------------------------------------------------
        # 3. Open3D visualization
        # --------------------------------------------------
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="3D Pose Estimation View", width=960, height=720)

        self.scene_pcd_vis = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.scene_pcd_vis)

        self.track_axis_map = {}      # track_id -> axis mesh
        self.track_axis_trans = {}    # track_id -> applied transform
        self.track_states = {}        # track_id -> state dict
        self.track_counter = 0

        self.view_inited = False

        # --------------------------------------------------
        # 4. States
        # --------------------------------------------------
        self.intrinsics = None
        self.depth_scale = 0.001
        self.clear_latest_result()

        # --------------------------------------------------
        # 5. ROS comms
        # --------------------------------------------------
        self.cv_bridge = CvBridge()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.color_sub = message_filters.Subscriber(
            self, Image, '/camera/camera/color/image_raw', qos_profile=qos
        )
        self.depth_sub = message_filters.Subscriber(
            self, Image, '/camera/camera/aligned_depth_to_color/image_raw', qos_profile=qos
        )
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], queue_size=10, slop=0.1
        )
        self.sync.registerCallback(self.process_callback)

        self.info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/aligned_depth_to_color/camera_info',
            self.info_callback,
            10
        )

        self.srv = self.create_service(
            GetObjectPose,
            '/vision/get_object_pose',
            self.handle_get_pose
        )

        self.get_logger().info("Yolo3DNode initialized.")

    # ======================================================
    # Helpers
    # ======================================================
    def clear_latest_result(self):
        self.latest_result = {
            "success": False,
            "x_mm": 0.0,
            "y_mm": 0.0,
            "z_mm": 0.0,
            "roll_deg": 0.0,
            "pitch_deg": 0.0,
            "yaw_deg": 0.0,
            "id": -1,
            "track_id": None,
            "stamp": 0.0,
        }

    # ======================================================
    # YAML / model loading
    # ======================================================
    def load_stl_model(self, path, number_of_points=3000):
        if not os.path.exists(path):
            self.get_logger().error(f"STL file not found: {path}")
            return None

        mesh = o3d.io.read_triangle_mesh(path)
        if mesh.is_empty():
            self.get_logger().error(f"Empty mesh: {path}")
            return None

        pcd = mesh.sample_points_uniformly(number_of_points=number_of_points)
        pcd.translate(-pcd.get_center())
        pcd.estimate_normals()
        return pcd

    def normalize_symmetry(self, sym_val):
        if isinstance(sym_val, dict):
            return {
                "roll": float(sym_val.get("roll", 360.0)),
                "pitch": float(sym_val.get("pitch", 360.0)),
                "yaw": float(sym_val.get("yaw", 360.0)),
            }
        s = float(sym_val) if sym_val is not None else 360.0
        return {"roll": s, "pitch": s, "yaw": s}

    def load_models_from_yaml(self, yaml_path):
        if not os.path.exists(yaml_path):
            self.get_logger().error(f"YAML not found: {yaml_path}")
            return {}

        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        configs = {}
        objects = data.get("objects", [])
        if not isinstance(objects, list):
            self.get_logger().error("YAML format error: 'objects' must be a list.")
            return {}

        for obj in objects:
            try:
                obj_id = int(obj["id"])
                name = str(obj.get("name", f"object_{obj_id}"))
                stl_path = str(obj["stl_path"])
                thickness = float(obj.get("thickness", 0.0))
                symmetry = self.normalize_symmetry(obj.get("symmetry", 360.0))
                icp_dist = float(obj.get("icp_dist", 0.05))
                voxel = float(obj.get("voxel", 0.005))
                min_pts = int(obj.get("min_pts", 50))
                color = obj.get("color", [0.7, 0.7, 0.7])
                axis_size = float(obj.get("axis_size", 0.08))
                conf = float(obj.get("conf", self.default_conf))
                normal_radius = float(obj.get("normal_radius", 0.01))
                normal_max_nn = int(obj.get("normal_max_nn", 30))
                sample_points = int(obj.get("sample_points", 3000))

                if len(color) != 3:
                    self.get_logger().warn(f"Invalid color for id={obj_id}. Use default.")
                    color = [0.7, 0.7, 0.7]

                source_pcd = self.load_stl_model(stl_path, number_of_points=sample_points)
                if source_pcd is None:
                    self.get_logger().warn(f"Skipping object id={obj_id} due to STL load failure.")
                    continue

                source_pcd.paint_uniform_color(color)

                configs[obj_id] = {
                    "id": obj_id,
                    "name": name,
                    "stl_path": stl_path,
                    "source_pcd": source_pcd,
                    "half_thickness": thickness / 2.0,
                    "symmetry": symmetry,
                    "icp_dist": icp_dist,
                    "voxel": voxel,
                    "min_pts": min_pts,
                    "color": [float(color[0]), float(color[1]), float(color[2])],
                    "axis_size": axis_size,
                    "conf": conf,
                    "normal_radius": normal_radius,
                    "normal_max_nn": normal_max_nn,
                }

                self.get_logger().info(
                    f"Loaded object: id={obj_id}, name={name}, "
                    f"icp_dist={icp_dist}, min_pts={min_pts}, color={color}"
                )
            except Exception as e:
                self.get_logger().error(f"Failed to parse object entry: {obj}, error={e}")

        return configs

    # ======================================================
    # ROS callbacks / service
    # ======================================================
    def info_callback(self, msg):
        if self.intrinsics is None:
            self.intrinsics = {
                'fx': msg.k[0],
                'fy': msg.k[4],
                'ppx': msg.k[2],
                'ppy': msg.k[5],
                'width': msg.width,
                'height': msg.height
            }
            self.get_logger().info("Camera intrinsics received.")

    def handle_get_pose(self, request, response):
        d = self.latest_result
        now = time.time()

        if d["success"] and (now - d["stamp"] < self.result_timeout_sec):
            response.success = True
            response.detected_id = int(d["id"])
            response.x = d["x_mm"] / 1000.0
            response.y = d["y_mm"] / 1000.0
            response.z = d["z_mm"] / 1000.0
            response.rz = float(d["yaw_deg"])
        else:
            response.success = False

        return response

    # ======================================================
    # Track management
    # ======================================================
    def create_empty_track_state(self, track_id, cls_id, bbox_center):
        return {
            "track_id": track_id,
            "cls_id": cls_id,
            "bbox_center": bbox_center,
            "last_seen": time.time(),
            "pos_queues": {k: deque(maxlen=self.window_size) for k in ['x', 'y', 'z']},
            "angle_queues": {
                'roll': {'sin': deque(maxlen=self.window_size), 'cos': deque(maxlen=self.window_size)},
                'pitch': {'sin': deque(maxlen=self.window_size), 'cos': deque(maxlen=self.window_size)},
                'yaw': {'sin': deque(maxlen=self.window_size), 'cos': deque(maxlen=self.window_size)},
            },
            "prev_angles": {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
            "latest_result": {
                "success": False,
                "x_mm": 0.0,
                "y_mm": 0.0,
                "z_mm": 0.0,
                "roll_deg": 0.0,
                "pitch_deg": 0.0,
                "yaw_deg": 0.0,
                "id": cls_id,
                "track_id": track_id,
                "stamp": 0.0,
            }
        }

    def create_axis_for_track(self, track_id, cls_id):
        cfg = self.object_configs[cls_id]
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=cfg["axis_size"])
        self.vis.add_geometry(axis)
        self.track_axis_map[track_id] = axis
        self.track_axis_trans[track_id] = np.eye(4)

    def match_or_create_track(self, cls_id, bbox_center):
        now = time.time()
        candidate_track = None
        best_dist = float('inf')

        for track_id, state in self.track_states.items():
            if state["cls_id"] != cls_id:
                continue
            if now - state["last_seen"] > self.track_timeout_sec:
                continue

            prev_cx, prev_cy = state["bbox_center"]
            cx, cy = bbox_center
            dist = math.hypot(cx - prev_cx, cy - prev_cy)

            if dist < best_dist:
                best_dist = dist
                candidate_track = track_id

        if candidate_track is not None and best_dist <= self.track_match_px:
            state = self.track_states[candidate_track]
            state["bbox_center"] = bbox_center
            state["last_seen"] = now
            return candidate_track

        new_track_id = f"{cls_id}_{self.track_counter}"
        self.track_counter += 1

        self.track_states[new_track_id] = self.create_empty_track_state(
            new_track_id, cls_id, bbox_center
        )
        self.create_axis_for_track(new_track_id, cls_id)

        return new_track_id

    def cleanup_stale_tracks(self):
        now = time.time()
        stale_ids = []

        for track_id, state in self.track_states.items():
            if now - state["last_seen"] > self.track_timeout_sec:
                stale_ids.append(track_id)

        for track_id in stale_ids:
            if track_id in self.track_axis_map:
                try:
                    self.vis.remove_geometry(self.track_axis_map[track_id], reset_bounding_box=False)
                except Exception:
                    pass
                del self.track_axis_map[track_id]

            if track_id in self.track_axis_trans:
                del self.track_axis_trans[track_id]

            del self.track_states[track_id]

    # ======================================================
    # Math / filters
    # ======================================================
    def stabilize_angle_symmetry(self, current, reference, symmetry_step_deg=360.0):
        if symmetry_step_deg <= 0.0 or symmetry_step_deg >= 360.0:
            return current
        step = math.radians(symmetry_step_deg)
        n = round((reference - current) / step)
        return current + n * step

    def rotation_matrix_to_euler_angles(self, R):
        sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        if sy > 1e-6:
            roll = math.atan2(R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            roll = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = 0.0
        return roll, pitch, yaw

    def enforce_z_not_opposite_camera(self, R):
        z_ref = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        x_axis = R[:, 0].copy()
        y_axis = R[:, 1].copy()
        z_axis = R[:, 2].copy()

        if np.dot(z_axis, z_ref) < 0:
            z_axis = -z_axis
            x_axis = -x_axis

            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-12)

            x_axis = np.cross(y_axis, z_axis)
            x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-12)

        R_fixed = np.column_stack((x_axis, y_axis, z_axis))
        return R_fixed

    def generate_pointcloud(self, depth_image):
        h, w = depth_image.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        depth_m = depth_image.astype(np.float32) * self.depth_scale

        X = (u - self.intrinsics['ppx']) * depth_m / self.intrinsics['fx']
        Y = (v - self.intrinsics['ppy']) * depth_m / self.intrinsics['fy']
        Z = depth_m

        return np.dstack((X, Y, Z)).reshape(-1, 3), u, v

    def point_to_plane_distance(self, pts, plane):
        a, b, c, d = plane
        denom = math.sqrt(a * a + b * b + c * c) + 1e-12
        return np.abs(a * pts[:, 0] + b * pts[:, 1] + c * pts[:, 2] + d) / denom

    def update_track_filter_and_result(self, track_id, cls_id, final_trans):
        state = self.track_states[track_id]
        cfg = self.object_configs[cls_id]

        curr_xyz = final_trans[:3, 3] * 1000.0
        for i, k in enumerate(['x', 'y', 'z']):
            state["pos_queues"][k].append(curr_xyz[i])

        avg_x, avg_y, avg_z = [
            np.median(state["pos_queues"][k]) for k in ['x', 'y', 'z']
        ]

        r_raw, p_raw, y_raw = self.rotation_matrix_to_euler_angles(final_trans[:3, :3])

        sym = cfg["symmetry"]
        r_stable = self.stabilize_angle_symmetry(r_raw, state["prev_angles"]["roll"], sym["roll"])
        p_stable = self.stabilize_angle_symmetry(p_raw, state["prev_angles"]["pitch"], sym["pitch"])
        y_stable = self.stabilize_angle_symmetry(y_raw, state["prev_angles"]["yaw"], sym["yaw"])

        curr_angles = {
            "roll": r_stable,
            "pitch": p_stable,
            "yaw": y_stable,
        }

        avgs_deg = {}
        for k in ["roll", "pitch", "yaw"]:
            state["angle_queues"][k]["sin"].append(math.sin(curr_angles[k]))
            state["angle_queues"][k]["cos"].append(math.cos(curr_angles[k]))

            med_sin = np.median(state["angle_queues"][k]["sin"])
            med_cos = np.median(state["angle_queues"][k]["cos"])
            res_rad = math.atan2(med_sin, med_cos)

            avgs_deg[k] = np.degrees(res_rad)
            state["prev_angles"][k] = res_rad

        result = {
            "success": True,
            "id": cls_id,
            "track_id": track_id,
            "x_mm": float(avg_x),
            "y_mm": float(avg_y),
            "z_mm": float(avg_z),
            "roll_deg": float(avgs_deg["roll"]),
            "pitch_deg": float(avgs_deg["pitch"]),
            "yaw_deg": float(avgs_deg["yaw"]),
            "stamp": time.time(),
        }

        state["latest_result"] = result
        self.latest_result = result

        return result

    def update_track_axis(self, track_id, final_trans):
        if track_id not in self.track_axis_map:
            return

        axis_mesh = self.track_axis_map[track_id]
        prev_trans = self.track_axis_trans[track_id]

        try:
            axis_mesh.transform(np.linalg.inv(prev_trans))
            axis_mesh.transform(final_trans)
            self.track_axis_trans[track_id] = final_trans.copy()
            self.vis.update_geometry(axis_mesh)
        except Exception as e:
            self.get_logger().warn(f"Axis update failed for {track_id}: {e}")

    # ======================================================
    # Main processing
    # ======================================================
    def process_callback(self, color_msg, depth_msg):
        if self.intrinsics is None or not self.object_configs:
            return

        try:
            color_image = self.cv_bridge.imgmsg_to_cv2(color_msg, 'bgr8')
            depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        except Exception as e:
            self.get_logger().warn(f"CV bridge conversion failed: {e}")
            return

        # YOLO inference
        try:
            results = self.model.predict(
                source=color_image,
                device=self.device,
                conf=self.default_conf,
                verbose=False
            )
        except Exception as e:
            self.get_logger().error(f"YOLO inference failed: {e}")
            return

        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            self.cleanup_stale_tracks()
            self.clear_latest_result()
            self.update_visualization(color_image, None)
            return

        boxes = results[0].boxes.xyxy.cpu().numpy()
        cls_ids = results[0].boxes.cls.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes.conf is not None else np.ones(len(boxes))

        # Depth -> point cloud
        pcd_all, u_map, v_map = self.generate_pointcloud(depth_image)
        valid_idx = np.isfinite(pcd_all[:, 2]) & (pcd_all[:, 2] > 0)
        if np.count_nonzero(valid_idx) < 100:
            self.cleanup_stale_tracks()
            self.clear_latest_result()
            self.update_visualization(color_image, None)
            return

        pcd_valid = pcd_all[valid_idx]
        u_valid = u_map.reshape(-1)[valid_idx]
        v_valid = v_map.reshape(-1)[valid_idx]

        # Ground removal
        ground_removed_pts = pcd_valid
        u_ng = u_valid
        v_ng = v_valid

        try:
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(pcd_valid)

            down = pcd_o3d.voxel_down_sample(0.005)
            if len(down.points) >= 50:
                plane_model, _ = down.segment_plane(
                    distance_threshold=self.ground_dist_thresh,
                    ransac_n=3,
                    num_iterations=50
                )
                non_ground_mask = self.point_to_plane_distance(pcd_valid, plane_model) > self.ground_dist_thresh
                ground_removed_pts = pcd_valid[non_ground_mask]
                u_ng = u_valid[non_ground_mask]
                v_ng = v_valid[non_ground_mask]
        except Exception as e:
            self.get_logger().warn(f"Ground removal failed. Use original valid cloud. error={e}")

        # --------------------------------------------------
        # Scene color initialization
        # --------------------------------------------------
        merged_pcd = None
        scene_points = ground_removed_pts
        scene_colors = None
        frame_has_valid_pose = False

        if len(scene_points) > 0:
            scene_colors = np.tile(
                np.array([[0.35, 0.35, 0.35]], dtype=np.float64),
                (len(scene_points), 1)
            )

        # --------------------------------------------------
        # Per detection
        # --------------------------------------------------
        for idx, (x1, y1, x2, y2) in enumerate(boxes.astype(int)):
            cls_id = int(cls_ids[idx])
            det_conf = float(confs[idx])

            if cls_id not in self.object_configs:
                continue

            cfg = self.object_configs[cls_id]
            if det_conf < cfg["conf"]:
                continue

            roi_mask = (u_ng >= x1) & (u_ng < x2) & (v_ng >= y1) & (v_ng < y2)
            roi_pts = ground_removed_pts[roi_mask]

            if len(roi_pts) < cfg["min_pts"]:
                continue

            # YAML color를 scene color에 직접 반영
            if scene_colors is not None:
                roi_color = np.array(cfg["color"], dtype=np.float64)
                scene_colors[roi_mask] = roi_color

            bbox_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            track_id = self.match_or_create_track(cls_id, bbox_center)
            state = self.track_states[track_id]
            state["bbox_center"] = bbox_center
            state["last_seen"] = time.time()

            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(roi_pts)

            try:
                target_pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=cfg["normal_radius"],
                        max_nn=cfg["normal_max_nn"]
                    )
                )

                obb = target_pcd.get_oriented_bounding_box()

                initial_trans = np.eye(4)
                initial_trans[:3, :3] = obb.R
                center = np.array(obb.center, dtype=np.float64)
                center[2] += cfg["half_thickness"]
                initial_trans[:3, 3] = center

                reg_p2p = o3d.pipelines.registration.registration_icp(
                    cfg["source_pcd"],
                    target_pcd,
                    cfg["icp_dist"],
                    initial_trans,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane()
                )

                final_trans = np.array(reg_p2p.transformation, dtype=np.float64, copy=True)
                final_trans[:3, :3] = self.enforce_z_not_opposite_camera(final_trans[:3, :3])

                result = self.update_track_filter_and_result(track_id, cls_id, final_trans)
                frame_has_valid_pose = True
                self.update_track_axis(track_id, final_trans)

                color_bgr = (
                    int(cfg["color"][2] * 255),
                    int(cfg["color"][1] * 255),
                    int(cfg["color"][0] * 255)
                )

                cv2.rectangle(color_image, (x1, y1), (x2, y2), color_bgr, 2)

                name_txt = f"{cfg['name']} (ID:{cls_id}, T:{track_id})"
                txt_xyz = f"XYZ(mm): {result['x_mm']:.1f}, {result['y_mm']:.1f}, {result['z_mm']:.1f}"
                txt_rpy = f"RPY(deg): {result['roll_deg']:.1f}, {result['pitch_deg']:.1f}, {result['yaw_deg']:.1f}"

                cv2.putText(
                    color_image, name_txt, (x1, max(20, y1 - 55)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2
                )
                cv2.putText(
                    color_image, txt_xyz, (x1, max(20, y1 - 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2
                )
                cv2.putText(
                    color_image, txt_rpy, (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2
                )

            except Exception as e:
                self.get_logger().warn(
                    f"Pose estimation failed for cls_id={cls_id}, track_id={track_id}: {e}"
                )
                continue

        self.cleanup_stale_tracks()

        if not frame_has_valid_pose:
            self.clear_latest_result()

        # --------------------------------------------------
        # Build final colored scene point cloud
        # --------------------------------------------------
        if scene_colors is not None and len(scene_points) > 0:
            try:
                merged_pcd = o3d.geometry.PointCloud()
                merged_pcd.points = o3d.utility.Vector3dVector(scene_points)
                merged_pcd.colors = o3d.utility.Vector3dVector(scene_colors)
            except Exception as e:
                self.get_logger().warn(f"Merged point cloud build failed: {e}")
                merged_pcd = None

        self.update_visualization(color_image, merged_pcd)

    # ======================================================
    # Visualization
    # ======================================================
    def update_visualization(self, color_image, merged_pcd=None):
        cv2.imshow("Final Pose Estimation", color_image)
        cv2.waitKey(1)

        try:
            if merged_pcd is not None and len(merged_pcd.points) > 0:
                self.scene_pcd_vis.points = merged_pcd.points
                self.scene_pcd_vis.colors = merged_pcd.colors
                self.vis.update_geometry(self.scene_pcd_vis)

            # 처음 한 번만 전체가 보이도록 자동 카메라 설정
            if not self.view_inited:
                self.vis.reset_view_point(True)
                self.view_inited = True

            self.vis.poll_events()
            self.vis.update_renderer()

        except Exception as e:
            self.get_logger().warn(f"Open3D visualization update failed: {e}")


def main():
    rclpy.init()
    node = Yolo3DNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
