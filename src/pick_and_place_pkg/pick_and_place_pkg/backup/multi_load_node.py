# #!/usr/bin/env python3           // 바닥 인식해서 물체 없으면 종료
# import threading
# import time
# import numpy as np

# import rclpy
# from rclpy.node import Node
# from rclpy.executors import MultiThreadedExecutor
# from rclpy.callback_groups import ReentrantCallbackGroup

# import rbpodo as rb
# from std_srvs.srv import Trigger
# from msgs_pkg.srv import GetObjectPose
# from msgs_pkg.srv import RunWS

# # ===============================
# # 설정값 (상수 선언)
# # ===============================
# ROBOT_IP = "10.0.2.7"

# # Joint poses (기본)
# HOME_JOINT_DEG = np.array([-90.0, 0.0, 90.0, 0.0, 90.0, 0.0])
# POSE_FINAL     = np.array([-90.0, -35.0, 125.0, 0.0, 90.0, 0.0])

# # CARGO 위치 목록 (1, 2, 3번 순서)
# POSE_CARGO1 = np.array([-65.31, -16.33, -31.69, -0.01, -131.95, 24.62])
# POSE_CARGO2 = np.array([-93.25, -7.91, -42.05, 0.0, -130.0, -3.32])
# POSE_CARGO3 = np.array([-117.67, -16.88, -30.94, 0.04, -132.13, -27.71])

# CARGO_POSES = [POSE_CARGO1, POSE_CARGO2, POSE_CARGO3]

# # Camera → TCP calibration offsets (mm)
# CAM_TO_TCP_OFFSET_X_MM = -51.0
# CAM_TO_TCP_OFFSET_Y_MM =  32.0

# # Z movement (mm)
# Z_APPROACH_MM = 365.0
# Z_DOWN_MM      = 62.0
# Z_UP_MM        = -70.0

# # 속도 설정
# J_VEL, J_ACC = 255, 255
# L_VEL, L_ACC = 500, 800

# MOVE_START_TIMEOUT_SEC = 1.0
# RETRY_SLEEP_SEC = 1.0

# # ===============================
# # Load Node (Service Mode)
# # ===============================
# class LoadNode(Node):

#     def __init__(self):
#         super().__init__("load_node")
#         self.get_logger().info("✅ Load Node Started (Adaptive Logic Applied)")

#         # 멀티스레드 콜백 그룹 (교착 상태 방지)
#         self.callback_group = ReentrantCallbackGroup()

#         # 로봇 연결 및 초기화
#         self.robot = rb.Cobot(ROBOT_IP)
#         self.rc = rb.ResponseCollector()
#         self.robot.set_operation_mode(self.rc, rb.OperationMode.Real)
#         self.robot.set_speed_bar(self.rc, 1.0)

#         # Clients 설정
#         self.open_client = self.create_client(Trigger, "/gripper/open", callback_group=self.callback_group)
#         self.grip_client = self.create_client(Trigger, "/gripper/grip", callback_group=self.callback_group)
#         self.pose_client = self.create_client(GetObjectPose, "/vision/get_object_pose", callback_group=self.callback_group)

#         self.get_logger().info("Waiting for services...")
#         self.open_client.wait_for_service()
#         self.grip_client.wait_for_service()
#         self.pose_client.wait_for_service()

#         self._busy_lock = threading.Lock()
#         self._busy = False

#         # Service server
#         self.srv = self.create_service(RunWS, "/task/load3", self.cb_load3, callback_group=self.callback_group)
#         self.get_logger().info("🟦 Ready for Coordinator command.")

#     def cb_load3(self, req: RunWS.Request, res: RunWS.Response):
#         with self._busy_lock:
#             if self._busy:
#                 res.success = False
#                 res.message = "busy"
#                 return res
#             self._busy = True

#         try:
#             # 적재 시퀀스 실행
#             ok = self.sequence_once()
#             res.success = bool(ok) # 물체가 없어서 끝난 경우도 True 반환
#             res.message = "ok"
#             return res
#         finally:
#             with self._busy_lock:
#                 self._busy = False

#     def sequence_once(self) -> bool:
#         self.get_logger().info("▶ START ADAPTIVE-PICK SEQUENCE")
#         if not self.go_home(): return False

#         cargo_idx = 0
#         all_done_successfully = True 

#         while rclpy.ok():
#             if cargo_idx >= len(CARGO_POSES):
#                 self.get_logger().info("🎉 All Cargo Slots Filled (3/3).")
#                 break

#             self.get_logger().info(f"🚀 Processing Cargo #{cargo_idx + 1}")

#             # run_once 내부에서 비전이 안 잡히면 False를 반환함
#             if not self.run_once(CARGO_POSES[cargo_idx]):
#                 # 🎯 핵심 로직: 비전 인식 실패 = '바닥에 물체 없음' = '작업 완료'
#                 self.get_logger().info("ℹ️ No more markers detected. Considering task COMPLETE.")
#                 # 에러가 아니므로 all_done_successfully는 True 유지
#                 break 

#             cargo_idx += 1
#             time.sleep(RETRY_SLEEP_SEC)

#         # 작업을 마치거나 중단되면 무조건 최종 포즈로 이동
#         self.go_final_pose()
#         return all_done_successfully

#     def run_once(self, cargo_pose_target) -> bool:
#         # 1단계: 비전 호출 (5초 타임아웃)
#         pose = self.call_pose(timeout_sec=5.0) 
#         if pose is None:
#             return False # 물체가 없음을 알림

#         self.get_logger().info(f"🎯 Target Locked ID: {pose.detected_id}")

#         # 2단계: Pick & Place 동작
#         if not self.open_gripper(): return False
#         if not self.align_yaw(pose): return False
#         if not self.align_xy(pose): return False
#         if not self.approach_z(): return False
#         if not self.grasp(): return False
#         if not self.go_home(): return False
#         if not self.go_cargo(cargo_pose_target): return False
#         if not self.move_z(Z_DOWN_MM, "CARGO_DOWN"): return False
#         if not self.open_gripper(): return False
#         if not self.move_z(Z_UP_MM, "CARGO_UP"): return False
#         if not self.go_home(): return False
#         return True

#     # --- 동작 및 통신 유틸리티 ---
#     def go_home(self) -> bool:
#         self.robot.move_j(self.rc, HOME_JOINT_DEG, J_VEL, J_ACC)
#         return self.wait_move("HOME")

#     def go_cargo(self, target_pose) -> bool:
#         self.robot.move_j(self.rc, target_pose, J_VEL, J_ACC)
#         return self.wait_move("CARGO")

#     def go_final_pose(self) -> bool:
#         self.get_logger().info("➡️ FINAL RETRACT")
#         self.robot.move_j(self.rc, POSE_FINAL, J_VEL, J_ACC)
#         return self.wait_move("FINAL")

#     def align_yaw(self, pose) -> bool:
#         target = HOME_JOINT_DEG.copy()
#         target[5] += float(pose.rz)
#         self.robot.move_j(self.rc, target, J_VEL, J_ACC)
#         return self.wait_move("ALIGN_YAW")

#     def align_xy(self, pose) -> bool:
#         dx_mm = -(pose.x * 1000.0) + CAM_TO_TCP_OFFSET_Y_MM
#         dy_mm =  (pose.y * 1000.0) + CAM_TO_TCP_OFFSET_X_MM
#         self.robot.move_l_rel(self.rc, np.array([dy_mm, dx_mm, 0.0, 0.0, 0.0, 0.0], dtype=float), L_VEL, L_ACC, rb.ReferenceFrame.Tool)
#         return self.wait_move("ALIGN_XY")

#     def approach_z(self) -> bool:
#         self.robot.move_l_rel(self.rc, np.array([0.0, 0.0, Z_APPROACH_MM, 0.0, 0.0, 0.0], dtype=float), L_VEL, L_ACC, rb.ReferenceFrame.Tool)
#         return self.wait_move("APPROACH_Z")

#     def move_z(self, dz_mm: float, name: str) -> bool:
#         self.robot.move_l_rel(self.rc, np.array([0.0, 0.0, dz_mm, 0.0, 0.0, 0.0], dtype=float), L_VEL, L_ACC, rb.ReferenceFrame.Tool)
#         return self.wait_move(name)

#     def open_gripper(self) -> bool: return self.call_gripper(self.open_client, "OPEN")
#     def grasp(self) -> bool: return self.call_gripper(self.grip_client, "GRIP")

#     def call_pose(self, timeout_sec=5.0):
#         req = GetObjectPose.Request()
#         future = self.pose_client.call_async(req)
#         start_t = time.time()
#         while rclpy.ok() and (time.time() - start_t < timeout_sec):
#             if future.done():
#                 res = future.result()
#                 if res and res.success: return res
#                 break
#             time.sleep(0.05)
#         return None

#     def call_gripper(self, client, name, timeout_sec=5.0) -> bool:
#         req = Trigger.Request()
#         future = client.call_async(req)
#         start_t = time.time()
#         while rclpy.ok() and (time.time() - start_t < timeout_sec):
#             if future.done():
#                 res = future.result()
#                 if res and res.success: return True
#                 break
#             time.sleep(0.05)
#         return False

#     def wait_move(self, name: str) -> bool:
#         if self.robot.wait_for_move_started(self.rc, MOVE_START_TIMEOUT_SEC).is_success():
#             self.robot.wait_for_move_finished(self.rc)
#             return True
#         return False

# def main(args=None):
#     rclpy.init(args=args)
#     node = LoadNode()
#     executor = MultiThreadedExecutor(num_threads=8)
#     executor.add_node(node)
#     try:
#         executor.spin()
#     except KeyboardInterrupt:
#         pass
#     finally:
#         if rclpy.ok():
#             node.destroy_node()
#             rclpy.shutdown()

# if __name__ == "__main__":
#     main()



# #!/usr/bin/env python3            // 바닥 인식해서 물체 없으면 종료 + 몇개 집었는지 기억
# import threading
# import time
# import os
# import numpy as np

# import rclpy
# from rclpy.node import Node
# from rclpy.executors import MultiThreadedExecutor
# from rclpy.callback_groups import ReentrantCallbackGroup

# import rbpodo as rb
# from std_srvs.srv import Trigger
# from msgs_pkg.srv import GetObjectPose, RunWS

# # 설정값
# ROBOT_IP = "10.0.2.7"
# COUNT_FILE = "/tmp/loaded_count.txt"

# HOME_JOINT_DEG = np.array([-90.0, 0.0, 90.0, 0.0, 90.0, 0.0])
# POSE_FINAL = np.array([-90.0, -35.0, 125.0, 0.0, 90.0, 0.0])

# CARGO_POSES = [
#     np.array([-65.31, -16.33, -31.69, -0.01, -131.95, 24.62]),
#     np.array([-93.25, -7.91, -42.05, 0.0, -130.0, -3.32]),
#     np.array([-117.67, -16.88, -30.94, 0.04, -132.13, -27.71])
# ]

# CAM_TO_TCP_OFFSET_X_MM = -51.0
# CAM_TO_TCP_OFFSET_Y_MM = 32.0
# Z_APPROACH_MM = 365.0
# Z_DOWN_MM = 62.0
# Z_UP_MM = -70.0

# J_VEL, J_ACC = 255, 255
# L_VEL, L_ACC = 500, 800

# class LoadNode(Node):
#     def __init__(self):
#         super().__init__("load_node")
#         self.get_logger().info("✅ Load Node Started (Adaptive Fixed)")

#         self.callback_group = ReentrantCallbackGroup()
        
#         try:
#             self.robot = rb.Cobot(ROBOT_IP)
#             self.rc = rb.ResponseCollector()
#             self.robot.set_operation_mode(self.rc, rb.OperationMode.Real)
#             self.get_logger().info("🤖 Robot Connected")
#         except Exception as e:
#             self.get_logger().error(f"❌ Connection Error: {e}")

#         self.open_client = self.create_client(Trigger, "/gripper/open", callback_group=self.callback_group)
#         self.grip_client = self.create_client(Trigger, "/gripper/grip", callback_group=self.callback_group)
#         self.pose_client = self.create_client(GetObjectPose, "/vision/get_object_pose", callback_group=self.callback_group)
#         self.srv = self.create_service(RunWS, "/task/load3", self.cb_load3, callback_group=self.callback_group)

#         self._busy_lock = threading.Lock()
#         self._busy = False

#     def cb_load3(self, req, res):
#         with self._busy_lock:
#             if self._busy: return res
#             self._busy = True
#         try:
#             count = self.sequence_load()
#             res.success = True
#             res.message = f"Loaded {count} items"
#             return res
#         finally:
#             with self._busy_lock: self._busy = False

#     def sequence_load(self):
#         self.get_logger().info("▶ START ADAPTIVE LOAD SEQUENCE")
#         self.call_gripper(self.open_client, "OPEN")
#         self.go_pose_j(HOME_JOINT_DEG, "HOME")

#         success_count = 0
#         for i in range(len(CARGO_POSES)):
#             self.get_logger().info(f"🚀 Slot #{i+1} Attempting...")
            
#             # run_once가 False를 반환하면 물체가 없다는 뜻입니다.
#             if self.run_once(CARGO_POSES[i]):
#                 success_count += 1
#                 self.get_logger().info(f"✅ Slot #{i+1} Loading Success.")
#             else:
#                 self.get_logger().warn("ℹ️ No more markers detected. Breaking loop now.")
#                 break 
        
#         # 개수 기록
#         try:
#             with open(COUNT_FILE, "w") as f:
#                 f.write(str(success_count))
#             self.get_logger().info(f"💾 Saved count: {success_count}")
#         except Exception as e:
#             self.get_logger().error(f"❌ File Write Error: {e}")

#         # 🎯 루프 탈출 후 즉시 최종 포즈 이동
#         self.get_logger().info("➡️ Final Retract to POSE_FINAL")
#         self.go_final_pose()
        
#         return success_count

#     def run_once(self, cargo_target):
#         # 비전 호출
#         pose = self.call_pose(5.0)
#         if pose is None: 
#             return False # 여기서 False를 주어야 sequence_load의 루프가 깨집니다.

#         # ---------------- 동작 시퀀스 ----------------
#         target_j = HOME_JOINT_DEG.copy()
#         target_j[5] += float(pose.rz)
#         self.go_pose_j(target_j, "ALIGN_YAW")

#         dx = -(pose.x * 1000.0) + CAM_TO_TCP_OFFSET_Y_MM
#         dy =  (pose.y * 1000.0) + CAM_TO_TCP_OFFSET_X_MM
#         self.robot.move_l_rel(self.rc, np.array([dy, dx, 0.0, 0.0, 0.0, 0.0]), L_VEL, L_ACC, rb.ReferenceFrame.Tool)
#         self.wait_move("ALIGN_XY")

#         self.robot.move_l_rel(self.rc, np.array([0.0, 0.0, Z_APPROACH_MM, 0.0, 0.0, 0.0]), L_VEL, L_ACC, rb.ReferenceFrame.Tool)
#         self.wait_move("APPROACH")

#         if not self.call_gripper(self.grip_client, "GRIP"): return False
        
#         self.go_pose_j(HOME_JOINT_DEG, "HOME_UP")
#         self.go_pose_j(cargo_target, "DROP_SLOT")
        
#         self.robot.move_l_rel(self.rc, np.array([0.0, 0.0, Z_DOWN_MM, 0.0, 0.0, 0.0]), L_VEL, L_ACC, rb.ReferenceFrame.Tool)
#         self.wait_move("DESCEND")
#         self.call_gripper(self.open_client, "RELEASE")
#         self.robot.move_l_rel(self.rc, np.array([0.0, 0.0, Z_UP_MM, 0.0, 0.0, 0.0]), L_VEL, L_ACC, rb.ReferenceFrame.Tool)
#         self.wait_move("ASCEND")
        
#         self.go_pose_j(HOME_JOINT_DEG, "HOME_BACK")
#         return True

#     def go_pose_j(self, joints, name):
#         self.robot.move_j(self.rc, joints, J_VEL, J_ACC)
#         return self.wait_move(name)

#     def go_final_pose(self):
#         self.robot.move_j(self.rc, POSE_FINAL, J_VEL, J_ACC)
#         return self.wait_move("FINAL")

#     def wait_move(self, name):
#         self.get_logger().info(f"⏳ Waiting {name}...")
#         self.robot.wait_for_move_finished(self.rc)
#         self.get_logger().info(f"✅ {name} DONE")
#         return True

#     def call_pose(self, timeout):
#         req = GetObjectPose.Request()
#         future = self.pose_client.call_async(req)
#         start = time.time()
#         while rclpy.ok() and (time.time() - start < timeout):
#             if future.done():
#                 try:
#                     res = future.result()
#                     # 🎯 success가 False면 마커가 없는 것이므로 즉시 None 반환
#                     if res is not None and res.success:
#                         return res
#                     else:
#                         return None
#                 except Exception:
#                     return None
#             time.sleep(0.1)
#         return None

#     def call_gripper(self, client, name):
#         future = client.call_async(Trigger.Request())
#         start = time.time()
#         while rclpy.ok() and (time.time() - start < 4.0):
#             if future.done(): return True
#             time.sleep(0.1)
#         return False

# def main():
#     rclpy.init()
#     node = LoadNode()
#     executor = MultiThreadedExecutor()
#     executor.add_node(node)
#     executor.spin()
#     rclpy.shutdown()

# if __name__ == "__main__": main()


# #!/usr/bin/env python3            // 통신 중복동작 수정 전
# import threading
# import time
# import os
# import numpy as np

# import rclpy
# from rclpy.node import Node
# from rclpy.executors import MultiThreadedExecutor
# from rclpy.callback_groups import ReentrantCallbackGroup

# import rbpodo as rb
# from std_srvs.srv import Trigger
# from msgs_pkg.srv import GetObjectPose, RunWS

# # 설정값 (기존 설정 유지)
# ROBOT_IP = "10.0.2.7"
# COUNT_FILE = "/tmp/loaded_count.txt"
# HOME_JOINT_DEG = np.array([-90.0, 0.0, 90.0, 0.0, 90.0, 0.0])
# POSE_FINAL = np.array([-90.0, -35.0, 125.0, 0.0, 90.0, 0.0])
# CARGO_POSES = [
#     np.array([-65.31, -16.33, -31.69, -0.01, -131.95, 24.62]), # Slot 1
#     np.array([-93.25, -7.91, -42.05, 0.0, -130.0, -3.32]),    # Slot 2
#     np.array([-117.67, -16.88, -30.94, 0.04, -132.13, -27.71]) # Slot 3
# ]

# CAM_TO_TCP_OFFSET_X_MM = -51.0
# CAM_TO_TCP_OFFSET_Y_MM = 32.0
# Z_APPROACH_MM = 365.0
# Z_DOWN_MM = 62.0
# Z_UP_MM = -70.0

# J_VEL, J_ACC = 255, 255
# L_VEL, L_ACC = 500, 800

# class LoadNode(Node):
#     def __init__(self):
#         super().__init__("load_node")
#         self.get_logger().info("✅ Load Node: Grip Failure Termination Mode Ready")

#         self.callback_group = ReentrantCallbackGroup()
        
#         try:
#             self.robot = rb.Cobot(ROBOT_IP)
#             self.rc = rb.ResponseCollector()
#             self.robot.set_operation_mode(self.rc, rb.OperationMode.Real)
#             self.get_logger().info("🤖 Robot Connected")
#         except Exception as e:
#             self.get_logger().error(f"❌ Connection Error: {e}")

#         self.open_client = self.create_client(Trigger, "/gripper/open", callback_group=self.callback_group)
#         self.grip_client = self.create_client(Trigger, "/gripper/grip", callback_group=self.callback_group)
#         self.pose_client = self.create_client(GetObjectPose, "/vision/get_object_pose", callback_group=self.callback_group)
#         self.srv = self.create_service(RunWS, "/task/load3", self.cb_load3, callback_group=self.callback_group)

#         self._busy_lock = threading.Lock()
#         self._busy = False

#     def get_current_count(self):
#         if os.path.exists(COUNT_FILE):
#             try:
#                 with open(COUNT_FILE, "r") as f:
#                     return int(f.read().strip())
#             except: return 0
#         return 0

#     def cb_load3(self, req, res):
#         with self._busy_lock:
#             if self._busy: return res
#             self._busy = True
#         try:
#             total = self.sequence_load()
#             res.success = True
#             res.message = f"Total count in tray: {total}"
#             return res
#         finally:
#             with self._busy_lock: self._busy = False

#     def sequence_load(self):
#         current_count = self.get_current_count()
#         self.get_logger().info(f"▶ START. Current Tray: {current_count}/3")

#         if current_count >= 3:
#             self.get_logger().warn("⚠️ Tray FULL.")
#             self.go_final_pose()
#             return current_count

#         self.call_gripper(self.open_client, "OPEN")
#         self.go_home_forced(timeout=1.0) 

#         newly_loaded = 0
#         for i in range(current_count, len(CARGO_POSES)):
#             self.get_logger().info(f"🚀 Slot #{i+1} Attempting")
            
#             # run_once의 결과에 따라 시나리오 분기
#             result = self.run_once(CARGO_POSES[i])
            
#             if result == "SUCCESS":
#                 newly_loaded += 1
#             elif result == "GRIP_FAILED":
#                 self.get_logger().error("🛑 Grip Failed! Terminating whole sequence.")
#                 break # 그리퍼 실패 즉시 반복문 탈출 (더 이상 시도 안 함)
#             else: # 비전 인식 실패(pose is None) 등
#                 self.get_logger().info("ℹ️ No more items detected. Stopping.")
#                 break 

#         updated_total = current_count + newly_loaded
#         with open(COUNT_FILE, "w") as f:
#             f.write(str(updated_total))
        
#         self.get_logger().info(f"💾 Updated Total: {updated_total}/3")
        
#         # 루프를 빠져나오면(정상 완료든 실패 중단이든) 무조건 파이널 포즈로 이동
#         self.go_final_pose()
#         return updated_total

#     def run_once(self, cargo_target):
#         pose = self.call_pose(5.0)
#         if pose is None: return "NO_ITEM"

#         # 비전 정렬
#         target_j = HOME_JOINT_DEG.copy()
#         target_j[5] += float(pose.rz)
#         self.robot.move_j(self.rc, target_j, J_VEL, J_ACC)
#         time.sleep(2.0) 

#         # 물체 위치로 이동
#         dx = -(pose.x * 1000.0) + CAM_TO_TCP_OFFSET_Y_MM
#         dy =  (pose.y * 1000.0) + CAM_TO_TCP_OFFSET_X_MM
#         self.robot.move_l_rel(self.rc, np.array([dy, dx, 0.0, 0.0, 0.0, 0.0]), L_VEL, L_ACC, rb.ReferenceFrame.Tool)
#         self.wait_move("ALIGN_XY")

#         self.robot.move_l_rel(self.rc, np.array([0.0, 0.0, Z_APPROACH_MM, 0.0, 0.0, 0.0]), L_VEL, L_ACC, rb.ReferenceFrame.Tool)
#         self.wait_move("APPROACH")

#         # [핵심] 그리퍼 잡기 시도
#         if not self.call_gripper(self.grip_client, "GRIP"):
#             # 그리퍼가 실패 신호를 보내면 "GRIP_FAILED" 반환
#             # 충돌 방지를 위해 살짝 위로 이동 후 홈 복귀
#             self.get_logger().warn("Grip failed! Returning to safe pose.")
#             self.robot.move_l_rel(self.rc, np.array([0.0, 0.0, -100.0, 0.0, 0.0, 0.0]), L_VEL, L_ACC, rb.ReferenceFrame.Tool)
#             self.wait_move("EMERGENCY_UP")
#             self.go_home()
#             return "GRIP_FAILED"
        
#         # 성공 시 시퀀스 계속 진행
#         self.go_home() 
#         self.go_pose_j(cargo_target, "DROP_SLOT")
        
#         self.robot.move_l_rel(self.rc, np.array([0.0, 0.0, Z_DOWN_MM, 0.0, 0.0, 0.0]), L_VEL, L_ACC, rb.ReferenceFrame.Tool)
#         self.wait_move("DESCEND")
#         self.call_gripper(self.open_client, "RELEASE")
#         self.robot.move_l_rel(self.rc, np.array([0.0, 0.0, Z_UP_MM, 0.0, 0.0, 0.0]), L_VEL, L_ACC, rb.ReferenceFrame.Tool)
#         self.wait_move("ASCEND")
        
#         self.go_home()
#         return "SUCCESS"

#     # --- 제어 함수들 ---

#     def go_home_forced(self, timeout=3.0):
#         self.get_logger().info(f"🏠 [Forced] Moving to HOME... ({timeout}s)")
#         self.robot.move_j(self.rc, HOME_JOINT_DEG, J_VEL, J_ACC)
#         time.sleep(timeout)
#         return True

#     def go_home(self):
#         self.robot.move_j(self.rc, HOME_JOINT_DEG, J_VEL, J_ACC)
#         return self.wait_move("HOME")

#     def wait_move(self, name):
#         self.robot.wait_for_move_finished(self.rc)
#         return True

#     def go_final_pose(self):
#         self.robot.move_j(self.rc, POSE_FINAL, J_VEL, J_ACC)
#         return self.wait_move("FINAL")

#     def go_pose_j(self, joints, name):
#         self.robot.move_j(self.rc, joints, J_VEL, J_ACC)
#         return self.wait_move(name)

#     def call_pose(self, timeout):
#         req = GetObjectPose.Request()
#         future = self.pose_client.call_async(req)
#         start = time.time()
#         while rclpy.ok() and (time.time() - start < timeout):
#             if future.done():
#                 res = future.result()
#                 if res and res.success: return res
#                 else: return None
#             time.sleep(0.1)
#         return None

#     def call_gripper(self, client, name):
#         future = client.call_async(Trigger.Request())
#         start = time.time()
#         # 그리퍼 노드가 5초 타임아웃을 가지므로 여기서는 여유 있게 6초 대기
#         while rclpy.ok() and (time.time() - start < 6.0):
#             if future.done():
#                 res = future.result()
#                 # 서비스 응답 객체의 success 필드 확인 (GripperNode가 False 반환 시 실패)
#                 if res and res.success: return True
#                 else: return False
#             time.sleep(0.1)
#         return False

# def main():
#     rclpy.init()
#     node = LoadNode()
#     executor = MultiThreadedExecutor()
#     executor.add_node(node)
#     try:
#         executor.spin()
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == "__main__": main()



import threading
import time
import os
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

import rbpodo as rb
from std_srvs.srv import Trigger
from msgs_pkg.srv import GetObjectPose, RunWS

# 설정값 (기존 설정 유지)
ROBOT_IP = "10.0.2.7"
COUNT_FILE = "/tmp/loaded_count.txt"
HOME_JOINT_DEG = np.array([-90.0, 38.97, 24.86, 0.0, 116.17, 0.0])
POSE_FINAL = np.array([-90.0, -22.08, 118.94, -0.33, 84.91, 0.0])
CARGO_POSES = [
    np.array([-65.31, -16.33, -31.69, -0.01, -131.95, 24.62]), # Slot 1
    np.array([-93.25, -7.91, -42.05, 0.0, -130.0, -3.32]),    # Slot 2
    np.array([-117.67, -16.88, -30.94, 0.04, -132.13, -27.71]) # Slot 3
]

CAM_TO_TCP_OFFSET_X_MM = -51.0
CAM_TO_TCP_OFFSET_Y_MM = 32.0
Z_APPROACH_MM = 450.0
Z_DOWN_MM = 15.0
Z_UP_MM = -15.0

J_VEL, J_ACC = 255, 255
L_VEL, L_ACC = 500, 800

class LoadNode(Node):
    def __init__(self):
        super().__init__("load_node")
        self.get_logger().info("✅ Load Node: Full/Empty Fast-Return Mode Ready")

        self.callback_group = ReentrantCallbackGroup()
        
        try:
            self.robot = rb.Cobot(ROBOT_IP)
            self.rc = rb.ResponseCollector()
            self.robot.set_operation_mode(self.rc, rb.OperationMode.Real)
            self.get_logger().info("🤖 Robot Connected")
        except Exception as e:
            self.get_logger().error(f"❌ Connection Error: {e}")

        self.open_client = self.create_client(Trigger, "/gripper/open", callback_group=self.callback_group)
        self.grip_client = self.create_client(Trigger, "/gripper/grip", callback_group=self.callback_group)
        self.pose_client = self.create_client(GetObjectPose, "/vision/get_object_pose", callback_group=self.callback_group)
        self.srv = self.create_service(RunWS, "/task/load3", self.cb_load3, callback_group=self.callback_group)

        self._busy_lock = threading.Lock()
        self._busy = False

    def get_current_count(self):
        if os.path.exists(COUNT_FILE):
            try:
                with open(COUNT_FILE, "r") as f:
                    return int(f.read().strip())
            except: return 0
        return 0

    def cb_load3(self, req, res):
        # 1. Busy 체크
        with self._busy_lock:
            if self._busy:
                self.get_logger().error("🚫 BUSY: Request ignored.")
                res.success = False
                res.message = "Busy"
                return res
            self._busy = True

        try:
            current_count = self.get_current_count()
            self.get_logger().info(f"📩 LOAD Request. Current Tray: {current_count}/3")

            # [핵심 수정 1] 이미 가득 찼다면(3개 이상) 동작 없이 즉시 종료
            if current_count >= 3:
                self.get_logger().warn("⚠️ Tray is already FULL. Skipping logic.")
                res.success = True
                res.message = "Tray Full"
                return res  # 여기서 리턴하면 finally로 직행하여 busy 해제됨

            # [핵심 수정 2] 시퀀스 실행 (current_count 전달)
            total = self.sequence_load(current_count)
            
            # 시퀀스가 정상적으로 끝난 경우에만 마지막 포즈 이동 (sequence_load 내부에서 처리됨)
            
            res.success = True
            res.message = f"Total count in tray: {total}"
            return res

        except Exception as e:
            self.get_logger().error(f"❌ Exception in cb_load3: {e}")
            res.success = False
            return res

        finally:
            # [중요] 어떤 상황에서도 Busy Lock 해제
            with self._busy_lock: 
                self._busy = False
            self.get_logger().info("🔓 Busy Lock Released.")

    def sequence_load(self, start_count):
        # 이미 cb_load3에서 3개 이상인 경우는 걸러냈으므로 바로 동작 시작
        self.call_gripper(self.open_client, "OPEN")
        self.go_home_forced(timeout=1.0) 

        newly_loaded = 0
        
        # start_count 부터 3(최대슬롯)까지 반복
        for i in range(start_count, len(CARGO_POSES)):
            self.get_logger().info(f"🚀 Slot #{i+1} Attempting")
            
            result = self.run_once(CARGO_POSES[i])
            
            if result == "SUCCESS":
                newly_loaded += 1
            elif result == "GRIP_FAILED":
                self.get_logger().error("🛑 Grip Failed! Terminating whole sequence.")
                break 
            else: 
                # 비전 인식 실패 (바닥에 물건 없음)
                self.get_logger().info("ℹ️ No item detected. Stopping sequence.")
                break 

        updated_total = start_count + newly_loaded
        
        # 파일 업데이트
        with open(COUNT_FILE, "w") as f:
            f.write(str(updated_total))
        self.get_logger().info(f"💾 Updated Total: {updated_total}/3")
        
        # [핵심 수정 3] 작업을 시도했다면 최종 포즈로 이동
        # (만약 비전 인식 실패로 하나도 못 집었더라도, Home까지 갔었으므로 복귀 필요)
        self.go_final_pose()
        
        return updated_total

    def run_once(self, cargo_target):
        pose = self.call_pose(5.0)
        if pose is None: return "NO_ITEM"

        # 비전 정렬
        target_j = HOME_JOINT_DEG.copy()
        target_j[5] += float(pose.rz)
        self.robot.move_j(self.rc, target_j, J_VEL, J_ACC)
        time.sleep(1.0) 

        # 물체 위치로 이동
        dx = -(pose.x * 1000.0) + CAM_TO_TCP_OFFSET_Y_MM
        dy =  (pose.y * 1000.0) + CAM_TO_TCP_OFFSET_X_MM
        self.robot.move_l_rel(self.rc, np.array([dy, dx, 0.0, 0.0, 0.0, 0.0]), L_VEL, L_ACC, rb.ReferenceFrame.Tool)
        self.wait_move("ALIGN_XY")

        self.robot.move_l_rel(self.rc, np.array([0.0, 0.0, Z_APPROACH_MM, 0.0, 0.0, 0.0]), L_VEL, L_ACC, rb.ReferenceFrame.Tool)
        self.wait_move("APPROACH")

        # 그리퍼 잡기 시도
        if not self.call_gripper(self.grip_client, "GRIP"):
            self.get_logger().warn("Grip failed! Returning to safe pose.")
            self.robot.move_l_rel(self.rc, np.array([0.0, 0.0, -100.0, 0.0, 0.0, 0.0]), L_VEL, L_ACC, rb.ReferenceFrame.Tool)
            self.wait_move("EMERGENCY_UP")
            self.go_home()
            return "GRIP_FAILED"
        
        # 성공 시 적재 동작
        self.go_home() 
        self.go_pose_j(cargo_target, "DROP_SLOT")
        
        self.robot.move_l_rel(self.rc, np.array([0.0, 0.0, Z_DOWN_MM, 0.0, 0.0, 0.0]), L_VEL, L_ACC, rb.ReferenceFrame.Tool)
        self.wait_move("DESCEND")
        self.call_gripper(self.open_client, "RELEASE")
        self.robot.move_l_rel(self.rc, np.array([0.0, 0.0, Z_UP_MM, 0.0, 0.0, 0.0]), L_VEL, L_ACC, rb.ReferenceFrame.Tool)
        self.wait_move("ASCEND")
        
        self.go_home()
        return "SUCCESS"

    # --- 제어 함수들 ---

    def go_home_forced(self, timeout=3.0):
        self.get_logger().info(f"🏠 [Forced] Moving to HOME... ({timeout}s)")
        self.robot.move_j(self.rc, HOME_JOINT_DEG, J_VEL, J_ACC)
        time.sleep(timeout)
        return True

    def go_home(self):
        self.robot.move_j(self.rc, HOME_JOINT_DEG, J_VEL, J_ACC)
        return self.wait_move("HOME")

    def wait_move(self, name):
        self.robot.wait_for_move_finished(self.rc)
        return True

    def go_final_pose(self):
        self.robot.move_j(self.rc, POSE_FINAL, J_VEL, J_ACC)
        return self.wait_move("FINAL")

    def go_pose_j(self, joints, name):
        self.robot.move_j(self.rc, joints, J_VEL, J_ACC)
        return self.wait_move(name)

    def call_pose(self, timeout):
        req = GetObjectPose.Request()
        future = self.pose_client.call_async(req)
        start = time.time()
        while rclpy.ok() and (time.time() - start < timeout):
            if future.done():
                res = future.result()
                if res and res.success: return res
                else: return None
            time.sleep(0.1)
        return None

    def call_gripper(self, client, name):
        future = client.call_async(Trigger.Request())
        start = time.time()
        while rclpy.ok() and (time.time() - start < 6.0):
            if future.done():
                res = future.result()
                if res and res.success: return True
                else: return False
            time.sleep(0.1)
        return False

def main():
    rclpy.init()
    node = LoadNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__": main()