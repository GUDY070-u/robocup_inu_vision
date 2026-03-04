# #!/usr/bin/env python3    //3개 무조건 내리기
# import threading
# import time
# import numpy as np

# import rclpy
# from rclpy.node import Node
# from rclpy.executors import MultiThreadedExecutor
# from rclpy.callback_groups import ReentrantCallbackGroup

# import rbpodo as rb
# from std_srvs.srv import Trigger
# from msgs_pkg.srv import RunWS

# # ===============================
# # 설정값 (Configuration)
# # ===============================
# ROBOT_IP = "10.0.2.7"

# HOME_JOINT_DEG  = np.array([-90.0,   0.0,  90.0, 0.0,  90.0, 0.0])
# POSE_FINAL      = np.array([-90.0, -35.0, 125.0, 0.0, 90.0, 0.0])

# # 트레이 위 물체 위치 (1, 2, 3번)
# POSE_CARGO1 = np.array([-65.31, -16.33, -31.69, -0.01, -131.95, 24.62])
# POSE_CARGO2 = np.array([-93.25, -7.91, -42.05, 0.0, -130.0, -3.32])
# POSE_CARGO3 = np.array([-117.67, -16.88, -30.94, 0.04, -132.13, -27.71])
# CARGO_LIST  = [POSE_CARGO1, POSE_CARGO2, POSE_CARGO3]

# # 하차 지점 (바닥) 및 오프셋
# POSE_DROP = np.array([-91.25, 55.35,  66.41, 0.0,  58.24, -1.33])
# DROP_OFFSETS = [
#     np.array([0.0,   0.0, 0.0, 0.0, 0.0, 0.0]),
#     np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
#     np.array([-100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# ]

# CARGO_DOWN_MM = 60.0
# CARGO_UP_MM   = -70.0
# DROP_DOWN_MM  = 50.0
# DROP_UP_MM    = -50.0

# J_VEL, J_ACC = 255, 255
# L_VEL, L_ACC = 500, 800
# MOVE_START_TIMEOUT_SEC = 2.0

# class UnloadNode(Node):
#     def __init__(self):
#         super().__init__("unload_node")
#         self.get_logger().info("✅ Unload Node Ready (Waiting for Coordinator Signal)")

#         # 멀티스레드 환경 설정
#         self.callback_group = ReentrantCallbackGroup()

#         # 로봇 연결
#         try:
#             self.robot = rb.Cobot(ROBOT_IP)
#             self.rc = rb.ResponseCollector()
#             self.robot.set_operation_mode(self.rc, rb.OperationMode.Real)
#             self.get_logger().info("🤖 Robot Connected")
#         except Exception as e:
#             self.get_logger().error(f"❌ Robot Connection Fail: {e}")

#         # 클라이언트 및 서비스 서버 (모두 같은 콜백 그룹 지정)
#         self.open_client = self.create_client(Trigger, "/gripper/open", callback_group=self.callback_group)
#         self.grip_client = self.create_client(Trigger, "/gripper/grip", callback_group=self.callback_group)
        
#         # Coordinator가 호출할 서비스 서버
#         self.srv = self.create_service(RunWS, "/task/unload3", self.cb_unload3, callback_group=self.callback_group)

#         self._busy_lock = threading.Lock()
#         self._busy = False

#     def cb_unload3(self, req, res):
#         self.get_logger().info(f"📩 Unload request received for WS{req.ws}")
        
#         with self._busy_lock:
#             if self._busy:
#                 self.get_logger().warn("⚠️ Already busy!")
#                 res.success = False
#                 res.message = "busy"
#                 return res
#             self._busy = True

#         try:
#             # 💡 하차 개수 판단: 여기서는 기본적으로 3개를 시도합니다.
#             ok = self.sequence_once(count=3)
#             res.success = bool(ok)
#             res.message = "ok"
#             return res
#         finally:
#             with self._busy_lock:
#                 self._busy = False

#     def sequence_once(self, count: int) -> bool:
#         self.get_logger().info(f"▶ START UNLOAD: {count} items")
        
#         if not self.call_gripper(self.open_client, "OPEN"): return False
#         if not self.go_pose_j(HOME_JOINT_DEG, "HOME"): return False

#         for i in range(count):
#             self.get_logger().info(f"🚀 Working on Item #{i+1}")
#             if not self.run_once(CARGO_LIST[i], DROP_OFFSETS[i]):
#                 return False
#             time.sleep(0.5)

#         self.go_pose_j(POSE_FINAL, "FINAL")
#         return True

#     def run_once(self, cargo_target, drop_offset) -> bool:
#         # 트레이에서 집기
#         if not self.go_pose_j(cargo_target, "CARGO_PICK"): return False
#         if not self.move_l_rel(CARGO_DOWN_MM, "DESCEND"): return False
#         if not self.call_gripper(self.grip_client, "GRIP"): return False
#         if not self.move_l_rel(CARGO_UP_MM, "ASCEND"): return False

#         # 바닥에 놓기
#         if not self.go_pose_j(POSE_DROP, "DROP_BASE"): return False
#         if np.any(drop_offset != 0):
#             self.robot.move_l_rel(self.rc, drop_offset, L_VEL, L_ACC, rb.ReferenceFrame.Base)
#             self.wait_move("SHIFT")

#         if not self.move_l_rel(DROP_DOWN_MM, "DROP_DESCEND"): return False
#         if not self.call_gripper(self.open_client, "RELEASE"): return False
#         if not self.move_l_rel(DROP_UP_MM, "DROP_ASCEND"): return False
        
#         return self.go_pose_j(HOME_JOINT_DEG, "HOME_BACK")

#     # --- 공통 함수 ---
#     def go_pose_j(self, joints, name):
#         self.get_logger().info(f"➡️ Moving to {name}")
#         self.robot.move_j(self.rc, joints, J_VEL, J_ACC)
#         return self.wait_move(name)

#     def move_l_rel(self, dz_mm, name):
#         self.get_logger().info(f"↕️ Linear {name}: {dz_mm}mm")
#         cmd = np.array([0.0, 0.0, dz_mm, 0.0, 0.0, 0.0])
#         self.robot.move_l_rel(self.rc, cmd, L_VEL, L_ACC, rb.ReferenceFrame.Tool)
#         return self.wait_move(name)

#     def call_gripper(self, client, name):
#         self.get_logger().info(f"✊ Gripper {name}...")
#         req = Trigger.Request()
#         future = client.call_async(req)
#         start_t = time.time()
#         while rclpy.ok() and (time.time() - start_t < 4.0):
#             if future.done():
#                 res = future.result()
#                 return res and res.success
#             time.sleep(0.1)
#         return False

#     def wait_move(self, name):
#         self.get_logger().info(f"⏳ Waiting {name}...")
#         # 시작 신호 관계없이 무조건 완료까지 블로킹 대기
#         self.robot.wait_for_move_finished(self.rc)
#         self.get_logger().info(f"✅ {name} DONE")
#         return True

# def main(args=None):
#     rclpy.init(args=args)
#     node = UnloadNode()
#     executor = MultiThreadedExecutor(num_threads=8)
#     executor.add_node(node)
#     try:
#         executor.spin()
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == "__main__":
#     main()



# #!/usr/bin/env python3            //load에서 몇개 집었는지 기억하고 내려놓기
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
# from msgs_pkg.srv import RunWS

# # 설정값
# ROBOT_IP = "10.0.2.7"
# COUNT_FILE = "/tmp/loaded_count.txt"

# HOME_JOINT_DEG = np.array([-90.0, 0.0, 90.0, 0.0, 90.0, 0.0])
# POSE_FINAL = np.array([-90.0, -35.0, 125.0, 0.0, 90.0, 0.0])

# CARGO_LIST = [
#     np.array([-65.31, -16.33, -31.69, -0.01, -131.95, 24.62]),
#     np.array([-93.25, -7.91, -42.05, 0.0, -130.0, -3.32]),
#     np.array([-117.67, -16.88, -30.94, 0.04, -132.13, -27.71])
# ]

# POSE_DROP = np.array([-91.25, 55.35,  66.41, 0.0,  58.24, -1.33])
# DROP_OFFSETS = [
#     np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
#     np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
#     np.array([-100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# ]

# CARGO_DOWN_MM, CARGO_UP_MM = 60.0, -70.0
# DROP_DOWN_MM, DROP_UP_MM = 50.0, -50.0
# J_VEL, J_ACC = 255, 255
# L_VEL, L_ACC = 500, 800

# class UnloadNode(Node):
#     def __init__(self):
#         super().__init__("unload_node")
#         self.get_logger().info("✅ Unload Node Ready (Persistence Mode)")

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
#         self.srv = self.create_service(RunWS, "/task/unload3", self.cb_unload3, callback_group=self.callback_group)

#         self._busy_lock = threading.Lock()
#         self._busy = False

#     def cb_unload3(self, req, res):
#         # 🎯 파일에서 실제 적재된 개수 읽기
#         count = 0
#         if os.path.exists(COUNT_FILE):
#             try:
#                 with open(COUNT_FILE, "r") as f:
#                     count = int(f.read().strip())
#             except: count = 0
        
#         self.get_logger().info(f"📩 Received Unload Request. Count from record: {count}")

#         with self._busy_lock:
#             if self._busy: return res
#             self._busy = True

#         try:
#             # 🎯 만약 개수가 0이더라도 무조건 POSE_FINAL로 이동한 뒤 성공 반환
#             if count <= 0:
#                 self.get_logger().warn("ℹ️ No items recorded. Moving to FINAL directly.")
#             else:
#                 self.sequence_unload(count)
            
#             # 🎯 하차 로직 종료 후(혹은 0일 때) 무조건 최종 포즈 이동
#             self.go_final_pose()
#             res.success = True
#             return res
#         finally:
#             with self._busy_lock: self._busy = False

#     def sequence_unload(self, count):
#         self.get_logger().info(f"▶ START UNLOAD SEQUENCE: {count} items")
#         self.call_gripper(self.open_client, "OPEN")
#         self.go_home()

#         for i in range(count):
#             self.get_logger().info(f"📦 Unloading item #{i+1}")
#             self.run_once(CARGO_LIST[i], DROP_OFFSETS[i])

#     def run_once(self, cargo_target, drop_offset):
#         self.go_pose_j(cargo_target, "PICK_POS")
#         self.robot.move_l_rel(self.rc, np.array([0.0,0.0,CARGO_DOWN_MM,0.0,0.0,0.0]), L_VEL, L_ACC, rb.ReferenceFrame.Tool)
#         self.wait_move("DESCEND")
#         self.call_gripper(self.grip_client, "GRIP")
#         self.robot.move_l_rel(self.rc, np.array([0.0,0.0,CARGO_UP_MM,0.0,0.0,0.0]), L_VEL, L_ACC, rb.ReferenceFrame.Tool)
#         self.wait_move("ASCEND")

#         self.go_pose_j(POSE_DROP, "DROP_BASE")
#         if np.any(drop_offset != 0):
#             self.robot.move_l_rel(self.rc, drop_offset, L_VEL, L_ACC, rb.ReferenceFrame.Base)
#             self.wait_move("SHIFT")

#         self.robot.move_l_rel(self.rc, np.array([0.0,0.0,DROP_DOWN_MM,0.0,0.0,0.0]), L_VEL, L_ACC, rb.ReferenceFrame.Tool)
#         self.wait_move("DROP_DESCEND")
#         self.call_gripper(self.open_client, "RELEASE")
#         self.robot.move_l_rel(self.rc, np.array([0.0,0.0,DROP_UP_MM,0.0,0.0,0.0]), L_VEL, L_ACC, rb.ReferenceFrame.Tool)
#         self.wait_move("DROP_ASCEND")
#         self.go_home()

#     def go_home(self):
#         self.robot.move_j(self.rc, HOME_JOINT_DEG, J_VEL, J_ACC)
#         return self.wait_move("HOME")

#     def go_final_pose(self):
#         self.robot.move_j(self.rc, POSE_FINAL, J_VEL, J_ACC)
#         return self.wait_move("FINAL")

#     def go_pose_j(self, joints, name):
#         self.robot.move_j(self.rc, joints, J_VEL, J_ACC)
#         return self.wait_move(name)

#     def wait_move(self, name):
#         self.robot.wait_for_move_finished(self.rc)
#         return True

#     def call_gripper(self, client, name):
#         future = client.call_async(Trigger.Request())
#         start = time.time()
#         while rclpy.ok() and (time.time() - start < 4.0):
#             if future.done(): return True
#             time.sleep(0.1)
#         return False

# def main():
#     rclpy.init()
#     node = UnloadNode()
#     executor = MultiThreadedExecutor()
#     executor.add_node(node)
#     executor.spin()
#     rclpy.shutdown()

# if __name__ == "__main__": main()


#!/usr/bin/env python3              통신 중복동작 수정 전
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
# from msgs_pkg.srv import RunWS

# # 설정값 (유지)
# ROBOT_IP = "10.0.2.7"
# COUNT_FILE = "/tmp/loaded_count.txt"

# HOME_JOINT_DEG = np.array([-90.0, 0.0, 90.0, 0.0, 90.0, 0.0])
# POSE_FINAL = np.array([-90.0, -35.0, 125.0, 0.0, 90.0, 0.0])

# CARGO_LIST = [
#     np.array([-65.31, -16.33, -31.69, -0.01, -131.95, 24.62]),
#     np.array([-93.25, -7.91, -42.05, 0.0, -130.0, -3.32]),
#     np.array([-117.67, -16.88, -30.94, 0.04, -132.13, -27.71])
# ]

# POSE_DROP = np.array([-90.0, 27.51,  101.66, 0.0,  50.83, 0.0])
# DROP_OFFSETS = [
#     np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
#     np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
#     np.array([-100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# ]

# CARGO_DOWN_MM, CARGO_UP_MM = 60.0, -70.0
# DROP_DOWN_MM, DROP_UP_MM = 100.0, -100.0
# J_VEL, J_ACC = 255, 255
# L_VEL, L_ACC = 500, 800

# class UnloadNode(Node):
#     def __init__(self):
#         super().__init__("unload_node")
#         self.get_logger().info("✅ Unload Node: Grip Failure Termination Mode Ready")

#         self.callback_group = ReentrantCallbackGroup()
        
#         try:
#             self.robot = rb.Cobot(ROBOT_IP)
#             self.rc = rb.ResponseCollector()
#             self.robot.set_operation_mode(self.rc, rb.OperationMode.Real)
#             self.robot.set_speed_bar(self.rc, 1.0)
#             self.get_logger().info("🤖 Robot Connected")
#         except Exception as e:
#             self.get_logger().error(f"❌ Connection Error: {e}")

#         self.open_client = self.create_client(Trigger, "/gripper/open", callback_group=self.callback_group)
#         self.grip_client = self.create_client(Trigger, "/gripper/grip", callback_group=self.callback_group)
#         self.srv = self.create_service(RunWS, "/task/unload3", self.cb_unload3, callback_group=self.callback_group)

#         self._busy_lock = threading.Lock()
#         self._busy = False

#     def cb_unload3(self, req, res):
#         count = 0
#         if os.path.exists(COUNT_FILE):
#             try:
#                 with open(COUNT_FILE, "r") as f:
#                     count = int(f.read().strip())
#             except: count = 0
        
#         self.get_logger().info(f"📩 UNLOAD Request. Items in Tray: {count}")

#         with self._busy_lock:
#             if self._busy: return res
#             self._busy = True

#         try:
#             if count > 0:
#                 # 실제로 하차에 성공한 개수를 추적하기 위해 sequence_unload의 반환값 사용
#                 success_count = self.sequence_unload(count)
                
#                 # [중요] 남은 개수 업데이트 (전체 하차 성공 시 0, 도중 실패 시 남은 개수 기록)
#                 remaining = count - success_count
#                 with open(COUNT_FILE, "w") as f:
#                     f.write(str(remaining))
#                 self.get_logger().info(f"♻️ Unload Process Finished. Remaining in tray: {remaining}")
#             else:
#                 self.get_logger().warn("ℹ️ Nothing to unload.")

#             self.go_final_pose()
#             res.success = True
#             return res
#         finally:
#             with self._busy_lock: self._busy = False

#     def sequence_unload(self, count):
#         self.get_logger().info(f"▶ START UNLOAD: {count} items")
#         self.call_gripper(self.open_client, "OPEN")
#         self.go_home_forced(timeout=1.0)

#         unloaded_so_far = 0
#         for i in range(count):
#             self.get_logger().info(f"📦 Unloading item #{i+1} from Slot #{i+1}")
            
#             result = self.run_once(CARGO_LIST[i], DROP_OFFSETS[i])
            
#             if result == "SUCCESS":
#                 unloaded_so_far += 1
#             elif result == "GRIP_FAILED":
#                 self.get_logger().error(f"🛑 Grip failed at Slot #{i+1}. Stopping unload sequence.")
#                 break # 실패 시 즉시 루프 탈출
                
#         return unloaded_so_far

#     def run_once(self, cargo_target, drop_offset):
#         # 1. 트레이 슬롯으로 이동
#         self.go_pose_j(cargo_target, "PICK_FROM_CARGO")
        
#         self.robot.move_l_rel(self.rc, np.array([0.0,0.0,CARGO_DOWN_MM,0.0,0.0,0.0]), L_VEL, L_ACC, rb.ReferenceFrame.Tool)
#         self.wait_move("DESCEND")
        
#         # [핵심] 그리퍼 잡기 시도 및 결과 확인
#         if not self.call_gripper(self.grip_client, "GRIP"):
#             self.get_logger().warn("Grip failed! Emergency return to home.")
#             # 안전을 위해 위로 회피 후 홈으로 이동
#             self.robot.move_l_rel(self.rc, np.array([0.0, 0.0, -100.0, 0.0, 0.0, 0.0]), L_VEL, L_ACC, rb.ReferenceFrame.Tool)
#             self.wait_move("EMERGENCY_UP")
#             self.go_home()
#             return "GRIP_FAILED"
        
#         self.robot.move_l_rel(self.rc, np.array([0.0,0.0,CARGO_UP_MM,0.0,0.0,0.0]), L_VEL, L_ACC, rb.ReferenceFrame.Tool)
#         self.wait_move("ASCEND")

#         # 2. 하차 지점으로 이동
#         self.go_pose_j(POSE_DROP, "DROP_BASE")
#         if np.any(drop_offset != 0):
#             self.robot.move_l_rel(self.rc, drop_offset, L_VEL, L_ACC, rb.ReferenceFrame.Base)
#             self.wait_move("SHIFT")

#         self.robot.move_l_rel(self.rc, np.array([0.0,0.0,DROP_DOWN_MM,0.0,0.0,0.0]), L_VEL, L_ACC, rb.ReferenceFrame.Tool)
#         self.wait_move("DROP_DESCEND")
        
#         self.call_gripper(self.open_client, "RELEASE")
        
#         self.robot.move_l_rel(self.rc, np.array([0.0,0.0,DROP_UP_MM,0.0,0.0,0.0]), L_VEL, L_ACC, rb.ReferenceFrame.Tool)
#         self.wait_move("DROP_ASCEND")

#         self.go_home()
#         return "SUCCESS"

#     # --- 제어 함수 (기존과 동일) ---

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

#     def call_gripper(self, client, name):
#         future = client.call_async(Trigger.Request())
#         start = time.time()
#         while rclpy.ok() and (time.time() - start < 6.0): # 그리퍼 노드 타임아웃 고려하여 6초
#             if future.done():
#                 res = future.result()
#                 if res and res.success: return True
#                 else: return False
#             time.sleep(0.1)
#         return False

# def main():
#     rclpy.init()
#     node = UnloadNode()
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
from msgs_pkg.srv import RunWS

# 설정값 (유지)
ROBOT_IP = "10.0.2.7"
COUNT_FILE = "/tmp/loaded_count.txt"

HOME_JOINT_DEG = np.array([-90.0, 0.0, 90.0, 0.0, 90.0, 0.0])
POSE_FINAL = np.array([-90.0, -22.08, 118.94, -0.33, 84.91, 0.0])

CARGO_LIST = [
    np.array([-65.31, -16.33, -31.69, -0.01, -131.95, 24.62]),
    np.array([-93.25, -7.91, -42.05, 0.0, -130.0, -3.32]),
    np.array([-117.67, -16.88, -30.94, 0.04, -132.13, -27.71])
]

POSE_DROP = np.array([-90.0, 27.51,  101.66, 0.0,  50.83, 0.0])
DROP_OFFSETS = [
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([-100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
]

CARGO_DOWN_MM, CARGO_UP_MM = 15.0, -15.0
DROP_DOWN_MM, DROP_UP_MM = 100.0, -100.0
J_VEL, J_ACC = 255, 255
L_VEL, L_ACC = 500, 800

class UnloadNode(Node):
    def __init__(self):
        super().__init__("unload_node")
        self.get_logger().info("✅ Unload Node: Grip Failure Termination Mode Ready")

        self.callback_group = ReentrantCallbackGroup()
        
        try:
            self.robot = rb.Cobot(ROBOT_IP)
            self.rc = rb.ResponseCollector()
            self.robot.set_operation_mode(self.rc, rb.OperationMode.Real)
            self.robot.set_speed_bar(self.rc, 1.0)
            self.get_logger().info("🤖 Robot Connected")
        except Exception as e:
            self.get_logger().error(f"❌ Connection Error: {e}")

        self.open_client = self.create_client(Trigger, "/gripper/open", callback_group=self.callback_group)
        self.grip_client = self.create_client(Trigger, "/gripper/grip", callback_group=self.callback_group)
        self.srv = self.create_service(RunWS, "/task/unload3", self.cb_unload3, callback_group=self.callback_group)

        self._busy_lock = threading.Lock()
        self._busy = False

    def cb_unload3(self, req, res):
        count = 0
        if os.path.exists(COUNT_FILE):
            try:
                with open(COUNT_FILE, "r") as f:
                    count = int(f.read().strip())
            except: count = 0
        
        self.get_logger().info(f"📩 UNLOAD Request. Items in Tray: {count}")

        with self._busy_lock:
            if self._busy:
                self.get_logger().error("🚫 BUSY: Request ignored because previous task is still running.")
                res.success = False
                res.message = "Busy"
                return res
            self._busy = True

        try:
            # [수정됨] 개수가 0이면 즉시 종료 (이동 없음)
            if count <= 0:
                self.get_logger().warn("ℹ️ Nothing to unload. Skipping logic.")
                res.success = True
                res.message = "Nothing to unload"
                return res  # 여기서 리턴하면 아래 finally 블록이 실행되어 busy가 해제됩니다.

            # --- count > 0 일 때만 아래 로직 실행 ---
            
            # 실제로 하차에 성공한 개수를 추적하기 위해 sequence_unload의 반환값 사용
            success_count = self.sequence_unload(count)
            
            # 남은 개수 업데이트
            remaining = count - success_count
            with open(COUNT_FILE, "w") as f:
                f.write(str(remaining))
            self.get_logger().info(f"♻️ Unload Process Finished. Remaining in tray: {remaining}")

            # 작업 완료 후 최종 위치 이동 (물건이 있어서 작업을 했을 때만 이동)
            self.go_final_pose()
            
            res.success = True
            return res

        except Exception as e:
            self.get_logger().error(f"❌ Exception in cb_unload3: {e}")
            res.success = False
            return res
            
        finally:
            # [중요] 어떤 상황(성공, 실패, 0개 등)에서도 반드시 Busy 상태를 해제함
            with self._busy_lock: 
                self._busy = False
            self.get_logger().info("🔓 Busy Lock Released.")

    def sequence_unload(self, count):
        self.get_logger().info(f"▶ START UNLOAD: {count} items")
        self.call_gripper(self.open_client, "OPEN")
        self.go_home_forced(timeout=1.0)

        unloaded_so_far = 0
        for i in range(count):
            self.get_logger().info(f"📦 Unloading item #{i+1} from Slot #{i+1}")
            
            result = self.run_once(CARGO_LIST[i], DROP_OFFSETS[i])
            
            if result == "SUCCESS":
                unloaded_so_far += 1
            elif result == "GRIP_FAILED":
                self.get_logger().error(f"🛑 Grip failed at Slot #{i+1}. Stopping unload sequence.")
                break # 실패 시 즉시 루프 탈출
                
        return unloaded_so_far

    def run_once(self, cargo_target, drop_offset):
        # 1. 트레이 슬롯으로 이동
        self.go_pose_j(cargo_target, "PICK_FROM_CARGO")
        
        self.robot.move_l_rel(self.rc, np.array([0.0,0.0,CARGO_DOWN_MM,0.0,0.0,0.0]), L_VEL, L_ACC, rb.ReferenceFrame.Tool)
        self.wait_move("DESCEND")
        
        # 그리퍼 잡기 시도 및 결과 확인
        if not self.call_gripper(self.grip_client, "GRIP"):
            self.get_logger().warn("Grip failed! Emergency return to home.")
            # 안전을 위해 위로 회피 후 홈으로 이동
            self.robot.move_l_rel(self.rc, np.array([0.0, 0.0, -100.0, 0.0, 0.0, 0.0]), L_VEL, L_ACC, rb.ReferenceFrame.Tool)
            self.wait_move("EMERGENCY_UP")
            self.go_home()
            return "GRIP_FAILED"
        
        self.robot.move_l_rel(self.rc, np.array([0.0,0.0,CARGO_UP_MM,0.0,0.0,0.0]), L_VEL, L_ACC, rb.ReferenceFrame.Tool)
        self.wait_move("ASCEND")

        # 2. 하차 지점으로 이동
        self.go_pose_j(POSE_DROP, "DROP_BASE")
        if np.any(drop_offset != 0):
            self.robot.move_l_rel(self.rc, drop_offset, L_VEL, L_ACC, rb.ReferenceFrame.Base)
            self.wait_move("SHIFT")

        self.robot.move_l_rel(self.rc, np.array([0.0,0.0,DROP_DOWN_MM,0.0,0.0,0.0]), L_VEL, L_ACC, rb.ReferenceFrame.Tool)
        self.wait_move("DROP_DESCEND")
        
        self.call_gripper(self.open_client, "RELEASE")
        
        self.robot.move_l_rel(self.rc, np.array([0.0,0.0,DROP_UP_MM,0.0,0.0,0.0]), L_VEL, L_ACC, rb.ReferenceFrame.Tool)
        self.wait_move("DROP_ASCEND")

        self.go_home()
        return "SUCCESS"

    # --- 제어 함수 ---

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
    node = UnloadNode()
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