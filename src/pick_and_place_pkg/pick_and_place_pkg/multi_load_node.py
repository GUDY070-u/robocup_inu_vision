import threading
import time
import os
import numpy as np
import math  # 필수

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

import rbpodo as rb
from std_srvs.srv import Trigger
from msgs_pkg.srv import GetObjectPose, RunWS

# 설정값
ROBOT_IP = "10.0.2.7"
COUNT_FILE = "/tmp/loaded_count.txt"
HOME_JOINT_DEG = np.array([-90.0, 0.0, 90.0, 0.0, 90.0, 0.0])
POSE_FINAL = np.array([-90.0, -22.08, 118.94, -0.33, 84.91, 0.0])
CARGO_POSES = [
    np.array([-65.31, -16.33, -31.69, -0.01, -131.95, 24.62]), # Slot 1
    np.array([-93.25, -7.91, -42.05, 0.0, -130.0, -3.32]),    # Slot 2
    np.array([-117.67, -16.88, -30.94, 0.04, -132.13, -27.71]) # Slot 3
]

CAM_TO_TCP_OFFSET_X_MM = -51.0
CAM_TO_TCP_OFFSET_Y_MM = 32.0
Z_APPROACH_MM = 365.0
Z_DOWN_MM = 20.0
Z_UP_MM = -20.0

J_VEL, J_ACC = 255, 255
L_VEL, L_ACC = 500, 800

class LoadNode(Node):
    def __init__(self):
        super().__init__("load_node")
        self.get_logger().info("✅ Load Node Ready (Rotation Limit -90~90)")

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

            if current_count >= 3:
                self.get_logger().warn("⚠️ Tray is already FULL. Skipping logic.")
                res.success = True
                res.message = "Tray Full"
                return res

            total = self.sequence_load(current_count)
            
            res.success = True
            res.message = f"Total count in tray: {total}"
            return res

        except Exception as e:
            self.get_logger().error(f"❌ Exception in cb_load3: {e}")
            res.success = False
            return res

        finally:
            with self._busy_lock: 
                self._busy = False
            self.get_logger().info("🔓 Busy Lock Released.")


    def sequence_load(self, start_count):
        self.call_gripper(self.open_client, "OPEN")
        self.go_home_forced(timeout=1.0) 

        newly_loaded = 0
        
        for i in range(start_count, len(CARGO_POSES)):
            self.get_logger().info(f"🚀 [Slot #{i+1}] 타겟 탐색 시작...")
            
            found_target = False
            retry_limit = 3         
            retry_delay = 0.5       
            
            for attempt in range(retry_limit):
                result = self.run_once(CARGO_POSES[i]) 
                
                if result == "SUCCESS":
                    self.get_logger().info(f"✅ Slot #{i+1} 적재 완료!")
                    newly_loaded += 1
                    found_target = True
                    break 
                
                elif result == "GRIP_FAILED":
                    self.get_logger().error("🛑 잡기 실패. 시퀀스 중단.")
                    self.go_final_pose()
                    return start_count + newly_loaded
                
                elif result == "NO_ITEM":
                    self.get_logger().warn(f"⚠️ [시도 {attempt+1}/{retry_limit}] 물체 없음. {retry_delay}초 후 재시도...")
                    time.sleep(retry_delay)
            
            if not found_target:
                self.get_logger().info(f"🚫 Slot #{i+1} 최종 실패. 시퀀스 종료.")
                break 

        updated_total = start_count + newly_loaded
        
        with open(COUNT_FILE, "w") as f:
            f.write(str(updated_total))
            
        self.get_logger().info(f"💾 최종 적재량: {updated_total}/3")
        self.go_final_pose()
        return updated_total

    def run_once(self, cargo_target):
        # 1. 포즈 요청
        pose = self.call_pose(10.0)
        if pose is None: return "NO_ITEM"

        # -----------------------------------------------------------
        # [수정] 각도 정규화 (손목 꺾임 방지)
        # -90 ~ +90 도 사이로 강제 변환
        # -----------------------------------------------------------
        angle_deg = float(pose.rz)

        # 180도 이상 차이나면 계속 180도를 빼거나 더해서 -90~90 범위로 맞춤
        # (그리퍼가 180도 대칭이라고 가정)
        while angle_deg > 90.0:
            angle_deg -= 180.0
        while angle_deg < -90.0:
            angle_deg += 180.0

        self.get_logger().info(f"📐 Adjusted Angle: {pose.rz:.1f} -> {angle_deg:.1f}")

        # -----------------------------------------------------------
        # [이전 로직] 회전 보정 및 이동 계산
        # 정규화된 angle_deg를 사용하여 계산하므로 완벽하게 작동함
        # -----------------------------------------------------------
        raw_x_move = (pose.y * 1000.0) + CAM_TO_TCP_OFFSET_X_MM
        raw_y_move = -(pose.x * 1000.0) + CAM_TO_TCP_OFFSET_Y_MM
        
        rad = math.radians(-angle_deg) 
        
        final_x_move = raw_x_move * math.cos(rad) - raw_y_move * math.sin(rad)
        final_y_move = raw_x_move * math.sin(rad) + raw_y_move * math.cos(rad)

        # -----------------------------------------------------------
        # 로봇 구동
        # -----------------------------------------------------------

        # (1) 손목 회전 (정규화된 각도 사용)
        target_j = HOME_JOINT_DEG.copy()
        target_j[5] += angle_deg
        self.robot.move_j(self.rc, target_j, J_VEL, J_ACC)
        time.sleep(1.0) 

        # (2) XY 이동
        self.robot.move_l_rel(self.rc, np.array([final_x_move, final_y_move, 0.0, 0.0, 0.0, 0.0]), L_VEL, L_ACC, rb.ReferenceFrame.Tool)
        self.wait_move("ALIGN_XY")

        # (3) Z축 접근
        self.robot.move_l_rel(self.rc, np.array([0.0, 0.0, Z_APPROACH_MM, 0.0, 0.0, 0.0]), L_VEL, L_ACC, rb.ReferenceFrame.Tool)
        self.wait_move("APPROACH")

        # (4) 잡기
        if not self.call_gripper(self.grip_client, "GRIP"):
            self.get_logger().warn("Grip failed! Returning to safe pose.")
            self.robot.move_l_rel(self.rc, np.array([0.0, 0.0, -100.0, 0.0, 0.0, 0.0]), L_VEL, L_ACC, rb.ReferenceFrame.Tool)
            self.wait_move("EMERGENCY_UP")
            self.go_home()
            return "GRIP_FAILED"
        
        # (5) 적재
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
                if res and res.success: 
                    return res
                else: 
                    self.get_logger().warn("Vision returned FAIL (No object found).")
                    return None
            time.sleep(0.1)
        self.get_logger().warn(f"Vision Service Call TIMEOUT ({timeout}s)")
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
