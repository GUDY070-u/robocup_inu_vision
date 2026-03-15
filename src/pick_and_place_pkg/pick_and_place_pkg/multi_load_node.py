#!/usr/bin/env python3
import threading
import time
import numpy as np
import math

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

import rbpodo as rb
from std_srvs.srv import Trigger
from msgs_pkg.srv import GetObjectPose, RunWS


# -----------------------------
# 설정값
# -----------------------------
ROBOT_IP = "10.0.2.7"

HOME_JOINT_DEG = np.array([-90.0, 0.0, 90.0, 0.0, 90.0, 0.0])

CAM_TO_TCP_OFFSET_X_MM = -51.0
CAM_TO_TCP_OFFSET_Y_MM = 32.0

Z_APPROACH_MM = 365.0
SAFE_RETRACT_MM = -100.0

J_VEL, J_ACC = 255, 255
L_VEL, L_ACC = 500, 800

VISION_TO_GRIPPER_YAW_OFFSET_DEG = -90.0


class LoadNode(Node):

    def __init__(self):
        super().__init__("load_node")

        self.get_logger().info(
            "✅ Pick Test Node Ready (Gripper Disabled, XY transform after wrist rotation fixed)"
        )

        self.callback_group = ReentrantCallbackGroup()

        try:
            self.robot = rb.Cobot(ROBOT_IP)
            self.rc = rb.ResponseCollector()
            self.robot.set_operation_mode(self.rc, rb.OperationMode.Real)
            self.get_logger().info("🤖 Robot Connected")
        except Exception as e:
            self.get_logger().error(f"❌ Robot Connection Error: {e}")

        # -----------------------------
        # Gripper 서비스 (현재 사용 안함)
        # -----------------------------
        self.open_client = self.create_client(
            Trigger, "/gripper/open", callback_group=self.callback_group
        )

        self.grip_client = self.create_client(
            Trigger, "/gripper/grip", callback_group=self.callback_group
        )

        # -----------------------------
        # Vision 서비스
        # -----------------------------
        self.pose_client = self.create_client(
            GetObjectPose, "/vision/get_object_pose", callback_group=self.callback_group
        )

        # -----------------------------
        # 작업 서비스
        # -----------------------------
        self.srv = self.create_service(
            RunWS, "/task/load3", self.cb_load3, callback_group=self.callback_group
        )

        self._busy_lock = threading.Lock()
        self._busy = False

    # -----------------------------
    # 각도 정규화
    # -----------------------------
    def normalize_angle_to_gripper_range(self, angle_deg):

        while angle_deg > 90.0:
            angle_deg -= 180.0

        while angle_deg < -90.0:
            angle_deg += 180.0

        return angle_deg

    # -----------------------------
    # 서비스 콜백
    # -----------------------------
    def cb_load3(self, req, res):

        with self._busy_lock:

            if self._busy:
                res.success = False
                res.message = "현재 다른 작업이 진행 중입니다. 잠시만 기다려주세요."
                return res

            self._busy = True

        try:

            result, pose_msg = self.pick_one_object()

            if result == "SUCCESS":
                res.success = True
                res.message = f"물체 위치 이동 성공 | {pose_msg}"

            elif result == "NO_ITEM":
                res.success = False
                res.message = "물체를 찾지 못했습니다."

            else:
                res.success = False
                res.message = "작업 실패"

            return res

        except Exception as e:

            self.get_logger().error(f"❌ Exception: {e}")

            try:
                self.go_home()
            except Exception:
                pass

            res.success = False
            res.message = f"Exception: {e}"

            return res

        finally:

            with self._busy_lock:
                self._busy = False

    # -----------------------------
    # 물체 위치 이동 테스트
    # -----------------------------
    def pick_one_object(self):

        # 1. HOME 이동
        self.go_home_forced(1.0)

        # 2. 관찰 안정화
        self.get_logger().info("👀 관찰 위치 도착. 1초 대기 후 비전 호출")
        time.sleep(1.0)

        # 3. 비전 포즈 요청
        pose = self.call_pose(10.0)

        if pose is None:
            self.go_home()
            return "NO_ITEM", "No pose"

        # 4. 비전 yaw -> 그리퍼 yaw
        vision_yaw_deg = float(pose.rz)
        gripper_yaw_deg = vision_yaw_deg + VISION_TO_GRIPPER_YAW_OFFSET_DEG
        gripper_yaw_deg = self.normalize_angle_to_gripper_range(gripper_yaw_deg)

        pose_msg = (
            f"ID:{pose.detected_id}, "
            f"X:{pose.x:.4f}m, Y:{pose.y:.4f}m, Z:{pose.z:.4f}m, "
            f"VisionYaw:{vision_yaw_deg:.2f}deg, GripperYaw:{gripper_yaw_deg:.2f}deg"
        )

        self.get_logger().info(
            f"📐 Vision Yaw: {vision_yaw_deg:.2f} deg -> Gripper Yaw: {gripper_yaw_deg:.2f} deg"
        )

        # 5. 관찰 자세 기준 raw XY 이동량 계산
        raw_x_move = (pose.y * 1000.0) + CAM_TO_TCP_OFFSET_X_MM
        raw_y_move = -(pose.x * 1000.0) + CAM_TO_TCP_OFFSET_Y_MM

        self.get_logger().info(
            f"📍 Raw Move(mm, observation frame) -> X:{raw_x_move:.2f}, Y:{raw_y_move:.2f}"
        )

        # 6. 손목 회전
        target_j = HOME_JOINT_DEG.copy()
        target_j[5] = HOME_JOINT_DEG[5] + gripper_yaw_deg

        self.robot.move_j(self.rc, target_j, J_VEL, J_ACC)
        self.wait_move()

        # 7. 실제 손목 회전량 기준으로 XY를 Tool frame에 맞게 변환
        # 관찰 시점 기준 XY를, 회전된 Tool 기준 상대이동으로 다시 표현
        wrist_delta_deg = target_j[5] - HOME_JOINT_DEG[5]
        rad = math.radians(wrist_delta_deg)

        final_x_move = raw_x_move * math.cos(rad) + raw_y_move * math.sin(rad)
        final_y_move = -raw_x_move * math.sin(rad) + raw_y_move * math.cos(rad)

        self.get_logger().info(
            f"🔄 Wrist Delta: {wrist_delta_deg:.2f} deg"
        )
        self.get_logger().info(
            f"📍 Final Tool Move(mm, rotated tool frame) -> X:{final_x_move:.2f}, Y:{final_y_move:.2f}"
        )

        # 8. XY 정렬
        self.robot.move_l_rel(
            self.rc,
            np.array([final_x_move, final_y_move, 0.0, 0.0, 0.0, 0.0]),
            L_VEL,
            L_ACC,
            rb.ReferenceFrame.Tool
        )
        self.wait_move()

        # 9. Z 접근
        self.robot.move_l_rel(
            self.rc,
            np.array([0.0, 0.0, Z_APPROACH_MM, 0.0, 0.0, 0.0]),
            L_VEL,
            L_ACC,
            rb.ReferenceFrame.Tool
        )
        self.wait_move()

        self.get_logger().info("📍 물체 위치 도착 (그리퍼 동작 생략)")

        time.sleep(2.0)

        # 10. 안전하게 위로 이동
        self.robot.move_l_rel(
            self.rc,
            np.array([0.0, 0.0, SAFE_RETRACT_MM, 0.0, 0.0, 0.0]),
            L_VEL,
            L_ACC,
            rb.ReferenceFrame.Tool
        )
        self.wait_move()

        # 11. HOME 복귀
        self.go_home()

        return "SUCCESS", pose_msg

    # -----------------------------
    # Robot Motion
    # -----------------------------
    def go_home_forced(self, timeout):

        self.get_logger().info(f"🏠 Moving to HOME ({timeout}s wait)")

        self.robot.move_j(self.rc, HOME_JOINT_DEG, J_VEL, J_ACC)

        time.sleep(timeout)

    def go_home(self):

        self.get_logger().info("🏠 Moving to HOME")

        self.robot.move_j(self.rc, HOME_JOINT_DEG, J_VEL, J_ACC)

        self.wait_move()

    def wait_move(self):

        self.robot.wait_for_move_finished(self.rc)

    # -----------------------------
    # Vision
    # -----------------------------
    def call_pose(self, timeout):

        req = GetObjectPose.Request()

        future = self.pose_client.call_async(req)

        start = time.time()

        while rclpy.ok() and (time.time() - start < timeout):

            if future.done():

                res = future.result()

                if res and res.success:
                    return res

                return None

            time.sleep(0.1)

        return None


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


if __name__ == "__main__":
    main()