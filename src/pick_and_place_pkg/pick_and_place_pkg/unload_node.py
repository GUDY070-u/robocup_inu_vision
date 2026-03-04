import threading
import time
import numpy as np
import rclpy
from rclpy.node import Node

import rbpodo as rb
from std_srvs.srv import Trigger


# ===============================
# 설정값
# ===============================
ROBOT_IP = "10.0.2.7"

HOME_JOINT_DEG  = np.array([-90.0,   0.0,  90.0, 0.0,  90.0, 0.0])
POSE_CARGO      = np.array([-90.0,  -7.89, -42.07, 0.0, -130.0, 0.0])
POSE_DROP       = np.array([-91.25, 55.35,  66.41, 0.0,  58.24, -1.33])
POSE_FINAL      = np.array([-90.0, -35.0, 125.0, 0.0, 90.0, 0.0])

# Z 이동 값 (mm)
CARGO_DOWN_MM = 60.0
CARGO_UP_MM   = -70.0
DROP_DOWN_MM  = 50.0
DROP_UP_MM    = -50.0

# 속도 / 가속
J_VEL, J_ACC = 255, 255
L_VEL, L_ACC = 500, 800

MOVE_START_TIMEOUT_SEC = 1.0
RETRY_SLEEP_SEC = 1.0


# ===============================
# Unload Node
# ===============================
class UnloadNode(Node):

    def __init__(self):
        super().__init__("unload_node")
        self.get_logger().info("✅ Unload Node Started")

        # Robot
        self.robot = rb.Cobot(ROBOT_IP)
        self.rc = rb.ResponseCollector()
        self.robot.set_operation_mode(self.rc, rb.OperationMode.Real)
        self.robot.set_speed_bar(self.rc, 1.0)

        # Gripper Clients
        self.open_client = self.create_client(Trigger, "/gripper/open")
        self.grip_client = self.create_client(Trigger, "/gripper/grip")

        self.open_client.wait_for_service()
        self.grip_client.wait_for_service()

        threading.Thread(target=self.sequence, daemon=True).start()

    # ==================================================
    # High-level sequence
    # ==================================================
    def sequence(self):
        while rclpy.ok():
            self.get_logger().info("🔽 Start Unload Sequence")

            if self.run_once():
                self.get_logger().info("🎉 UNLOAD SUCCESS")
                break

            self.get_logger().warn(f"🔁 Unload failed → retry in {RETRY_SLEEP_SEC}s")
            time.sleep(RETRY_SLEEP_SEC)

    def run_once(self) -> bool:
        return (
            self.go_cargo() and
            self.approach_and_grip_cargo() and
            self.go_drop() and
            self.approach_and_release_drop() and
            self.go_home() and
            self.go_final_pose()
        )

    # ==================================================
    # Sequence blocks
    # ==================================================
    def approach_and_grip_cargo(self) -> bool:
        if not self.move_l_rel(CARGO_DOWN_MM, "CARGO_DOWN"):
            return False

        if not self.call_gripper(self.grip_client, "GRIP_FROM_CARGO"):
            return False

        if not self.move_l_rel(CARGO_UP_MM, "CARGO_UP"):
            return False

        return True

    def approach_and_release_drop(self) -> bool:
        if not self.move_l_rel(DROP_DOWN_MM, "DROP_DOWN"):
            return False

        if not self.call_gripper(self.open_client, "RELEASE_AT_DROP"):
            return False

        if not self.move_l_rel(DROP_UP_MM, "DROP_UP"):
            return False

        return True

    # ==================================================
    # Primitive motions
    # ==================================================
    def go_cargo(self) -> bool:
        self.get_logger().info("➡️ Move to CARGO")
        self.robot.move_j(self.rc, POSE_CARGO, J_VEL, J_ACC)
        return self.wait_move("CARGO_MOVE")

    def go_drop(self) -> bool:
        self.get_logger().info("➡️ Move to DROP")
        self.robot.move_j(self.rc, POSE_DROP, J_VEL, J_ACC)
        return self.wait_move("DROP_MOVE")

    def go_home(self) -> bool:
        self.get_logger().info("➡️ Move to HOME")
        self.robot.move_j(self.rc, HOME_JOINT_DEG, J_VEL, J_ACC)
        return self.wait_move("HOME_MOVE")

    def go_final_pose(self) -> bool:
        self.get_logger().info("➡️ Move to FINAL POSE")
        self.robot.move_j(self.rc, POSE_FINAL, J_VEL, J_ACC)
        return self.wait_move("FINAL_POSE_MOVE")

    def move_l_rel(self, dz_mm: float, name: str) -> bool:
        self.get_logger().info(f"↕️ {name}: dz={dz_mm:.1f}mm")
        self.robot.move_l_rel(
            self.rc,
            np.array([0.0, 0.0, dz_mm, 0.0, 0.0, 0.0], dtype=float),
            L_VEL,
            L_ACC,
            rb.ReferenceFrame.Tool
        )
        return self.wait_move(name)

    # ==================================================
    # Utilities
    # ==================================================
    def call_gripper(self, client, name, timeout_sec=5.0) -> bool:
        req = Trigger.Request()
        future = client.call_async(req)

        done_evt = threading.Event()
        future.add_done_callback(lambda f: done_evt.set())

        if not done_evt.wait(timeout_sec):
            self.get_logger().error(f"❌ {name} TIMEOUT")
            return False

        res = future.result()
        if res and res.success:
            self.get_logger().info(f"✅ {name} OK")
            return True

        self.get_logger().error(f"❌ {name} FAIL")
        return False

    def wait_move(self, step_name: str) -> bool:
        if self.robot.wait_for_move_started(self.rc, MOVE_START_TIMEOUT_SEC).is_success():
            self.robot.wait_for_move_finished(self.rc)
            self.get_logger().info(f"✅ {step_name} done")
            return True

        self.get_logger().error(f"❌ {step_name} start timeout")
        return False


# ===============================
def main(args=None):
    rclpy.init(args=args)
    node = UnloadNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()