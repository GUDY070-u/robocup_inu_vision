import threading
import time
import numpy as np
import rclpy
from rclpy.node import Node

import rbpodo as rb
from std_srvs.srv import Trigger
from msgs_pkg.srv import GetObjectPose


# ===============================
# 설정값
# ===============================
ROBOT_IP = "10.0.2.7"

# Joint poses
HOME_JOINT_DEG = np.array([-90.0, 0.0, 90.0, 0.0, 90.0, 0.0])
POSE_CARGO     = np.array([-90.0, -7.89, -42.07, 0.0, -130.0, 0.0])
POSE_FINAL     = np.array([-90.0, -35.0, 125.0, 0.0, 90.0, 0.0])

# Camera → TCP calibration offsets (mm)
CAM_TO_TCP_OFFSET_X_MM = -51.0
CAM_TO_TCP_OFFSET_Y_MM =  32.0

# Z movement (mm)
Z_APPROACH_MM = 365.0
Z_DOWN_MM     = 62.0
Z_UP_MM       = -70.0

# Unified speed / acceleration
J_VEL = 255
J_ACC = 255
L_VEL = 500
L_ACC = 800

MOVE_START_TIMEOUT_SEC = 1.0
RETRY_SLEEP_SEC = 1.0


# ===============================
# Load Node
# ===============================
class LoadNode(Node):

    def __init__(self):
        super().__init__("load_node")
        self.get_logger().info("✅ Load Node Started")

        # Robot
        self.robot = rb.Cobot(ROBOT_IP)
        self.rc = rb.ResponseCollector()
        self.robot.set_operation_mode(self.rc, rb.OperationMode.Real)

        # ROS clients
        self.open_client = self.create_client(Trigger, "/gripper/open")
        self.grip_client = self.create_client(Trigger, "/gripper/grip")
        self.pose_client = self.create_client(GetObjectPose, "/vision/get_object_pose")

        self.open_client.wait_for_service()
        self.grip_client.wait_for_service()
        self.pose_client.wait_for_service()

        threading.Thread(target=self.sequence, daemon=True).start()

    # ==================================================
    # ONE SEQUENCE
    # ==================================================
    def sequence(self):
        while rclpy.ok():
            self.get_logger().info("▶ START LOAD SEQUENCE")

            if self.run_once():
                self.get_logger().info("🎉 SEQUENCE SUCCESS")
                break

            self.get_logger().warn("🔁 SEQUENCE FAILED → retry")
            time.sleep(RETRY_SLEEP_SEC)

    def run_once(self) -> bool:
        # ----- PICK -----
        if not self.open_gripper():
            return False

        if not self.go_home():
            return False

        if not self.align_yaw():
            return False

        if not self.align_xy():
            return False

        if not self.approach_z():
            return False

        if not self.grasp():
            return False

        if not self.go_home():
            return False

        # ----- PLACE (CARGO) -----
        if not self.go_cargo():
            return False

        if not self.move_z(Z_DOWN_MM, "CARGO_DOWN"):
            return False

        if not self.open_gripper():
            return False

        if not self.move_z(Z_UP_MM, "CARGO_UP"):
            return False

        if not self.go_home():
            return False

        # ----- FINISH -----
        if not self.go_final_pose():
            return False

        return True

    # ==================================================
    # Joint motions
    # ==================================================
    def go_home(self) -> bool:
        self.get_logger().info("➡️ GO HOME")
        self.robot.move_j(self.rc, HOME_JOINT_DEG, J_VEL, J_ACC)
        return self.wait_move("HOME")

    def go_cargo(self) -> bool:
        self.get_logger().info("➡️ GO CARGO")
        self.robot.move_j(self.rc, POSE_CARGO, J_VEL, J_ACC)
        return self.wait_move("CARGO")

    def go_final_pose(self) -> bool:
        self.get_logger().info("➡️ GO FINAL")
        self.robot.move_j(self.rc, POSE_FINAL, J_VEL, J_ACC)
        return self.wait_move("FINAL")

    # ==================================================
    # Vision alignment
    # ==================================================
    def align_yaw(self) -> bool:
        pose = self.call_pose()
        if pose is None:
            return False

        target = HOME_JOINT_DEG.copy()
        target[5] += float(pose.rz)

        self.get_logger().info(f"➡️ Align YAW: rz={pose.rz:.2f}")
        self.robot.move_j(self.rc, target, J_VEL, J_ACC)
        return self.wait_move("ALIGN_YAW")

    def align_xy(self) -> bool:
        pose = self.call_pose()
        if pose is None:
            return False

        # camera → TCP offset applied here
        dx_mm = -(pose.x * 1000.0) + CAM_TO_TCP_OFFSET_Y_MM
        dy_mm =  (pose.y * 1000.0) + CAM_TO_TCP_OFFSET_X_MM

        self.get_logger().info(
            f"➡️ Align XY: cam=({pose.x*1000:.1f}, {pose.y*1000:.1f}) "
            f"offset=({CAM_TO_TCP_OFFSET_X_MM}, {CAM_TO_TCP_OFFSET_Y_MM}) "
            f"cmd(dx,dy)=({dx_mm:.1f}, {dy_mm:.1f})"
        )

        self.robot.move_l_rel(
            self.rc,
            np.array([dy_mm, dx_mm, 0.0, 0.0, 0.0, 0.0], dtype=float),
            L_VEL,
            L_ACC,
            rb.ReferenceFrame.Tool
        )
        return self.wait_move("ALIGN_XY")

    def approach_z(self) -> bool:
        self.get_logger().info("⬇️ APPROACH Z")
        self.robot.move_l_rel(
            self.rc,
            np.array([0.0, 0.0, Z_APPROACH_MM, 0.0, 0.0, 0.0], dtype=float),
            L_VEL,
            L_ACC,
            rb.ReferenceFrame.Tool
        )
        return self.wait_move("APPROACH_Z")

    # ==================================================
    # Z-only linear move
    # ==================================================
    def move_z(self, dz_mm: float, name: str) -> bool:
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
    # Gripper
    # ==================================================
    def open_gripper(self) -> bool:
        return self.call_gripper(self.open_client, "OPEN")

    def grasp(self) -> bool:
        return self.call_gripper(self.grip_client, "GRIP")

    # ==================================================
    # Utilities
    # ==================================================
    def call_pose(self, timeout_sec=5.0):
        req = GetObjectPose.Request()
        future = self.pose_client.call_async(req)

        evt = threading.Event()
        future.add_done_callback(lambda f: evt.set())

        if not evt.wait(timeout_sec):
            self.get_logger().error("❌ VISION TIMEOUT")
            return None

        res = future.result()
        if not res or not res.success:
            self.get_logger().error("❌ VISION FAIL")
            return None

        return res

    def call_gripper(self, client, name, timeout_sec=5.0) -> bool:
        req = Trigger.Request()
        future = client.call_async(req)

        evt = threading.Event()
        future.add_done_callback(lambda f: evt.set())

        if not evt.wait(timeout_sec):
            self.get_logger().error(f"❌ {name} TIMEOUT")
            return False

        res = future.result()
        if res and res.success:
            self.get_logger().info(f"✅ {name} OK")
            return True

        self.get_logger().error(f"❌ {name} FAIL")
        return False

    def wait_move(self, name: str) -> bool:
        if self.robot.wait_for_move_started(self.rc, MOVE_START_TIMEOUT_SEC).is_success():
            self.robot.wait_for_move_finished(self.rc)
            self.get_logger().info(f"✅ {name} DONE")
            return True

        self.get_logger().error(f"❌ {name} START TIMEOUT")
        return False


# ===============================
def main(args=None):
    rclpy.init(args=args)
    node = LoadNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()