import rclpy
from rclpy.node import Node

import serial
import time

from std_srvs.srv import Trigger


# ===============================
# 사용자 설정
# ===============================
GRIPPER_PORT = "/dev/ttyACM0"
GRIPPER_BAUD = 115200


# ===============================
# Gripper Node
# ===============================
class GripperNode(Node):

    def __init__(self):
        super().__init__("gripper_node")

        self.get_logger().info("✅ Gripper Node Started")

        # -------------------------------
        # Serial Connect
        # -------------------------------
        try:
            self.ser = serial.Serial(
                GRIPPER_PORT,
                GRIPPER_BAUD,
                timeout=1
            )
            time.sleep(2)
            self.get_logger().info("✅ OpenCR Gripper Connected")

        except Exception as e:
            self.ser = None
            self.get_logger().error(f"❌ Gripper Connection Failed: {e}")

        # -------------------------------
        # Services
        # -------------------------------
        self.srv_open = self.create_service(
            Trigger,
            "/gripper/open",
            self.handle_open
        )

        self.srv_grip = self.create_service(
            Trigger,
            "/gripper/grip",
            self.handle_grip
        )


    # ===============================
    # Utility: Wait Result
    # ===============================
    def wait_grasp_result(self, timeout=5.0):

        if self.ser is None:
            return False, "Serial not connected"

        start = time.time()

        while time.time() - start < timeout:

            if self.ser.in_waiting:

                line = self.ser.readline().decode(errors="ignore").strip()

                if "[RESULT] GRASP_OK" in line:
                    return True, "GRASP_OK"

                elif "[RESULT] GRASP_FAIL" in line:
                    return False, "GRASP_FAIL"

            time.sleep(0.01)

        return False, "Timeout"


    # ===============================
    # Service Callback: Open
    # ===============================
    def handle_open(self, request, response):

        if self.ser is None:
            response.success = False
            response.message = "Gripper not connected"
            return response

        self.ser.write(b"open\n")
        self.get_logger().info("📌 Sent: open")

        time.sleep(1.0)

        response.success = True
        response.message = "Gripper opened"
        return response


    # ===============================
    # Service Callback: Grip
    # ===============================
    def handle_grip(self, request, response):

        if self.ser is None:
            response.success = False
            response.message = "Gripper not connected"
            return response

        self.ser.write(b"grip\n")
        self.get_logger().info("📌 Sent: grip (waiting result...)")

        success, msg = self.wait_grasp_result(timeout=5.0)

        response.success = success
        response.message = msg
        return response


    # ===============================
    # Shutdown Cleanup
    # ===============================
    def destroy_node(self):

        if self.ser and self.ser.is_open:
            self.ser.close()
            self.get_logger().info("✅ Serial Closed")

        super().destroy_node()


# ===============================
# Main
# ===============================
def main(args=None):

    rclpy.init(args=args)

    node = GripperNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()