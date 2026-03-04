#!/usr/bin/env python3
import threading
import time
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from msgs_pkg.srv import RunWS

class WorkcellCoordinator(Node):
    def __init__(self):
        super().__init__("workcell_coordinator")

        # 1. 구독 및 발행 (ReentrantCallbackGroup 적용 권장)
        self.rx_sub = self.create_subscription(String, "/serial_rx", self.on_rx, 10)
        self.tx_pub = self.create_publisher(String, "/serial_tx", 10)

        # 2. 서비스 클라이언트
        self.load_cli = self.create_client(RunWS, "/task/load3")
        self.unload_cli = self.create_client(RunWS, "/task/unload3")

        self.get_logger().info("🔍 Waiting for load3 and unload3 services...")
        self.load_cli.wait_for_service()
        self.unload_cli.wait_for_service()
        self.get_logger().info("✅ All Services Ready. Coordinator Active.")

        self._busy_lock = threading.Lock()
        self._busy = False

    def on_rx(self, msg: String):
        raw = msg.data.strip()
        self.get_logger().info(f"📩 RX Received: '{raw}'")

        # 1. 문자열 파싱 (더 유연하게 수정)
        line = "".join(ch for ch in raw if ch.isprintable()).strip()
        parts = [p.strip() for p in line.split(",") if p.strip()]

        if len(parts) < 3 or parts[0].upper() != "ARRIVED":
            return

        ws_token = parts[1].upper()  # "WS1"
        job_token = parts[2].upper() # "PICK3" or "PLACE3"

        if "PICK" in job_token:
            job = "PICK"
        elif "PLACE" in job_token:
            job = "PLACE"
        else:
            self.get_logger().warn(f"❓ Unknown Job: {job_token}")
            return

        # 2. 작업 중복 방지
        with self._busy_lock:
            if self._busy:
                self.get_logger().warn("⚠️ Coordinator is BUSY. Ignoring request.")
                return
            self._busy = True

        # 3. 별도 스레드에서 서비스 호출 (중요: 여기서 안 움직이면 Thread 문제일 가능성 높음)
        self.get_logger().info(f"🚀 [START JOB] {job} at {ws_token}")
        t = threading.Thread(target=self._run_job, args=(ws_token, job, job_token))
        t.daemon = True
        t.start()

    def _run_job(self, ws_token, job, last_raw):
        try:
            req = RunWS.Request()
            # "WS1"에서 숫자만 추출
            try:
                req.ws = int("".join(filter(str.isdigit, ws_token)))
            except:
                req.ws = 1

            # 작업에 맞는 클라이언트 선택
            cli = self.load_cli if job == "PICK" else self.unload_cli
            
            # 서비스 호출 및 대기
            future = cli.call_async(req)
            
            # 서비스 응답을 기다리는 동안 블로킹되지 않도록 루프 처리
            while rclpy.ok() and not future.done():
                time.sleep(0.1)

            res = future.result()
            if res and res.success:
                self.get_logger().info(f"✅ Job SUCCESS: {ws_token}, {last_raw}")
                # 완료 신호 전송
                out = String()
                out.data = f"DONE,{ws_token},{last_raw}"
                self.tx_pub.publish(out)
            else:
                self.get_logger().error(f"❌ Job FAILED or Response None")

        except Exception as e:
            self.get_logger().error(f"❗ Error in _run_job: {e}")
        finally:
            with self._busy_lock:
                self._busy = False
            self.get_logger().info("🔓 Coordinator is now FREE")

def main(args=None):
    rclpy.init(args=args)
    node = WorkcellCoordinator()
    # 넉넉한 스레드 할당
    executor = MultiThreadedExecutor(num_threads=4)
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
