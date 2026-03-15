#!/usr/bin/env python3
import threading
import time
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from msgs_pkg.srv import RunWS


class ManualWorkcellCoordinator(Node):
    def __init__(self):
        super().__init__("manual_coordinator")

        # 1. 발행 (결과 보고용)
        self.tx_pub = self.create_publisher(String, "/serial_tx", 10)

        # 2. 서비스 클라이언트
        self.load_cli = self.create_client(RunWS, "/task/load3")
        self.unload_cli = self.create_client(RunWS, "/task/unload3")

        self.get_logger().info("🔍 서비스 연결 대기 중 (/task/load3, /task/unload3)...")
        while not self.load_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("서비스가 아직 활성화되지 않았습니다 (load3)...")
        while not self.unload_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("서비스가 아직 활성화되지 않았습니다 (unload3)...")

        self.get_logger().info("✅ 모든 서비스 준비 완료. 명령을 입력할 수 있습니다.")

        self._busy_lock = threading.Lock()
        self._busy = False

        # 입력을 받기 위한 별도 스레드 시작
        self.input_thread = threading.Thread(target=self.terminal_input_loop)
        self.input_thread.daemon = True
        self.input_thread.start()

    def terminal_input_loop(self):
        while rclpy.ok():
            print("\n" + "=" * 50)
            print("명령어 예시: '1 pick' 또는 '2 place' (종료: q)")
            user_input = input(">> 명령 입력: ").strip().lower()

            if user_input == 'q':
                rclpy.shutdown()
                break

            try:
                parts = user_input.split()
                if len(parts) < 2:
                    print("❌ 잘못된 형식입니다. [번호] [작업] 형태로 입력하세요.")
                    continue

                ws_num = int(parts[0])
                job_type = parts[1]

                if "pick" in job_type:
                    job = "PICK"
                elif "place" in job_type:
                    job = "PLACE"
                else:
                    print(f"❌ 알 수 없는 작업: {job_type}")
                    continue

                self.request_job(ws_num, job)

            except ValueError:
                print("❌ 워크스테이션 번호는 숫자로 입력해야 합니다.")
            except Exception as e:
                print(f"❌ 에러 발생: {e}")

    def request_job(self, ws_num, job):
        with self._busy_lock:
            if self._busy:
                print("⚠️ 현재 다른 작업이 진행 중입니다. 잠시만 기다려주세요.")
                return
            self._busy = True

        print(f"🚀 [작업 시작] WS{ws_num}에서 {job} 수행")

        t = threading.Thread(target=self._run_job_logic, args=(ws_num, job))
        t.daemon = True
        t.start()

    def _run_job_logic(self, ws_num, job):
        try:
            req = RunWS.Request()
            req.ws = ws_num

            cli = self.load_cli if job == "PICK" else self.unload_cli
            future = cli.call_async(req)

            while rclpy.ok() and not future.done():
                time.sleep(0.1)

            res = future.result()
            if res and res.success:
                self.get_logger().info(f"✅ 작업 성공: WS{ws_num}, {job}")

                # 실제 사용한 pose는 load node가 응답 message에 넣어줌
                if job == "PICK" and res.message:
                    print("\n🎯 Used Pick Pose")
                    print(res.message + "\n")

                out = String()
                out.data = f"DONE,WS{ws_num},{job}3"
                self.tx_pub.publish(out)
            else:
                err_msg = res.message if res else "서비스 응답이 없습니다."
                self.get_logger().error(f"❌ 작업 실패: {err_msg}")

        except Exception as e:
            self.get_logger().error(f"❗ 실행 중 에러 발생: {e}")
        finally:
            with self._busy_lock:
                self._busy = False
            print("🔓 이제 다음 명령을 입력할 수 있습니다.")


def main(args=None):
    rclpy.init(args=args)
    node = ManualWorkcellCoordinator()
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