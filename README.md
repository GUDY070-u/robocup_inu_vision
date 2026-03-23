# 🛠️ Requirements
- Ubuntu 22.04
- ROS2 Humble
- Intel RealSense D435
- Python 3.10+
- OpenCV
- Open3D
- PyTorch (YOLO)

## P.S : <ros2_ws> - 사용하는 로스 워크 스페이스 이름 변경 필요

# ⚡ Alias 설정 (권장)

.bashrc에 아래 내용을 추가하면 명령어를 간단히 사용할 수 있습니다.

    alias rb='source ~/.bashrc'
    alias sb='source /opt/ros/humble/setup.bash'
    alias is='source ~/<ros2_ws>/install/local_setup.bash'
    alias cbc='rm -rf build install log && colcon build --symlink-install'

| Alias | 설명              |
| ----- | --------------- |
| rb    | bashrc 다시 적용    |
| sb    | ROS2 환경 설정      |
| is    | workspace 환경 설정 |
| cbc   | 클린 빌드           |

# 🔧 Build 방법
    cd ~/<ros2_ws>
    rb
    sb
    is
    cbc
