🛠️ Requirements
- Ubuntu 22.04
- ROS2 Humble
- Intel RealSense D435
- Python 3.10+
- OpenCV
- Open3D
- PyTorch (YOLO)

⚡ Alias 설정 (권장)

.bashrc에 아래 내용을 추가하면 명령어를 간단히 사용할 수 있습니다.

alias rb='source ~/.bashrc'
alias sb='source /opt/ros/humble/setup.bash'
alias is='source ~/robocup_ws/install/local_setup.bash'
alias cbc='rm -rf build install log && colcon build --symlink-install'
