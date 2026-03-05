#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    # 0) 초기화
    init_loaded_count = ExecuteProcess(
        cmd=["bash", "-lc", "echo 0 > /tmp/loaded_count.txt"],
        output="screen",
    )

    # 1) 카메라 (RealSense)
    realsense_launch = ExecuteProcess(
        cmd=["ros2", "launch", "realsense2_camera", "rs_launch.py", "align_depth.enable:=true"],
        output="screen",
    )

    # 2) yolo_3d 서비스
    yolo_3d = Node(
        package="vision",
        executable="yolo_3d_node",
        name="yolo_3d_node",
        output="screen",
    )

    # 3) 그리퍼 노드 (권한 해결됨)
    gripper = Node(
        package="pick_and_place_pkg",
        executable="gripper_node",
        name="gripper_node",
        output="screen",
    )

    # 4) Load 노드
    multi_load = Node(
        package="pick_and_place_pkg",
        executable="multi_load_node",
        name="multi_load_node",
        output="screen",
    )

    # 5) Unload 노드
    multi_unload = Node(
        package="pick_and_place_pkg",
        executable="multi_unload_node",
        name="multi_unload_node",
        output="screen",
    )

    # 모든 노드를 동시에 실행 (체인 제거)
    return LaunchDescription([
        init_loaded_count,
        realsense_launch,
        yolo_3d,
        gripper,
        multi_load,
        multi_unload,
    ])
