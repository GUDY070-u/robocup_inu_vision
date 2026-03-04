#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch_ros.actions import Node


def generate_launch_description():
    # 0) loaded_count 초기화
    init_loaded_count = ExecuteProcess(
        cmd=["bash", "-lc", "echo 0 > /tmp/loaded_count.txt && echo '[INIT] loaded_count = 0'"],
        output="screen",
    )

    # 1) 카메라 (너가 쓰는 명령 그대로) // realsense_launch 부분 수정
    realsense_launch = ExecuteProcess(
    cmd=["ros2", "launch", "realsense2_camera", "rs_launch.py", "align_depth:=true"],
    output="screen",
    )

    # 2) 시리얼 브릿지
    serial_flag_bridge = Node(
        package="serial_pkg",
        executable="serial_flag_bridge",
        name="serial_flag_bridge",
        output="screen",
    )

    # 3) coordinator
    workcell_coordinator = Node(
        package="serial_pkg",
        executable="workcell_coordinator",
        name="workcell_coordinator",
        output="screen",
    )

    # 4) ArUco
    multi_aruco = Node(
        package="pick_and_place_pkg",
        executable="multi_aruco_pose_service_node",
        name="multi_aruco_pose_service_node",
        output="screen",
    )

    # #4) 3d vision
    # vision = Node(
    #     package="pick_and_place_pkg",
    #     executable="multi_aruco_pose_service_node",
    #     name="multi_aruco_pose_service_node",
    #     output="screen",
    # )

    # 5) 그리퍼
    gripper = Node(
        package="pick_and_place_pkg",
        executable="gripper_node",
        name="gripper_node",
        output="screen",
    )

    # 6) load
    multi_load = Node(
        package="pick_and_place_pkg",
        executable="multi_load_node",
        name="multi_load_node",
        output="screen",
    )

    # 7) unload
    multi_unload = Node(
        package="pick_and_place_pkg",
        executable="multi_unload_node",
        name="multi_unload_node",
        output="screen",
    )

    # ✅ 순차 실행 체인
    chain_0 = RegisterEventHandler(
        OnProcessStart(target_action=init_loaded_count, on_start=[realsense_launch])
    )
    chain_1 = RegisterEventHandler(
        OnProcessStart(target_action=realsense_launch, on_start=[serial_flag_bridge])
    )
    chain_2 = RegisterEventHandler(
        OnProcessStart(target_action=serial_flag_bridge, on_start=[workcell_coordinator])
    )
    chain_3 = RegisterEventHandler(
        OnProcessStart(target_action=workcell_coordinator, on_start=[multi_aruco])
    )
    chain_4 = RegisterEventHandler(
        OnProcessStart(target_action=multi_aruco, on_start=[gripper])
        # OnProcessStart(target_action=vision, on_start=[gripper])
    )
    chain_5 = RegisterEventHandler(
        OnProcessStart(target_action=gripper, on_start=[multi_load])
    )
    chain_6 = RegisterEventHandler(
        OnProcessStart(target_action=multi_load, on_start=[multi_unload])
    )

    return LaunchDescription([
        init_loaded_count,
        chain_0,
        chain_1,
        chain_2,
        chain_3,
        chain_4,
        chain_5,
        chain_6,
    ])
