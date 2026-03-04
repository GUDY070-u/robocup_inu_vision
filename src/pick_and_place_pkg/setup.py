from setuptools import find_packages, setup

package_name = 'pick_and_place_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='FCSL_mani_team',
    maintainer_email='han@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        'load_node = pick_and_place_pkg.load_node:main',
        'unload_node = pick_and_place_pkg.unload_node:main',
        'gripper_node = pick_and_place_pkg.gripper_node:main',
        'aruco_pose_service_node = pick_and_place_pkg.aruco_pose_service_node:main',
        'multi_load_node = pick_and_place_pkg.multi_load_node:main',
        'multi_unload_node = pick_and_place_pkg.multi_unload_node:main',
        'multi_aruco_pose_service_node = pick_and_place_pkg.multi_aruco_pose_service_node:main',
        ],
    },
)
