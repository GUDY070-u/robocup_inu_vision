from setuptools import find_packages, setup

package_name = 'serial_pkg'

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
    maintainer_email='jeho4557@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'serial_flag_bridge = serial_pkg.serial_flag_bridge:main',
            'workcell_coordinator = serial_pkg.workcell_coordinator:main',

        ],
    },
)
