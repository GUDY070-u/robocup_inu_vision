import os
from glob import glob
from setuptools import setup

package_name = 'launch_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='FCSL_mani_team',
    maintainer_email='user@todo.todo',
    description='',
    license='TODO',
    entry_points={
        'console_scripts': [],
    },
)
