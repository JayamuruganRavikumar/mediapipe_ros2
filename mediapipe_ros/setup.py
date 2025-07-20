from setuptools import find_packages, setup

package_name = 'mediapipe_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + "/model", ['model/pose_landmarker_full.task']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jay',
    maintainer_email='jayamurugan.pr@gmail.com',
    description='A ROS2 wrapper for mediapipe',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pose_estimation_node = mediapipe_ros.pose_estimation_node:main',
        ],
    },
)
