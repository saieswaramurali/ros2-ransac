from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ransac_perception'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Include example launch files
        (os.path.join('share', package_name, 'examples'), glob('examples/*.py')),
        # Include config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),
        # Include msg files for reference
        (os.path.join('share', package_name, 'msg'), glob('msg/*.msg')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='SAI ESWARA M',
    maintainer_email='saimurali2005@gmail.com',
    description='ROS2 RANSAC-based geometric primitive detection for perception pipelines',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ransac_node = ransac_perception.ransac_node:main',
            'test_publisher = ransac_perception.test_publisher:main',
        ],
    },
)
