import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'collision_avoidance_pkg'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # We also include the launch files and the world files in the package data
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'worlds'), glob(os.path.join('worlds', '*.world'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Team_19',
    maintainer_email='s349452@studenti.polito.it',
    description='Project 5 - Deep Reinforcement Learning Collision Avoidance',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # command name = packageName.module:function
            'random_agent = collision_avoidance_pkg.random_agent:main',
        ],
    },
)