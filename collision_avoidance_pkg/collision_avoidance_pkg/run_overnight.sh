#!/bin/bash
source install/setup.bash

echo "Starting Benchmark 1: Speed = 0.15 m/s"
ros2 run collision_avoidance_pkg train_ddqn --speed 0.15

sleep 10

echo "Starting Benchmark 2: Speed = 0.25 m/s"
ros2 run collision_avoidance_pkg train_ddqn --speed 0.25