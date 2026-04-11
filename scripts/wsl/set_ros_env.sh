#!/usr/bin/env bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_LOCALHOST_ONLY=0

echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}"
echo "ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY}"
echo "Use the same values on Windows so DDS discovery can cross the Windows/WSL boundary."
