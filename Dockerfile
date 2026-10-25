# tograsp-socp: task-dependent grasp metric (SOCP) in a ROS 2 container.
#
# The code is CPU-only (numpy, scipy, cvxpy, open3d), so this image needs
# neither an NVIDIA L4T base image nor the NVIDIA container runtime. A plain
# ROS 2 image works on the Orin as long as it is built for linux/arm64.
#
# Build natively (on the Orin, or on an Apple Silicon Mac; both are linux/arm64):
#   docker build -t tograsp-socp .
#
# Cross-build for the Orin from an x86 machine (works, but slow under emulation):
#   docker buildx build --platform linux/arm64 -t tograsp-socp:arm64 --load .
#
# Match the perception container's ROS 2 distro (humble = Ubuntu 22.04 / Python 3.10,
# jazzy = Ubuntu 24.04 / Python 3.12; both are in this repo's supported range):
#   docker build --build-arg ROS_DISTRO=jazzy -t tograsp-socp .

ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-base

# open3d's Linux wheels link against these system libraries. The import fails
# without them, even though this code never opens a window.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        libgl1 \
        libegl1 \
        libx11-6 \
        libgfortran5 \
        libgomp1 \
        libidn2-0 \
    && rm -rf /var/lib/apt/lists/*

# MPLBACKEND=Agg: there is no display in the container.
# PIP_BREAK_SYSTEM_PACKAGES=1: needed for pip on Ubuntu 24.04 (jazzy); ignored by
# the older pip on Ubuntu 22.04 (humble).
ENV MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

WORKDIR /app

# Install dependencies before copying the code, so that editing the code does
# not reinstall everything on each rebuild.
COPY requirements.txt .

# numpy<2: ROS 2's own Python packages are installed from apt and built against
# numpy 1.x. Upgrading the image's numpy to 2.x risks breaking compiled ROS
# Python modules. Your code runs unchanged on numpy 1.26.
RUN pip3 install "numpy<2" -r requirements.txt

COPY . .

# Default: run the bundled example. The ROS entrypoint has already sourced
# /opt/ros/$ROS_DISTRO/setup.bash, so ros2 commands also work here.
CMD ["python3", "main_pickup.py", "--filename", "nontextured.ply", "--trial", "1", "--log-dir", "/app/logs"]
