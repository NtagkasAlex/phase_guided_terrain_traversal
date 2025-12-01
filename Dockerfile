FROM ros:humble

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/opt/ros/humble/lib/python3/dist-packages

SHELL ["/bin/bash", "-lc"]


RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-rviz2 \
    python3-colcon-common-extensions \
    build-essential \
    git \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglu1-mesa \
    libx11-6 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libxcb1 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ros-humble-pcl-ros \
      ros-humble-tf-transformations\
      libpcl-dev \
      ros-humble-cv-bridge \
      libopencv-dev \
      python3-opencv \
      ros-humble-ament-cmake \
      ros-humble-ament-cmake-core \
      ros-humble-ament-package \
      python3-ament-package \
      python3-rosdep \
      python3-vcstool \
      liboctomap-dev \
      ros-humble-octomap \
      ros-humble-octomap-msgs \
      ros-humble-nav2-costmap-2d \
      ros-humble-filters \
      ros-humble-grid-map \
    && rm -rf /var/lib/apt/lists/*



    

#ENV AMENT_TRACE_SETUP_FILES=0
#ENV PYTHONPATH=/opt/ros/humble/lib/python3/dist-packages:${PYTHONPATH:-}
RUN source /opt/ros/humble/setup.bash


ENV DEBIAN_FRONTEND=noninteractive
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV MESA_LOADER_DRIVER_OVERRIDE=llvmpipe

RUN apt-get update && apt-get install -y --no-install-recommends \
    mesa-utils \
    libgl1-mesa-dri \
    libgl1-mesa-glx \
    libglx-mesa0 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*
    
WORKDIR /root/
COPY . /root/
#ENTRYPOINT ["/bin/bash", "-c", "cd /root/ros_ws && rm -rf build install log || true && #colcon build && exec bash"]

CMD ["bash"]

