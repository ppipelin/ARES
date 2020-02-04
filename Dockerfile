FROM python:3.7

RUN apt-get update && \
	apt-get install -y build-essential cmake git wget unzip yasm pkg-config libswscale-dev libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libavformat-dev libpq-dev python3-pip libsm6 libxext6 libxrender-dev libopenexr-dev zlib1g-dev g++ python-opengl libosmesa6 libgtk2.0-dev && \
	rm -rf /var/lib/apt/lists/*

ADD requirements.txt /

RUN pip3 install -r requirements.txt

ENV PYOPENGL_PLATFORM=osmesax

ENV OPENCV_VERSION="4.2.0"

RUN wget https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip \
&& unzip -q ${OPENCV_VERSION}.zip \
&& rm ${OPENCV_VERSION}.zip

RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
&& unzip -q ${OPENCV_VERSION}.zip \
&& mkdir /opencv-${OPENCV_VERSION}/cmake_binary \
&& cd /opencv-${OPENCV_VERSION}/cmake_binary \
&& cmake -DBUILD_TIFF=ON \
  -DBUILD_opencv_java=OFF \
  -DOPENCV_EXTRA_MODULES_PATH=/opencv_contrib-${OPENCV_VERSION}/modules \
  -DOPENCV_ENABLE_NONFREE=ON \
  -DWITH_CUDA=OFF \
  -DWITH_OPENGL=ON \
  -DWITH_OPENCL=ON \
  -DWITH_IPP=ON \
  -DWITH_TBB=ON \
  -DWITH_EIGEN=ON \
  -DWITH_V4L=ON \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DCMAKE_INSTALL_PREFIX=$(python3.7 -c "import sys; print(sys.prefix)") \
  -DPYTHON_EXECUTABLE=$(which python3.7) \
  -DPYTHON_INCLUDE_DIR=$(python3.7 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
  -DPYTHON_PACKAGES_PATH=$(python3.7 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
  .. \
&& make -j3 install \
&& rm /${OPENCV_VERSION}.zip \
&& rm -r /opencv-${OPENCV_VERSION}

RUN ln -s \
  /usr/local/python/cv2/python-3.7/cv2.cpython-37m-x86_64-linux-gnu.so \
  /usr/local/lib/python3.7/site-packages/cv2.so