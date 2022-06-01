FROM tensorflow/tensorflow:nightly-jupyter

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London
RUN apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/3bf863cc.pub
RUN apt-get install ffmpeg libsm6 libxext6 git -y python3-tk
RUN pip install --upgrade pip
RUN pip install opencv-python \
    numpy \
    matplotlib \
    pandas \
    imageio \
    git+https://github.com/tensorflow/docs \
    tensorflow-hub \
    scikit-learn \
    pandas \
    PySimpleGUI 
    
    

EXPOSE 8888

WORKDIR /data