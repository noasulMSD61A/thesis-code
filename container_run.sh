xhost +local:docker
docker run -p 8888:8888 --rm -it --gpus all -e DISPLAY=$DISPLAY -v $(pwd):/data thesis-container bash