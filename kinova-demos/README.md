# Kinova-Gen3-Face-Follow
![kinova mp4](https://user-images.githubusercontent.com/22056265/167654474-709d0f7e-6b22-4d16-bdd7-5ef0a0927964.gif)

This is the face following demo using the Kinova Gen3 robot arm.

## Requirements

Create a virtual environment and install the requirements:

```bash
conda create -n kinova python=3.9
conda activate kinova
pip install -e .
```

## Usage

### Show Camera with YOLO Face Detector

`python src/camera.py`

### Follow Faces

`python src/follow.py`


