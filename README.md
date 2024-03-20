## Introduction

This is a set of utility functions for facial recognition using the facenet_pytorch library. The following functions are provided:


- `open_cam(cam_num)`: Opens a camera device and returns a VideoCapture object.
- capture_frame(cap): Captures a frame from the given video capture device.
- `recognize_faces(unknown_embeddings, stored_embeddings)`: Calculates the distances between unknown embeddings and stored embeddings.
- `draw_bounding_boxes(box, frame, name, dist)`: Draws bounding boxes around detected faces and displays name and distance from stored embeddings.
- `facial_feature(model, img_cropped_list)`: Computes the facial feature embeddings of the input cropped face images using the given PyTorch model.
- `detect_face(frame, model)`: Detects a face in a frame using a given face detection model.
- `main()`: Runs the main facial recognition program.

## Dependencies
The following Python packages are required to run the facial recognition utility:

- facenet_pytorch
- Pillow
- torchvision
- torch

**The following third-party libraries are required:**

- cv2 (OpenCV)


## Installation

To use this project first install the required packages as listed:

--Install all required packages from requirements.txt file  :

```bash
  pip install -r requirements.txt

```

## Run Locally

Clone the project

```bash
  git clone 
```

Go to the project directory

```bash
  cd Computer-Vision
```

Install dependencies

```bash
  pip install requirements.txt
```

Open terminal

```bash
  python infer_recognize_face.py 
```
