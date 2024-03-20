import cv2 as cv
import numpy as np
import torch
import time
import os
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw
import torchvision.transforms as T
from calc_thresh import Threshold


def get_features(img_path, resnet):
    """
    Extracts the features of an image using the provided ResNet model.

    Parameters
    ----------
    img_path : str
        Path to the input image file.
    resnet : torch.nn.Module
        Pre-trained ResNet model to extract features.

    Returns
    -------
    numpy.ndarray
        1D array of feature values extracted from the image.

    """
    img = Image.open(img_path)
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    with torch.no_grad():
        features = resnet(img_tensor)
    return features.squeeze().numpy()


def get_mean_embeddings(data_dir, resnet):
    """
    Computes the mean embeddings of each person in the given directory using the provided ResNet model.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing subdirectories with images for each person.
    resnet : torch.nn.Module
        Pre-trained ResNet model to extract features.

    Returns
    -------
    Tuple[torch.Tensor, List[str]]
        A tuple containing the mean embeddings and their corresponding labels.
        - The mean embeddings are a PyTorch tensor of shape (num_persons, feature_dim), where num_persons
          is the number of people in the directory and feature_dim is the dimensionality of the extracted
          features.
        - The labels are a list of strings containing the name of each person in the directory.

    """

    embeddings = []
    labels = []
    # Loop over all folders containing images and calculate mean embeddings for each person

    for person_dir in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, person_dir)):
            # Get the name of the person from the folder name
            person_name = person_dir

            # Get the list of image files for this person
            image_files = [
                os.path.join(data_dir, person_dir, f)
                for f in os.listdir(os.path.join(data_dir, person_dir))
            ]

            # Get the feature vectors for each image of the person
            feature_vectors = [
                get_features(img_path, resnet) for img_path in image_files
            ]

            # Calculate the mean feature vector
            num_images = len(feature_vectors)
            sum_vector = feature_vectors[0]
            for i in range(1, num_images):
                sum_vector += feature_vectors[i]
            mean_vector = sum_vector / num_images

            # Add the mean vector and label to the list
            embeddings.append(mean_vector)
            labels.append(person_name)

    # Save the embeddings and labels as a PyTorch tensor
    embedding_list = torch.stack([torch.from_numpy(e) for e in embeddings])
    name_list = labels
    data = [embedding_list, name_list]
    return data


# def val_data(resnet, mtcnn, path):
#     dataset = datasets.ImageFolder(path)
#     # accessing names of peoples from folder names
#     idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}

#     def collate_fn(x):
#         return x[0]

#     loader = DataLoader(dataset, collate_fn=collate_fn)

#     name_list = []
#     embedding_list = []
#     for img, idx in loader:
#         face, prob = mtcnn(img, return_prob=True)
#         if face is not None and prob > 0.92:
#             emb = resnet(face)
#             embedding_list.append(emb.detach())
#             name_list.append(idx_to_class[idx])

#     data = [embedding_list, name_list]
#     return data


def save_faces(path, faces):
    """
    Saves the given list of face images to disk in the specified directory.

    Parameters
    ----------
    path : str
        Path to the directory where the face images will be saved.
    faces : List[Image]
        A list of PIL Image objects representing face images.

    Returns
    -------
    None
        This function does not return anything, it only saves the face images to disk.

    """
    name = input("Please enter your name:")

    # create directory if not exists
    if not os.path.exists(f"{path}/{name}/"):
        os.makedirs(f"{path}/{name}/")
    for face in faces:
        face.save(f"{path}/{name}/{name}-{int(time.time())}.jpg")
        time.sleep(1)


def crop_face(rgb_frame, bounding_box):
    """
    Crops a face region from a given RGB image frame based on the specified bounding box.

    Parameters
    ----------
    rgb_frame : PIL.Image
        An RGB image frame represented as a PIL Image object.
    bounding_box : tuple
        A tuple containing the (left, top, right, bottom) coordinates of the face bounding box.

    Returns
    -------
    PIL.Image
        A PIL Image object representing the cropped face region.

    """
    offset = 5
    faces = rgb_frame.crop(
        (
            int(bounding_box[0] + offset),
            int(bounding_box[1] + offset),
            int(bounding_box[2] - offset),
            int(bounding_box[3] - offset),
        )
    )
    return faces


def draw_bounding_boxes(bounding_boxes, rgb_frame):
    """
    Draws bounding boxes on the given RGB image frame based on the specified coordinates.

    Parameters
    ----------
    bounding_boxes : numpy.ndarray
        An array of shape (N, 4) containing the (left, top, right, bottom) coordinates of N face bounding boxes.
    rgb_frame : PIL.Image
        An RGB image frame represented as a PIL Image object.

    Returns
    -------
    numpy.ndarray
        A numpy array representing the frame with the drawn bounding boxes in BGR color format.

    """
    if bounding_boxes is not None:
        draw = ImageDraw.Draw(rgb_frame)
        for box in bounding_boxes:
            draw.rectangle(box.tolist(), outline="blue", width=3)
        frame_with_bounding_boxes = cv.cvtColor(np.array(rgb_frame), cv.COLOR_RGB2BGR)
    else:
        frame_with_bounding_boxes = cv.cvtColor(np.array(rgb_frame), cv.COLOR_RGB2BGR)
    return frame_with_bounding_boxes


def detect_faces(frame, model):
    """
    Detects faces in the given image frame using a deep neural network face detection model.

    Parameters
    ----------
    frame : numpy.ndarray
        A numpy array representing the input image frame in BGR color format.
    model : cv2.dnn_Net
        A deep neural network model pre-trained for face detection.

    Returns
    -------
    Tuple[PIL.Image, numpy.ndarray]
        A tuple containing the input RGB image frame represented as a PIL Image object and an array of shape
        (N, 4) containing the (left, top, right, bottom) coordinates of N face bounding boxes.

    """
    rgb_frame = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    # Detect faces
    bounding_boxes, _ = model.detect(rgb_frame)
    return rgb_frame, bounding_boxes


def open_cam(cam_num):
    """
    Opens a video capture device with the given camera number.

    Parameters
    ----------
    cam_num : int
        The index of the camera to open.

    Returns
    -------
    cv2.VideoCapture
        A VideoCapture object representing the opened camera device.

    Raises
    ------
    SystemExit
        If the camera cannot be opened.

    """
    cap = cv.VideoCapture(cam_num)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    else:
        return cap


def capture_frame(cap):
    """
    Captures a frame from the given video capture device.

    Parameters
    ----------
    cap : cv2.VideoCapture
        A VideoCapture object representing an opened camera device.

    Returns
    -------
    numpy.ndarray
        A NumPy array representing the captured frame.

    Raises
    ------
    SystemExit
        If a frame cannot be captured.

    """
    ret, frame = cap.read()

    # If frame is read correctly ret is True
    if not ret:
        print("Can't receive frame. Exiting ...")
        exit()
    return frame


def main():
    # Check if the 'cuda' is available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print("Using device:", device)

    # Load face detection model 'MTCNN' on device
    mtcnn = MTCNN(keep_all=True, device=device)

    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    # Set web cam as a cam device
    cam_num = 0
    cap = open_cam(cam_num)

    faces = []
    path_to_save_faces = "data/test_images"

    while True:
        frame = capture_frame(cap)
        rgb_frame, bounding_boxes = detect_faces(frame, mtcnn)
        frame_with_bounding_boxes = draw_bounding_boxes(bounding_boxes, rgb_frame)
        if cv.waitKey(1) == ord("c"):
            for i in range(10):
                frame = capture_frame(cap)
                rgb_frame, bounding_boxes = detect_faces(frame, mtcnn)
                frame_with_bounding_boxes = draw_bounding_boxes(
                    bounding_boxes, rgb_frame
                )
                face = crop_face(rgb_frame, bounding_boxes[0])
                faces.append(face)
                time.sleep(1)
                text = f"Capturing image {i}"
                frame_with_bounding_boxes = cv.putText(
                    frame_with_bounding_boxes,
                    text,
                    (10, 10),
                    cv.FONT_HERSHEY_PLAIN,
                    2,
                    (128, 34, 255),
                    2,
                )
            # cv.imshow("abc", frame_with_bounding_boxes)
            save_faces(path_to_save_faces, faces)
            print("capture complete")
        cv.imshow("Frame", frame_with_bounding_boxes)
        

        if (cv.waitKey(1) & 0xFF == ord("q")) or (cv.waitKey(1) & 0xFF == ord("z")):
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

    data = get_mean_embeddings(path_to_save_faces, resnet)
    # val_data = val_data_feature(path_of_validation_data, resnet)

    threshold = Threshold(data, data)  
    thresh = threshold()
    print(thresh)
    torch.save(data, "data/data.pt")


if __name__ == "__main__":
    main()
