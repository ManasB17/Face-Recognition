from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.models as models
import cv2 as cv
import torch


def open_cam(cam_num):
    """
    Opens a camera device and returns a VideoCapture object.

    Parameters
    ----------
    cam_num : int
        The index of the camera device to open. If only one camera is connected,
        use 0.

    Returns
    -------
    cv2.VideoCapture
        A VideoCapture object representing the opened camera device.

    Raises
    ------
    SystemExit
        If the camera device cannot be opened.

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


def recognize_faces(unknown_embeddings, stored_embeddings):
    """
    Calculate the distances between unknown embeddings and stored embeddings.

    Args:
        unknown_embeddings (list): A list of PyTorch tensors containing the unknown embeddings.
        stored_embeddings (list): A list of PyTorch tensors containing the stored embeddings.

    Returns:
        list: A 2D list of distances between each unknown embedding and each stored embedding.
    """

    dists = [
        [(e1 - e2).norm().item() for e2 in stored_embeddings]
        for e1 in unknown_embeddings
    ]
    return dists


def draw_bounding_boxes(box, frame, name, dist):
    """
    Draw bounding boxes around detected faces and display name and distance from stored embeddings.

    Args:
        box (list): List of bounding box coordinates (x_min, y_min, x_max, y_max) for detected faces.
        frame (numpy.ndarray): Input image frame.
        name (str): Name of the recognized person.
        dist (float): Distance between unknown face embedding and stored embeddings.

    Returns:
        numpy.ndarray: Image frame with bounding boxes, names and distances drawn.
    """
    if box is not None:
        if dist < 2.2:
            frame = cv.putText(
                frame,
                name,
                (int(box[0]), int(box[1])),
                cv.FONT_HERSHEY_PLAIN,
                2,
                (128, 34, 255),
                2,
            )
        else:
            frame_with_bounding_boxes = cv.putText(
                frame,
                "Unknown",
                (int(box[0]), int(box[1])),
                cv.FONT_HERSHEY_PLAIN,
                2,
                (128, 34, 255),
                2,
            )
        frame_with_bounding_boxes = cv.rectangle(
            frame,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            (255, 0, 0),
            2,
        )
    return frame_with_bounding_boxes


def facial_feature(model, img_cropped_list):
    """
    Computes the facial feature embeddings of the input cropped face images using the given PyTorch model.

    Args:
        model (nn.Module): PyTorch model used to compute facial feature embeddings.
        img_cropped_list (torch.Tensor): Tensor of cropped face images, of shape (batch_size, channels, height, width).

    Returns:
        torch.Tensor: Tensor of facial feature embeddings, of shape (batch_size, embedding_size).
    """

    embeddings = model(img_cropped_list.unsqueeze(0)).detach()
    return embeddings


def detect_face(frame, model):
    """
    Detects a face in a frame using a given face detection model.

    Args:
        frame: A numpy array representing the frame to detect a face in.
        model: A face detection model.

    Returns:
        A tuple containing:
        - rgb_frame: The frame as a PIL Image object with RGB colorspace.
        - bounding_boxes: A list of tuples representing the bounding boxes of the detected faces.
        - img_cropped_list: A list of PIL Image objects representing the cropped face images.
        - prob_list: A list of probabilities that the detected bounding box contains a face.
    """
    rgb_frame = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    img_cropped_list, prob_list = model(rgb_frame, return_prob=True)

    if img_cropped_list is not None:
        bounding_boxes, _ = model.detect(rgb_frame)

    return rgb_frame, bounding_boxes, img_cropped_list, prob_list


def main():
    # Check if the 'cuda' is available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print("Using device:", device)

    # loading data.pt file
    load_data = torch.load("data/data.pt")
    stored_embeddings = load_data[0]
    name_list = load_data[1]

    # Load face detection model 'MTCNN' on device
    mtcnn = MTCNN(keep_all=True, device=device)

    # Load face feature extraction model'InceptionResnet' on device
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    # Set web cam as a cam device
    cam_num = 0
    cap = open_cam(cam_num)

    while True:
        frame = capture_frame(cap)
        rgb_frame, boxes, img_cropped_list, prob_list = detect_face(frame, mtcnn)
        for i, prob in enumerate(prob_list):
            if prob > 0.90:
                dist_list = []
                unknown_embeddings = facial_feature(resnet, img_cropped_list[i])
                dist = recognize_faces(unknown_embeddings, stored_embeddings)
                dist_list.append(dist[0])
                min_dist = min(dist_list[0])  # get minumum dist value
                min_dist_idx = dist_list[0].index(min_dist)  # get minumum dist index
                name = name_list[min_dist_idx]  # get name corrosponding to minimum dist
                frame = draw_bounding_boxes(boxes[i], frame, name, min_dist)
        cv.imshow("IMG", frame)
        if (cv.waitKey(1) & 0xFF == ord("q")) or (cv.waitKey(1) & 0xFF == ord("z")):
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
