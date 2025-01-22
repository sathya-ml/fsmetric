import cv2
import numpy as np
from mtcnn import MTCNN
import mediapipe as mp
import pickle

# TODO: fix exception which requires to replace "keras" with "tensorflow.keras" in factory.py of the mtcnn lib


def use_dlib(frame, fc):
    """Detects a face in a frame using the dlib method.

    This function converts the input frame to grayscale and uses the provided
    face classifier to detect faces. It assumes that only one face is present
    and returns the first detected face.

    Args:
        frame (numpy.ndarray): The input image frame in BGR format.
        fc (cv2.CascadeClassifier): The face classifier for detecting faces.

    Returns:
        tuple: A tuple representing the bounding box of the detected face (x, y, w, h).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = fc.detectMultiScale(gray, 1.1, 4)
    face = faces[0]  # Assumes only one face is present

    return face


def use_mtcnn(frame, detector):
    """Detects a face in a frame using the MTCNN method.

    This function converts the input frame to RGB format and uses the MTCNN
    detector to find faces. It returns the bounding box of the first detected face.

    Args:
        frame (numpy.ndarray): The input image frame in BGR format.
        detector (MTCNN): The MTCNN face detector.

    Returns:
        list: A list representing the bounding box of the detected face [x, y, width, height].
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = detector.detect_faces(frame)[0]['box']

    return face


def use_ssd(frame, fc):
    """Detects a face in a frame using the SSD method with MediaPipe.

    This function processes the input frame with MediaPipe's face detection
    model to find faces. It returns the bounding box of the first detected face.

    Args:
        frame (numpy.ndarray): The input image frame in BGR format.
        fc (mediapipe.solutions.face_detection): The MediaPipe face detection module.

    Returns:
        mediapipe.framework.formats.detection_pb2.Detection: The first detected face.
    """
    # TODO: Fix the warning: Can't find file: mediapipe/modules/face_detection/face_detection_front.tflite
    with fc.FaceDetection(min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    return results[0]


def crop_frame(frame, face):
    """Crops a frame to the region defined by the face bounding box.

    This function extracts the region of the input frame that corresponds to
    the provided face bounding box.

    Args:
        frame (numpy.ndarray): The input image frame.
        face (tuple): A tuple representing the bounding box of the face (x, y, width, height).

    Returns:
        numpy.ndarray: The cropped image frame containing the face.
    """
    x, y, w, h = face

    return frame[y:y + h, x:x + w]


def face_dimension(video_path, method="mtcnn", cascade_classifier_config_path=None):
    '''
    Extract face locations in the video every 10 frames
    '''
    face = None
    faces = list()
    cap = cv2.VideoCapture(video_path)

    if method == "dlib":
        fc = cv2.CascadeClassifier(cascade_classifier_config_path)
    elif method == "mtcnn":
        fc = MTCNN()
    elif method == "ssd":
        fc = mp.solutions.face_detection

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if i % 10 == 0:
            if method=="dlib":
                face = use_dlib(frame, fc)
            elif method == "mtcnn":
                # FIXME: Why is there an IndexError?
                try:
                    face = use_mtcnn(frame, fc)
                except IndexError:
                    continue
            elif method == "ssd":
                face = use_ssd(frame, fc)
            faces.append(face)
        i += 1
    return faces


def load_video(path, faces, max_frames=0, resize=(224, 224), normalize=True):
    """Loads and processes video frames by cropping based on face locations.

    This function reads frames from a video file, crops them using provided face
    locations, resizes them to the specified dimensions, and optionally normalizes
    the pixel values.

    Args:
        path (str): Path to the video file.
        faces (list): List of face bounding boxes for cropping frames.
        max_frames (int, optional): Maximum number of frames to load. Defaults to 0, which means all frames.
        resize (tuple, optional): Dimensions to resize frames to. Defaults to (224, 224).
        normalize (bool, optional): Whether to normalize pixel values to the range [0, 1]. Defaults to True.

    Returns:
        numpy.ndarray: Array of processed video frames.
    """
    cap = cv2.VideoCapture(path)
    frames = list()
    count = 0
    i = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if i % 10 == 0:
                # Ensure the face index does not exceed the number of detected faces
                if count >= len(faces):
                    break

                face = faces[count]
                count += 1

            cropped_frame = crop_frame(frame, face)

            # If the cropped frame is empty, use the original frame
            if cropped_frame.shape[0] == 0:
                cropped_frame = frame

            # Convert the frame from BGR to RGB
            frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

            # Resize the frame to the specified dimensions
            frame = cv2.resize(frame, resize)
            frames.append(frame)

            i += 1

            # Stop if the maximum number of frames is reached
            if len(frames) == max_frames:
                break
    finally:
        cap.release()

    # Normalize the frames if required
    if normalize:
        return np.array(frames) / 255.0
    else:
        return np.array(frames)

def save_face_locations(video_path_source, face_path):
    """Saves face locations from a video to a file.

    This function detects face locations in the given video and saves them to a specified file.
    If no file path is provided, it generates a default file path based on the video path.

    Args:
        video_path_source (str): Path to the source video file.
        face_path (str): Path to the file where face locations will be saved. If None, a default path is generated.

    Returns:
        list: A list of face locations detected in the video.
    """
    if face_path is None:
        face_path = video_path_source[:-4] + "_faces.txt"
    
    faces = face_dimension(video_path_source)
    
    with open(face_path, "wb") as fp:
        pickle.dump(faces, fp)
    
    return faces


def load_face_locations(video_path_source, face_path):
    """Loads face locations from a file or computes and saves them if not available.

    This function attempts to load face locations from a specified file. If the file does not exist,
    it computes the face locations from the video, saves them to the file, and then returns them.

    Args:
        video_path_source (str): Path to the source video file.
        face_path (str): Path to the file containing face locations. If None, a default path is generated.

    Returns:
        list: A list of face locations detected in the video.
    """
    if face_path is None:
        face_path = video_path_source[:-4] + "_faces.txt"
    
    try:
        with open(face_path, "rb") as fp:
            faces = pickle.load(fp)
    except FileNotFoundError:
        faces = save_face_locations(video_path_source, face_path)

    return faces


def load_cropped_videos(video_path_source, video_path_dest, face_path, normalize=False, resize=(224, 224)):
    """Loads and preprocesses two videos by cropping faces and resizing frames.

    This function loads two videos, crops the frames based on detected face locations,
    resizes the frames to the specified dimensions, and normalizes the pixel values if required.
    The face locations are either loaded from a file or computed and saved if not already available.

    Args:
        video_path_source (str): Path to the source video file.
        video_path_dest (str): Path to the destination video file.
        face_path (str): Path to the file containing face locations.
        normalize (bool, optional): Whether to normalize the pixel values to the range [0, 1]. Defaults to False.
        resize (tuple, optional): The dimensions to resize the frames to. Defaults to (224, 224).

    Returns:
        list of numpy.ndarray: A list containing two numpy arrays, each representing the preprocessed frames
        of the source and destination videos. The number of frames in both videos will be the same.
    """
    # Load or create and save face positions
    faces = load_face_locations(video_path_source, face_path)
    
    # Load and preprocess the videos
    videos = [
        load_video(video_path_source, faces, normalize=normalize, resize=resize),
        load_video(video_path_dest, faces, normalize=normalize, resize=resize)
    ]

    # Ensure both videos have the same number of frames
    if videos[0].shape[0] > videos[1].shape[0]:
        videos[0] = videos[0][:videos[1].shape[0]]

    return videos
