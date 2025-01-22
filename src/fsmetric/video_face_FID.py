"""
Loads pretrained model of I3d Inception architecture from the paper: 'https://arxiv.org/abs/1705.07750'
Evaluates a RGB similar to the paper's github repo: 'https://github.com/deepmind/kinetics-i3d'
"""

import argparse
import os

import numpy as np
from tensorflow.keras.models import Model
from scipy.linalg import sqrtm

from fsmetric.i3d_inception import Inception_Inflated3d
from fsmetric.video_utils import load_cropped_videos, save_face_locations

# Number of classes in the I3D Inception model
NUM_CLASSES = 400

# Name of the pre-trained model to be used
MODEL_NAME = "rgb_imagenet_and_kinetics"

# Name of the layer from which features will be extracted
MODEL_LAYER = "global_avg_pool"

# Filename of the pre-trained weights for the I3D Inception model
WEIGHTS_FILENAME = "rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels.h5"

# Filename of the configuration file for the Haar Cascade Classifier used for face detection
CASCADE_CLASSIFIER_CONFIG_FILENAME = "haarcascade_frontalface_default.xml"


def crop_center_square(frame):
    """Crops the central square region from a given frame.

    This function takes a frame and extracts the largest possible square
    from the center of the frame.

    Args:
        frame (numpy.ndarray): The input frame from which the central square
            region is to be cropped.

    Returns:
        numpy.ndarray: The cropped central square region of the frame.
    """
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)

    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]


def calculate_fid(features):
    """Calculates the Frechet Inception Distance (FID) between two sets of features.

    The FID is a metric used to evaluate the similarity between two datasets of
    features, often used in the context of comparing generated images to real images.
    The FID is calculated using the formula:

    FID d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))

    Args:
        features (list of numpy.ndarray): A list containing two numpy arrays, each
            representing a set of features. The first array corresponds to the real
            dataset, and the second array corresponds to the generated dataset.

    Returns:
        float: The calculated FID score, which quantifies the difference between
        the two sets of features. A lower score indicates higher similarity.
    """
    # Calculate mean and covariance statistics for both feature sets
    mu1, sigma1 = features[0].mean(axis=0), np.cov(features[0], rowvar=False)
    mu2, sigma2 = features[1].mean(axis=0), np.cov(features[1], rowvar=False)

    # Calculate the sum of squared differences between the means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    # Calculate the square root of the product of the covariance matrices
    covmean = sqrtm(sigma1.dot(sigma2))

    # Check and correct for any imaginary numbers resulting from the square root
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Calculate the FID score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

    return fid


def rgb_layer_predict(video_path_source, video_path_dest, face_path, cascade_classifier_config_path, weights_path):
    """Predicts features from RGB video layers and calculates the Frechet Inception Distance (FID).

    This function processes two videos, extracts features using a pre-trained Inception Inflated 3D model,
    and calculates the FID score to quantify the similarity between the feature sets of the two videos.

    Args:
        video_path_source (str): Path to the source video file.
        video_path_dest (str): Path to the destination video file.
        face_path (str): Path to the file containing face locations.
        cascade_classifier_config_path (str): Path to the cascade classifier configuration file.
        weights_path (str): Path to the pre-trained model weights.

    Returns:
        float: The calculated FID score, indicating the similarity between the two video datasets.
    """
    # Load and preprocess the videos, normalizing and resizing them to 224x224
    rgb_video = load_cropped_videos(video_path_source, video_path_dest, face_path, normalize=True, resize=(224, 224))

    # List to store feature vectors for each video
    feature_vectors_list = list()

    for video in rgb_video:
        # Initialize the Inception Inflated 3D model with pre-trained weights
        rgb_model = Inception_Inflated3d(
            include_top=True,
            weights=MODEL_NAME,
            weights_path=weights_path,
            input_shape=(video.shape[0], video.shape[1], video.shape[2], video.shape[3]),
            classes=NUM_CLASSES
        )

        # Extract features from the specified layer of the model
        layer_output = rgb_model.get_layer(MODEL_LAYER).output
        layer_model = Model(inputs=rgb_model.input, outputs=layer_output)
        features = layer_model.predict(np.expand_dims(video, axis=0))

        # Remove the singleton dimension from the features
        feature_vectors_list.append(np.squeeze(features))

    # Calculate and return the FID score based on the extracted features
    return calculate_fid(feature_vectors_list)


if __name__ == "__main__":
    """Main entry point for calculating the Frechet Inception Distance (FID) between two videos.

    This script processes two video files to compute the FID score using the I3d Inception architecture.
    The FID score is a measure of similarity between the feature sets of the two videos.

    Command Line Arguments:
        --video_A: str, required
            Path to the first video file (video A).
        --video_B: str, required
            Path to the second video file (video B).
        --faces: str, required
            Path to the file containing face locations.
        --model_dir: str, required
            Path to the directory containing the model files.

    Raises:
        FileNotFoundError: If the specified face locations path does not exist and cannot be created.
    """
    arg_parser = argparse.ArgumentParser(
        description=(
            "Calculates the Frechet Inception Distance between video A and video B "
            "using the I3d Inception architecture described in: 'https://arxiv.org/abs/1705.07750'. "
            "The result is printed to screen."
        )
    )
    arg_parser.add_argument("--video_A", help="Path to video A", required=True)
    arg_parser.add_argument("--video_B", help="Path to video B", required=True)
    arg_parser.add_argument("--faces", help="Path to face locations list", required=True)
    arg_parser.add_argument("--model_dir", help="Path to the directory containing the model files", required=True)

    args = arg_parser.parse_args()

    if not os.path.exists(args.faces):
        os.makedirs('/'.join(args.faces.split('/')[:-1]))

    # Check if face locations have already been computed
    if args.faces is None:
        print("Computing face locations...")
        save_face_locations(args.video_A)

    CASCADE_CLASSIFIER_CONFIG_PATH = os.path.join(args.model_dir, CASCADE_CLASSIFIER_CONFIG_FILENAME)
    WEIGHTS_PATH = os.path.join(args.model_dir, WEIGHTS_FILENAME)

    fid = rgb_layer_predict(
        args.video_A, args.video_B, args.faces, CASCADE_CLASSIFIER_CONFIG_PATH, WEIGHTS_PATH
    )

    print(fid)
