import argparse
import os

import cv2
import numpy
import skimage.metrics

from fsmetric import fsmlib
from fsmetric.fsmlib import get_random_frames_source_destination

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

CASCADE_CLASSIFIER_CONFIG_FILENAME = "haarcascade_frontalface_default.xml"

# Required to have the same dimensions after extracting the face
WIDTH = 250
HEIGHT = 250

# Path for the video used in pixelation
VIDEO_PATH = "example/ConvertA.mkv"

# Number of pixels for pixelation
PIXELATION_WIDTH = 32
PIXELATION_HEIGHT = 32

# Size of the kernel for blurring
BLURRING_KERNEL = 16

NUM_FRAMES = 25


def get_pixelation(frame_list):
    """Applies pixelation effect to a list of frames.

    This function resizes each frame to a smaller size and then enlarges it back
    to the original size to create a pixelation effect.

    Args:
        frame_list (list): A list of frames (images) to be pixelated.

    Returns:
        list: A list of pixelated frames.
    """
    pixelated_list = []

    for frame in frame_list:
        height, width = frame.shape[:2]

        # Reduce the image size
        temp = cv2.resize(frame, (PIXELATION_WIDTH, PIXELATION_HEIGHT), interpolation=cv2.INTER_LINEAR)

        # Enlarge the image to create the pixelation effect
        output = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

        pixelated_list.append(output)

    return pixelated_list


def get_blurring(frame_list):
    """Applies blurring effect to a list of frames.

    This function uses a blurring kernel to blur each frame in the list.

    Args:
        frame_list (list): A list of frames (images) to be blurred.

    Returns:
        list: A list of blurred frames.
    """
    return [cv2.blur(frame, (BLURRING_KERNEL, BLURRING_KERNEL), cv2.BORDER_DEFAULT) for frame in frame_list]


def frame_to_face_list(fc, frame_list):
    """Extracts and resizes faces from a list of frames.

    This function converts each frame to grayscale, detects faces, extracts them,
    and resizes them to a uniform size.

    Args:
        fc (cv2.CascadeClassifier): A face detector initialized with a cascade classifier.
        frame_list (list): A list of frames (images) from which faces are to be extracted.

    Returns:
        list: A list of extracted and resized face images.
    """
    face_list = []

    for frame in frame_list:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect the face
        face_box = fc.detectMultiScale(gray, 1.1, 4)[0]

        # Extract the face
        x, y, w, h = face_box
        face = frame[y:y + h, x:x + w]

        # Resize the image to have uniform dimensions
        face = cv2.resize(face, (WIDTH, HEIGHT))
        face_list.append(face)

    return face_list


def psnr_from_image_list(src, dst):
    """Calculates PSNR for pairs of images.

    This function computes the Peak Signal-to-Noise Ratio (PSNR) between each pair
    of source and destination images.

    Args:
        src (list): A list of source images.
        dst (list): A list of destination images.

    Returns:
        list: A list of PSNR values for each image pair.
    """
    return [skimage.metrics.peak_signal_noise_ratio(x, y) for (x, y) in zip(src, dst)]


def ssim_multichannel_from_image_list(src, dst):
    """Calculates multichannel SSIM for pairs of images.

    This function computes the Structural Similarity Index (SSIM) for each pair
    of source and destination images, considering all color channels.

    Args:
        src (list): A list of source images.
        dst (list): A list of destination images.

    Returns:
        list: A list of multichannel SSIM values for each image pair.
    """
    return [skimage.metrics.structural_similarity(x, y, multichannel=True) for (x, y) in zip(src, dst)]


def ssim_from_image_list(src, dst):
    """Calculates SSIM for grayscale pairs of images.

    This function converts each image to grayscale and computes the Structural
    Similarity Index (SSIM) between each pair of source and destination images.

    Args:
        src (list): A list of source images.
        dst (list): A list of destination images.

    Returns:
        list: A list of SSIM values for each grayscale image pair.
    """
    src_grey = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in src]
    dst_grey = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in dst]

    return [skimage.metrics.structural_similarity(x, y) for (x, y) in zip(src_grey, dst_grey)]


def ssim_tf_from_image_list(src, dst):
    """Calculates SSIM using TensorFlow for pairs of images.

    This function uses TensorFlow to compute the Structural Similarity Index (SSIM)
    and Multiscale SSIM between each pair of source and destination images.

    Args:
        src (list): A list of source images.
        dst (list): A list of destination images.

    Returns:
        tuple: Two lists containing SSIM and Multiscale SSIM values for each image pair.
    """
    ssim_tf = []
    ssim_tf_multiscale = []

    with tf.compat.v1.Session() as sess:
        src_tf = [tf.convert_to_tensor(img) for img in src]
        dst_tf = [tf.convert_to_tensor(img) for img in dst]

        for (x, y) in zip(src_tf, dst_tf):
            ssim_tf.append(sess.run(tf.image.ssim(x, y, max_val=255)))
            ssim_tf_multiscale.append(sess.run(tf.image.ssim_multiscale(x, y, max_val=255)))

    return ssim_tf, ssim_tf_multiscale



def calculate_values(vid_path_src: str, vid_path_dst: str, cascade_classifier_config_path: str, num_frames: int) -> dict:
    """Calculates PSNR and SSIM metrics between two videos.

    This function extracts random frames from two videos, detects faces within those frames,
    and calculates the Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM)
    for both grayscale and color images.

    Args:
        vid_path_src (str): Path to the source video file.
        vid_path_dst (str): Path to the destination video file.
        cascade_classifier_config_path (str): Path to the cascade classifier configuration file for face detection.
        num_frames (int): Number of random frames to extract from each video.

    Returns:
        dict: A dictionary containing the mean PSNR, SSIM, and multichannel SSIM values.
    """
    # Extract random frames from the source and destination videos
    print("Extracting random frames from the videos: " + vid_path_src + " and: " + vid_path_dst)
    src, dst = get_random_frames_source_destination(vid_path_src, vid_path_dst, num_frames)

    # Initialize the face detector with the given configuration
    fc = cv2.CascadeClassifier(cascade_classifier_config_path)

    # Extract faces from the frames
    src = frame_to_face_list(fc, src)
    dst = frame_to_face_list(fc, dst)

    # Calculate the PSNR index
    psnr = psnr_from_image_list(src, dst)

    # Calculate the SSIM index for grayscale images
    ssim = ssim_from_image_list(src, dst)

    # Calculate the SSIM index for color images
    # (implemented as the average of SSIM values independently for the three components)
    ssim_multichannel = ssim_multichannel_from_image_list(src, dst)

    results_dict: dict = {
        "PSNR_mean": float(numpy.mean(psnr)),
        "SSIM_mean": float(numpy.mean(ssim)),
        "SSIM_multichannel_mean": float(numpy.mean(ssim_multichannel)),
    }

    return results_dict


def calculate_values_blurred():
    """Calculates and prints PSNR and SSIM metrics for pixelated and blurred frames.

    This function extracts random frames from a specified video, detects faces within those frames,
    and applies pixelation and blurring effects. It then calculates and prints the Peak Signal-to-Noise
    Ratio (PSNR) and Structural Similarity Index (SSIM) for both effects, using grayscale and color images.
    Additionally, it calculates SSIM using TensorFlow for both standard and multiscale methods.
    """
    # Extract random frames from the video for testing
    print("Extracting random frames from the video: " + VIDEO_PATH)
    src = fsmlib.get_random_frames(VIDEO_PATH, NUM_FRAMES)

    # Initialize the face detector with the given configuration
    fc = cv2.CascadeClassifier(CASCADE_CLASSIFIER_CONFIG_PATH)
    src = frame_to_face_list(fc, src)

    # Apply pixelation and blurring effects
    pixelation = get_pixelation(src)
    blurring = get_blurring(src)

    # Calculate and print PSNR and SSIM values for pixelation
    print("PIXELATION")

    # Calculate the PSNR index
    psnr = psnr_from_image_list(src, pixelation)
    print("PSNR mean: " + str(numpy.mean(psnr)))

    # Calculate the SSIM index for grayscale images
    ssim = ssim_from_image_list(src, pixelation)
    print("SSIM mean: " + str(numpy.mean(ssim)))

    # Calculate the SSIM index for color images
    # (implemented as the average of SSIM values independently for the three components)
    ssim_multichannel = ssim_multichannel_from_image_list(src, pixelation)
    print("SSIM Multichannel mean: " + str(numpy.mean(ssim_multichannel)))

    # Calculate SSIM and Multiscale SSIM using TensorFlow
    ssim_tf, ssim_tf_multiscale = ssim_tf_from_image_list(src, pixelation)
    print("SSIM TensorFlow mean: " + str(numpy.mean(ssim_tf)))
    print("SSIM Multiscale TensorFlow mean: " + str(numpy.mean(ssim_tf_multiscale)))

    # Calculate and print PSNR and SSIM values for blurring
    print("BLURRING")

    # Calculate the PSNR index
    psnr = psnr_from_image_list(src, blurring)
    print("PSNR mean: " + str(numpy.mean(psnr)))

    # Calculate the SSIM index for grayscale images
    ssim = ssim_from_image_list(src, blurring)
    print("SSIM mean: " + str(numpy.mean(ssim)))

    # Calculate the SSIM index for color images
    # (implemented as the average of SSIM values independently for the three components)
    ssim_multichannel = ssim_multichannel_from_image_list(src, blurring)
    print("SSIM Multichannel mean: " + str(numpy.mean(ssim_multichannel)))

    # Calculate SSIM and Multiscale SSIM using TensorFlow
    ssim_tf, ssim_tf_multiscale = ssim_tf_from_image_list(src, blurring)
    print("SSIM TensorFlow mean: " + str(numpy.mean(ssim_tf)))
    print("SSIM Multiscale TensorFlow mean: " + str(numpy.mean(ssim_tf_multiscale)))


if __name__ == "__main__":
    """Main entry point for calculating the Frechet Inception Distance.

    This script calculates the Frechet Inception Distance (FID) between two videos,
    video A and video B, using the I3D Inception architecture. The FID is a metric
    used to evaluate the quality of generated images or videos by comparing them
    to real images or videos. The result is printed to the screen.

    The script requires the paths to the two videos, the number of frames to process,
    and the directory containing the model files.

    Command Line Arguments:
        --video_A (str): Path to video A.
        --video_B (str): Path to video B.
        --num_frames (int): The number of frames to process.
        --model_dir (str): Path to the directory containing the model files.

    Example:
        python ssim_psnr.py --video_A path/to/videoA.mp4 --video_B path/to/videoB.mp4
                            --num_frames 25 --model_dir path/to/model_dir
    """
    arg_parser = argparse.ArgumentParser(
        description="Calculates the Frechet Inception Distance between video A and video B"
                    " using the I3D Inception architecture described in: 'https://arxiv.org/abs/1705.07750"
                    "\n The result is printed to screen."
    )
    arg_parser.add_argument("--video_A", help="Path to video A")
    arg_parser.add_argument("--video_B", help="Path to video B")
    arg_parser.add_argument("--num_frames", help="The number of frames", type=int)
    arg_parser.add_argument("--model_dir", help="Path to the directory containing the model files")

    args = arg_parser.parse_args()

    CASCADE_CLASSIFIER_CONFIG_PATH = os.path.join(args.model_dir, CASCADE_CLASSIFIER_CONFIG_FILENAME)

    results = calculate_values(
        args.video_A, args.video_B, CASCADE_CLASSIFIER_CONFIG_PATH, args.num_frames
    )

    print(results)
