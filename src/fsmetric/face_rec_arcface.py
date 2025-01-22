import argparse
import json

import cv2
import numpy
from scipy.spatial.distance import euclidean
from terran.face import extract_features
from terran.face import face_detection

from fsmetric import fsmlib


def calculate_values_arcface(vid_path: str, A_encoding_img: str, B_encoding_img: str, num_frames: int):
    """Calculates the mean and standard deviation of Euclidean distances between face encodings.

    This function extracts random frames from a video and compares the face encodings of these frames
    with the face encodings of two reference images. It calculates the Euclidean distance between the
    encodings to determine the similarity.

    Args:
        vid_path (str): Path to the video file.
        A_encoding_img (str): Path to the image file for person A's face encoding.
        B_encoding_img (str): Path to the image file for person B's face encoding.
        num_frames (int): Number of frames to extract from the video for comparison.

    Returns:
        dict: A dictionary containing the video path, encoding image paths, number of frames,
              and the mean and standard deviation of the Euclidean distances for both A and B.
    """
    # Extract random frames for testing
    dst_faces = fsmlib.get_random_frames(vid_path, num_frames)

    # Load the reference source and destination images
    src_A = cv2.imread(A_encoding_img)
    dest_B = cv2.imread(B_encoding_img)

    # Start comparisons with the swapped video
    src_swap_results = []
    dest_swap_results = []

    features_source_face = extract_features(src_A, faces_per_image=face_detection(src_A))
    features_destination_face = extract_features(dest_B, faces_per_image=face_detection(dest_B))
    for swap_face in dst_faces:
        features_swap_face = extract_features(swap_face, faces_per_image=face_detection(swap_face))

        # Calculate the Euclidean distance (values [0, inf])
        src_swap_results.append(euclidean(features_source_face, features_swap_face))
        dest_swap_results.append(euclidean(features_destination_face, features_swap_face))

    print("Video Swp - src recognition value mean: " + str(numpy.mean(src_swap_results)) + " and std: " + str(
        numpy.std(src_swap_results)))
    print("Video Swp - dst recognition value mean: " + str(numpy.mean(dest_swap_results)) + " and std: " + str(
        numpy.std(dest_swap_results)))

    results = {
        "vid_path": vid_path,
        "A_enc_path": A_encoding_img,
        "B_enc_path": B_encoding_img,
        "num_frames": num_frames,

        "A_mean_arc": numpy.mean(src_swap_results).tolist(),
        "B_mean_arc": numpy.mean(dest_swap_results).tolist(),
        "A_std_arc": numpy.std(src_swap_results).tolist(),
        "B_std_arc": numpy.std(dest_swap_results).tolist(),
    }

    return results


def main(vid_path: str, A_enc_path: str, B_enc_path: str, output_path: str, num_frames: int) -> None:
    """Calculates and saves the mean and standard deviation of the distances between face encodings.

    This function calculates the mean and standard deviation of the Euclidean distances between
    the face encodings of two persons (A and B) and the face encodings extracted from a specified
    number of frames sampled randomly from a video. The results are saved to a specified output path
    in JSON format.

    Args:
        vid_path (str): The path to the video file against which to calculate distances.
        A_enc_path (str): The path to an image file of person A to extract A's face encoding.
        B_enc_path (str): The path to an image file of person B to extract B's face encoding.
        output_path (str): The path where the output JSON file will be saved.
        num_frames (int): The number of frames to sample from the video for comparison.

    Returns:
        None
    """
    results = calculate_values_arcface(
        vid_path, A_enc_path, B_enc_path, num_frames
    )

    with open(output_path, "w") as ostream:
        json.dump(results, ostream)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description=(
            "Calculates the mean and standard deviation of the distances between the face encodings "
            "of two persons A and B and the face encodings from n frames of a video sampled at random "
            "with replacement. For info about the encodings see https://github.com/ageitgey/face_recognition/"
        )
    )
    arg_parser.add_argument("--vid", help="Video against which to calculate distances", required=True)
    arg_parser.add_argument("--A_enc", help="An image of person A to extract A's face encoding", required=True)
    arg_parser.add_argument("--B_enc", help="An image of person B to extract B's face encoding", required=True)
    arg_parser.add_argument("-n", "--num_frames", help="Number of frames for comparison", type=int, default=50)
    arg_parser.add_argument("-o", "--output_path", help="Output path", required=True)

    args = arg_parser.parse_args()

    main(
        args.vid,
        args.A_enc,
        args.B_enc,
        args.output_path,
        args.num_frames
    )
