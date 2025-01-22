import argparse
import json

import face_recognition
import numpy
import tqdm

from fsmetric.fsmlib import extract_random_frames_from_video


def get_distances_vector(frames, encoding):
    """Calculates the face distances between a given encoding and encodings from a list of frames.

    This function iterates over each frame in the provided list, extracts face encodings,
    and calculates the face distance between the given encoding and the first encoding found
    in each frame. If no face is detected in a frame, that frame is skipped.

    Args:
        frames (list): A list of video frames (images) from which to extract face encodings.
        encoding (numpy.ndarray): A face encoding to compare against the encodings extracted from the frames.

    Returns:
        list: A list of face distances between the given encoding and the encodings extracted from the frames.
    """
    results_list = list()
    for frame in frames:
        frame_encodings = face_recognition.face_encodings(frame)
        if len(frame_encodings) == 0:
            continue
        frame_encoding = frame_encodings[0]
        face_distance = face_recognition.face_distance([encoding], frame_encoding)[0]
        results_list.append(face_distance)

    return results_list


def calculate_face_recognition_distances(
        vid_path: str, A_enc_path: str, B_enc_path: str, num_frames: int
) -> dict:
    """Calculates the mean and standard deviation of face distances for two reference images against video frames.

    This function loads two reference images, extracts their face encodings, and compares these encodings
    against face encodings extracted from a specified number of frames sampled randomly from a video.
    It calculates the mean and standard deviation of the face distances for both reference images.

    Args:
        vid_path (str): Path to the video file from which frames are extracted.
        A_enc_path (str): Path to the image file for person A's face encoding.
        B_enc_path (str): Path to the image file for person B's face encoding.
        num_frames (int): Number of frames to extract from the video for comparison.

    Returns:
        dict: A dictionary containing the video path, encoding image paths, number of frames,
              and the mean and standard deviation of the face distances for both A and B.
    """
    A_img = face_recognition.load_image_file(A_enc_path)
    B_img = face_recognition.load_image_file(B_enc_path)
    
    A_encoding = face_recognition.face_encodings(A_img)[0]
    B_encoding = face_recognition.face_encodings(B_img)[0]

    video_frames = extract_random_frames_from_video(vid_path, num_frames)

    A_distances_vec = get_distances_vector(video_frames, A_encoding)
    B_distances_vec = get_distances_vector(video_frames, B_encoding)

    results = {
        "vid_path": vid_path,
        "A_enc_path": A_enc_path,
        "B_enc_path": B_enc_path,
        "num_frames": num_frames,

        "A_mean": numpy.mean(A_distances_vec).tolist(),
        "B_mean": numpy.mean(B_distances_vec).tolist(),
        "A_std": numpy.std(A_distances_vec).tolist(),
        "B_std": numpy.std(B_distances_vec).tolist(),
    }

    return results


def get_frame_encodings(frames):
    """Extracts face encodings from a list of video frames.

    This function iterates over each frame in the provided list and extracts
    the face encodings using the face_recognition library. If no face is detected
    in a frame, it is skipped.

    Args:
        frames (list): A list of video frames (images) from which to extract face encodings.

    Returns:
        list: A list of face encodings extracted from the frames.
    """
    encodings_list = list()
    for frame in tqdm.tqdm(frames):
        import code
        code.interact(local=locals())

        frame_encodings = face_recognition.face_encodings(frame)
        if len(frame_encodings) == 0:
            continue
        encodings_list.append(frame_encodings[0])

    return encodings_list


def compare_encoding_distances(original_encodings, swapped_encodings):
    """Calculates the face distance between original and swapped encodings.

    This function computes the face distance between each encoding in the original_encodings
    list and each encoding in the swapped_encodings list using the face_recognition library.

    Args:
        original_encodings (list): A list of face encodings from the original video.
        swapped_encodings (list): A list of face encodings from the swapped video.

    Returns:
        list: A list of face distances between the original and swapped encodings.
    """
    distances_list = list()
    for original_encoding in original_encodings:
        for swapped_encoding in swapped_encodings:
            face_distance = face_recognition.face_distance([original_encoding], swapped_encoding)[0]
            distances_list.append(face_distance)

    return distances_list


def calculate_face_recognition_distances_no_enc(
        original_vid_path: str, swapped_vid_path: str, num_frames: int
) -> dict:
    """Calculates face recognition distances between two videos without precomputed encodings.

    This function extracts random frames from two videos, computes face encodings for each frame,
    and calculates the mean and standard deviation of the face distances between the encodings
    from the original and swapped videos.

    Args:
        original_vid_path (str): Path to the original video file.
        swapped_vid_path (str): Path to the swapped video file.
        num_frames (int): Number of frames to extract from each video for comparison.

    Returns:
        dict: A dictionary containing the paths of the original and swapped videos, the number of frames,
              and the mean and standard deviation of the face distances.
    """
    original_video_frames = extract_random_frames_from_video(original_vid_path, num_frames)
    swapped_video_frames = extract_random_frames_from_video(swapped_vid_path, num_frames)

    original_encodings = get_frame_encodings(original_video_frames)
    swapped_encodings = get_frame_encodings(swapped_video_frames)

    distances_vector = compare_encoding_distances(
        original_encodings=original_encodings,
        swapped_encodings=swapped_encodings
    )

    results = {
        "video_A_path": original_vid_path,
        "video_B_path": swapped_vid_path,
        "num_frames": num_frames,
        "mean_dist": numpy.mean(distances_vector).tolist(),
        "std_dist": numpy.std(distances_vector).tolist()
    }

    return results


def main(vid_path: str, A_enc_path: str, B_enc_path: str, output_path: str, num_frames: int) -> None:
    """Calculates and saves the mean and standard deviation of face encoding distances.

    This function calculates the mean and standard deviation of the distances between
    the face encodings of two persons (A and B) and the face encodings extracted from
    a specified number of frames sampled randomly from a video. The results are saved
    to a specified output path in JSON format.

    Args:
        vid_path (str): The path to the video file against which to calculate distances.
        A_enc_path (str): The path to an image file of person A to extract A's face encoding.
        B_enc_path (str): The path to an image file of person B to extract B's face encoding.
        output_path (str): The path where the output JSON file will be saved.
        num_frames (int): The number of frames to sample from the video for comparison.

    Returns:
        None
    """
    results = calculate_face_recognition_distances(
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
