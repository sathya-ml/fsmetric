from typing import Iterable, Optional
import cv2
import numpy


def extract_random_frames_from_video(video_path: str, num_frames: Optional[int] = None, exact_frames: Optional[Iterable[int]] = None) -> list:
    """Extracts random or specific frames from a video file.

    This function allows for the extraction of either a specified number of random frames
    or specific frames from a video file. The function reads the video file and either
    selects random frames based on the `num_frames` parameter or specific frames based on
    the `exact_frames` parameter.

    Args:
        video_path (str): The path to the video file from which frames are to be extracted.
        num_frames (int, optional): The number of random frames to extract. Defaults to None.
        exact_frames (Iterable[int], optional): Specific frame indices to extract. Defaults to None.

    Returns:
        list: A list of extracted frames as images.

    Raises:
        ValueError: If both `num_frames` and `exact_frames` are None.

    Notes:
        - If `num_frames` is provided, the function extracts that many random frames.
        - If `exact_frames` is provided, the function extracts frames at those specific indices.
        - If both parameters are None, a ValueError is raised.
    """
    if num_frames is not None:
        # Open the video file
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()

        video_frames = list()
        # Read all frames from the video
        while success:
            video_frames.append(image.copy())
            success, image = vidcap.read()

        vidcap.release()

        # Select random indices for the frames
        indices = numpy.random.randint(len(video_frames), size=num_frames)

        # Return the list of frames
        return [video_frames[idx] for idx in indices]

    if exact_frames is not None:
        # Open the video
        vidcap = cv2.VideoCapture(video_path)

        # Read the frame
        success, image = vidcap.read()
        count = 0

        video_frames = list()

        # Iterate over frames and select those at the specified indices
        while success:
            if count in exact_frames:
                video_frames.append(image.copy())

            success, image = vidcap.read()
            count += 1

        # Close the file
        vidcap.release()

        # Return the list of frames (without transformations)
        return video_frames

    else:
        raise ValueError("Both arguments None")


def get_random_frames_source_destination(vid_path_src: str, vid_path_dst: str, num_frames: int):
    """Extracts random frames from two video files at the same indices.

    This function opens two video files, determines the number of frames in each,
    and extracts a specified number of random frames from both videos at the same
    indices. The number of frames to extract is determined by the minimum length
    of the two videos to avoid selecting excess frames.

    Args:
        vid_path_src (str): Path to the source video file.
        vid_path_dst (str): Path to the destination video file.
        num_frames (int): Number of random frames to extract.

    Returns:
        tuple: A tuple containing two lists of frames, one from each video.
    """
    # Open both videos
    vidcap1 = cv2.VideoCapture(vid_path_src)
    vidcap2 = cv2.VideoCapture(vid_path_dst)

    # Find the number of frames in both videos
    len_src = int(vidcap1.get(cv2.CAP_PROP_FRAME_COUNT))
    len_dst = int(vidcap2.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine the maximum number of frames to extract
    # (choose the minimum length to avoid selecting excess frames)
    length = min(len_src, len_dst)

    # Extract num_frames random indices
    indices = numpy.random.randint(length, size=num_frames)

    # Close the files
    vidcap1.release()
    vidcap2.release()

    # Extract frames from both videos
    # (not enough memory to save both lists and then do a list comprehension,
    # so extract and save directly what is needed one video at a time)
    return extract_random_frames_from_video(vid_path_src, exact_frames=indices), extract_random_frames_from_video(
        vid_path_dst,
        exact_frames=indices)


def get_random_frames(vid_path_src: str, num_frames: int):
    """Extracts random frames from a single video file.

    This function opens a video file, determines the number of frames,
    and extracts a specified number of random frames.

    Args:
        vid_path_src (str): Path to the source video file.
        num_frames (int): Number of random frames to extract.

    Returns:
        list: A list of randomly extracted frames from the video.
    """
    # Open the video
    vidcap1 = cv2.VideoCapture(vid_path_src)

    # Find the number of frames
    len_src = int(vidcap1.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine the maximum number of frames to extract
    length = len_src

    # Extract num_frames random indices
    indices = numpy.random.randint(length, size=num_frames)

    # Close the file
    vidcap1.release()

    # Extract frames from the video
    # (not enough memory to save both lists and then do a list comprehension,
    # so extract and save directly what is needed one video at a time)
    return extract_random_frames_from_video(vid_path_src, exact_frames=indices)
