'''
Compute the attribute preservation index based on AUs extracted with OpenFace software
or using the penultimate layer of a VGG net trained for emotion recognition,
according to the type of arguments passed to the script
'''
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas
from sklearn.metrics import mean_squared_error


# List of Action Units (AUs) used for attribute preservation metrics.
# AUs are specific facial muscle movements encoded by the Facial Action Coding System (FACS).
ACTION_UNITS = [
    "au1", "au2", "au4", "au5", "au6", "au7", "au9",
    "au10", "au12", "au14", "au15", "au17",
    "au20", "au23", "au25", "au26", "au45"
]

# Path to the Haar Cascade Classifier configuration file for face detection.
CASCADE_CLASSIFIER_CONFIG_FILENAME = "haarcascade_frontalface_default.xml"


def concordance_correlation_coefficient(
        y_true, y_pred,
        sample_weight=None,
        multioutput='uniform_average'
):
    """Calculates the Concordance Correlation Coefficient (CCC).

    The CCC is a measure of inter-rater agreement, assessing the deviation of the 
    relationship between predicted and true values from the 45-degree line.

    For more information, see: 
    https://en.wikipedia.org/wiki/Concordance_correlation_coefficient

    Original paper: Lawrence, I., and Kuei Lin. "A concordance correlation coefficient 
    to evaluate reproducibility." Biometrics (1989): 255-268.

    Args:
        y_true (array-like): Ground truth (correct) target values, with shape 
            (n_samples,) or (n_samples, n_outputs).
        y_pred (array-like): Estimated target values, with shape 
            (n_samples,) or (n_samples, n_outputs).
        sample_weight (optional): Sample weights. Defaults to None.
        multioutput (str, optional): Defines aggregating of multiple output values. 
            Defaults to 'uniform_average'.

    Returns:
        float: A value in the range [-1, 1]. A value of 1 indicates perfect agreement 
        between the true and predicted values.

    Example:
        >>> y_true = [3, -0.5, 2, 7]
        >>> y_pred = [2.5, 0.0, 2, 8]
        >>> concordance_correlation_coefficient(y_true, y_pred)
        0.97678916827853024
    """
    # Calculate the Pearson correlation coefficient between true and predicted values
    cor = np.corrcoef(y_true, y_pred)[0][1]

    # Calculate means of true and predicted values
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    # Calculate variances of true and predicted values
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    # Calculate standard deviations of true and predicted values
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)

    # Calculate the numerator of the CCC formula
    numerator = 2 * cor * sd_true * sd_pred

    # Calculate the denominator of the CCC formula
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    # Return the CCC value
    return numerator / denominator


def run_openface_script(openface_script_path: str, input_vid_path: str, output_folder_path: str, verbose=False) -> str:
    """Executes the OpenFace script on a given video file and stores the output in a specified folder.

    This function runs an external OpenFace script, which processes a video file to extract facial features
    and saves the results as a CSV file in the specified output directory.

    Args:
        openface_script_path (str): The file path to the OpenFace script to be executed.
        input_vid_path (str): The file path to the input video file to be processed.
        output_folder_path (str): The directory path where the output CSV file will be saved.
        verbose (bool, optional): If True, the script's output will be printed to the console. Defaults to False.

    Returns:
        str: The file path to the generated CSV file containing the processed data.

    Raises:
        FileNotFoundError: If the specified OpenFace script or input video file does not exist.
        PermissionError: If there are insufficient permissions to execute the script or write to the output directory.
    """
    # Convert input video and output folder paths to absolute paths
    input_vid_absolute_path = os.path.abspath(input_vid_path)
    output_folder_absolute_path = os.path.abspath(output_folder_path)

    # Construct the command to execute the OpenFace script
    command = [openface_script_path, input_vid_absolute_path, output_folder_absolute_path]

    # Execute the command with or without verbose output
    if verbose:
        # Run the process and output to standard streams
        process = subprocess.Popen(
            command, shell=False, stderr=sys.stderr, stdout=sys.stdout, stdin=sys.stdin
        )
    else:
        # Run the process and suppress standard output
        process = subprocess.Popen(
            command, shell=False, stderr=sys.stderr, stdout=subprocess.DEVNULL, stdin=sys.stdin
        )
    
    # Wait for the process to complete
    process.wait()

    # Generate the CSV filename based on the input video filename
    basename_wo_ext = os.path.basename(input_vid_path).rsplit(".", 1)[0]
    csv_filename = basename_wo_ext + ".csv"

    # Return the full path to the generated CSV file
    return os.path.join(output_folder_absolute_path, csv_filename)


def load_and_parse_openface_csv(csv_file_path):
    """Loads and parses an OpenFace CSV file to extract action unit data.

    This function reads a CSV file generated by the OpenFace tool, which contains
    frame-by-frame data of facial action units (AUs). It extracts the relevant
    columns and returns them as a dictionary of NumPy arrays for further analysis.

    Args:
        csv_file_path (str): The file path to the OpenFace CSV file to be loaded.

    Returns:
        dict: A dictionary where keys are the names of the data fields and values
        are NumPy arrays containing the data for each field. The dictionary includes:
            - "frames": Frame numbers.
            - "au1": Intensity of Action Unit 1.
            - "au2": Intensity of Action Unit 2.
            - "au4": Intensity of Action Unit 4.
            - "au5": Intensity of Action Unit 5.
            - "au6": Intensity of Action Unit 6.
            - "au7": Intensity of Action Unit 7.
            - "au9": Intensity of Action Unit 9.
            - "au10": Intensity of Action Unit 10.
            - "au12": Intensity of Action Unit 12.
            - "au14": Intensity of Action Unit 14.
            - "au15": Intensity of Action Unit 15.
            - "au17": Intensity of Action Unit 17.
            - "au20": Intensity of Action Unit 20.
            - "au23": Intensity of Action Unit 23.
            - "au25": Intensity of Action Unit 25.
            - "au26": Intensity of Action Unit 26.
            - "au45": Intensity of Action Unit 45.

    Raises:
        FileNotFoundError: If the specified CSV file does not exist.
        pd.errors.EmptyDataError: If the CSV file is empty.
        pd.errors.ParserError: If the CSV file cannot be parsed.
    """
    df = pandas.read_csv(csv_file_path, sep=",")

    return {
        "frames": (df["frame"]).to_numpy(),
        "au1": (df[" AU01_r"]).to_numpy(),
        "au2": (df[" AU02_r"]).to_numpy(),
        "au4": (df[" AU04_r"]).to_numpy(),
        "au5": (df[" AU05_r"]).to_numpy(),
        "au6": (df[" AU06_r"]).to_numpy(),
        "au7": (df[" AU07_r"]).to_numpy(),
        "au9": (df[" AU09_r"]).to_numpy(),
        "au10": (df[" AU10_r"]).to_numpy(),
        "au12": (df[" AU12_r"]).to_numpy(),
        "au14": (df[" AU14_r"]).to_numpy(),
        "au15": (df[" AU15_r"]).to_numpy(),
        "au17": (df[" AU17_r"]).to_numpy(),
        "au20": (df[" AU20_r"]).to_numpy(),
        "au23": (df[" AU23_r"]).to_numpy(),
        "au25": (df[" AU25_r"]).to_numpy(),
        "au26": (df[" AU26_r"]).to_numpy(),
        "au45": (df[" AU45_r"]).to_numpy()
    }


def keep_common_frames(au_A, au_B):
    """Identifies and retains common frames between two sets of Action Unit (AU) data.

    This function takes two dictionaries containing AU data and returns two new dictionaries
    that only include data for frames that are common to both input dictionaries.

    Args:
        au_A (dict): A dictionary containing AU data for the first set. The dictionary should
            have keys corresponding to 'frames' and various AU intensities.
        au_B (dict): A dictionary containing AU data for the second set. The dictionary should
            have keys corresponding to 'frames' and various AU intensities.

    Returns:
        tuple: A tuple containing two dictionaries. Each dictionary contains AU data for frames
        that are common to both input dictionaries. The structure of the dictionaries is the same
        as the input dictionaries, excluding the 'frames' key.

    """
    # Identify common frames between the two AU datasets
    common_frames = tuple([
        int(A_frame_num) for A_frame_num in au_A["frames"]
        if A_frame_num in au_B["frames"]
    ])
    dict_keys = list(au_A.keys())

    def extract_common_values(au_X):
        """Extracts AU data for common frames from a given AU dataset.

        Args:
            au_X (dict): A dictionary containing AU data, including 'frames' and AU intensities.

        Returns:
            dict: A dictionary containing AU data only for the common frames.
        """
        X_common = {key: list() for key in dict_keys}
        for common_frame in common_frames:
            # Find indices of frames that match the current common frame
            matching_frames = [
                i for i, frame in enumerate(au_X["frames"]) if frame == common_frame
            ]
            # Warn if more than one matching frame is found and choose the first
            if len(matching_frames) != 1:
                print("WARNING: More than one matching common frame. Choosing first.")

            idx = matching_frames[0]

            # Append AU data for the common frame to the result dictionary
            for key in dict_keys:
                if key == "frames":
                    continue
                X_common[key].append(au_X[key][idx])

        return X_common

    # Extract common AU data for both input datasets
    reta, retb = extract_common_values(au_A), extract_common_values(au_B)
    return reta, retb


def get_results(au_A: dict, au_B: dict) -> dict:
    """Calculates correlation coefficient, RMSE, and CCC for each action unit.

    This function computes the correlation coefficient, root mean square error (RMSE),
    and concordance correlation coefficient (CCC) for each action unit present in the
    provided AU datasets.

    Args:
        au_A (dict): A dictionary containing AU data for the first set. The keys should
            correspond to action units, and the values should be lists of AU intensities.
        au_B (dict): A dictionary containing AU data for the second set. The keys should
            correspond to action units, and the values should be lists of AU intensities.

    Returns:
        dict: A dictionary where each key is an action unit and the value is a list
        containing the correlation coefficient, RMSE, and CCC for that action unit.
    """
    results = dict()

    for key in ACTION_UNITS:
        # Calculate the correlation coefficient between the two sets of AU data
        corr_coef_val = np.ma.corrcoef(au_A[key], au_B[key])[0, 1].tolist()
        # Calculate the root mean square error between the two sets of AU data
        rmse_val = np.sqrt(mean_squared_error(au_A[key], au_B[key])).tolist()
        # Calculate the concordance correlation coefficient between the two sets of AU data
        ccc = concordance_correlation_coefficient(au_A[key], au_B[key])

        results[key] = [corr_coef_val, rmse_val, ccc]

    return results


def get_attribute_preservation_metrics(A_vid_AUs_path: str, B_vid_AUs_path: str) -> dict:
    """Computes attribute preservation metrics for two video AU datasets.

    This function loads AU data from two video files, identifies common frames,
    and calculates the attribute preservation metrics for those frames.

    Args:
        A_vid_AUs_path (str): The file path to the AU data of the first video.
        B_vid_AUs_path (str): The file path to the AU data of the second video.

    Returns:
        dict: A dictionary containing the attribute preservation metrics for the
        common frames in the two AU datasets.
    """
    # Load and parse AU data from the provided CSV files
    au_A = load_and_parse_openface_csv(csv_file_path=A_vid_AUs_path)
    au_B = load_and_parse_openface_csv(csv_file_path=B_vid_AUs_path)

    # Extract AU data for frames common to both datasets
    au_A_common, au_B_common = keep_common_frames(au_A, au_B)
    return get_results(au_A_common, au_B_common)


def compare_aus(A_vid_AUs_path: str, B_vid_AUs_path: str, output_path: str) -> None:
    """Compares AU data from two videos and writes the results to a file.

    This function computes the attribute preservation metrics for AU data from
    two videos and writes the results to a specified output file in JSON format.

    Args:
        A_vid_AUs_path (str): The file path to the AU data of the first video.
        B_vid_AUs_path (str): The file path to the AU data of the second video.
        output_path (str): The file path where the results will be written.

    Returns:
        None
    """
    # Compute attribute preservation metrics for the two video AU datasets
    results_dict = get_attribute_preservation_metrics(A_vid_AUs_path, B_vid_AUs_path)
    # Write the results to the specified output file in JSON format
    with open(output_path, "w") as ostream:
        json.dump(results_dict, ostream)


if __name__ == '__main__':
    """Main entry point for the script.

    This script compares Action Unit (AU) data from two videos and writes the
    results to a specified output file. It uses command-line arguments to
    specify the input and output file paths, as well as other necessary
    parameters.

    Command-line Arguments:
        --A_vid_AUs (str): Path to the CSV file containing AU intensities for video A.
        --B_vid_AUs (str): Path to the CSV file containing AU intensities for video B.
        --video_A (str): Path to the source video file.
        --video_B (str): Path to the faceswap video file.
        --faces (str): Path to the precomputed face locations list.
        --model_dir (str): Path to the directory containing the model files.
        -o, --output_path (str): Path where the results will be saved.

    Raises:
        FileNotFoundError: If the specified face locations path does not exist.
    """
    arg_parser = argparse.ArgumentParser(
        description="Compare AU data from two videos and output the results."
    )
    arg_parser.add_argument("--A_vid_AUs", help="CSV file of AU intensities as output by OpenFace for video A")
    arg_parser.add_argument("--B_vid_AUs", help="CSV file of AU intensities as output by OpenFace for video B")
    arg_parser.add_argument("--video_A", help="Source video")
    arg_parser.add_argument("--video_B", help="Faceswap video")
    arg_parser.add_argument("--faces", help="Path to face locations list if already computed")
    arg_parser.add_argument("--model_dir", help="Path to the directory containing the model files")
    arg_parser.add_argument("-o", "--output_path", help="Where to save the results")

    args = arg_parser.parse_args()

    if not os.path.exists(args.faces):
        os.makedirs('/'.join(args.faces.split('/')[:-1]))

    CASCADE_CLASSIFIER_CONFIG_PATH = os.path.join(args.model_dir, CASCADE_CLASSIFIER_CONFIG_FILENAME)

    if args.A_vid_AUs is not None:
        print("Starting AUs comparison... \nThe result will be stored in ", args.output_path)
        compare_aus(
            args.A_vid_AUs, args.B_vid_AUs, args.output_path
        )
    else:
        pass
