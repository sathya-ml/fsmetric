import argparse
import json
import os
import tempfile

# suppress all TF logs except ERROR
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # or any other in {"0", "1", "2", "3"}

import fsmetric

METRIC_NAMES = ["deid", "attr_pres", "photo_reality"]


def get_argument_parser():
    """
    Creates and returns an argument parser for the script.

    The parser includes arguments for specifying the metrics to compute, paths to the input videos,
    the output file, and additional optional parameters such as the number of frames, the path to
    the OpenFace script, the model directory, and a verbosity flag.

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    argument_parser = argparse.ArgumentParser(
        description="This script processes two input videos to compute specified metrics, such as "
                    "de-identification, attribute preservation, and photorealism. It requires paths to both "
                    "videos and outputs the results in a JSON file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Positional argument for metrics
    argument_parser.add_argument(
        "metrics",
        nargs="+",
        help=f"A list of metrics to output for the inputs among {METRIC_NAMES}, or 'all' to include all metrics"
    )

    # Required arguments for input videos and output file
    argument_parser.add_argument(
        "-A",
        "--video_A",
        help="Path to video A, i.e. the source video - always required",
        required=True,
        type=str
    )
    argument_parser.add_argument(
        "-B",
        "--video_B",
        help="Path to video B, i.e. the destination video - always required",
        required=True,
        type=str
    )
    argument_parser.add_argument(
        "-o",
        "--output",
        help="Where to output results (in json) - always required",
        required=True,
        type=str
    )

    # Optional arguments with default values
    # argument_parser.add_argument(
    #     "--B_enc",
    #     help="An image of person B to extract B's face encoding - required for the de-id metric",
    #     type=str,
    #     required=False
    # )
    # argument_parser.add_argument(
    #     "--A_enc",
    #     help="An image of person A to extract A's face encoding - required for the de-id metric",
    #     type=str,
    #     required=False
    # )
    argument_parser.add_argument(
        "--num_frames",
        help="Number of video frames to use for de-identification - required for the de-id metric",
        type=int,
        default=50,
        required=False
    )
    argument_parser.add_argument(
        "--openface",
        help="Path to script/program/executable that extracts AUs with openface. "
             "It should take two positional arguments: the input video, "
             "and the output directory - required for the attribute preservation metric",
        type=str,
        default="fsmetric/extract_action_units_docker.sh",
        required=False
    )
    argument_parser.add_argument(
        "--model_dir",
        help="Path to the directory containing the model files - required for the photo-reality metric",
        type=str,
        default="models",
        required=False
    )
    argument_parser.add_argument(
        "-v",
        "--verbose",
        help="Prints output about what it's doing",
        action="store_true"
    )

    return argument_parser


def get_and_verify_clarg_requirements(parser):
    """
    Parses and verifies the command-line arguments.

    Ensures that the provided metrics are valid and that the required arguments for each metric are present.
    If the 'all' option is specified for metrics, it includes all available metrics.

    Args:
        parser (argparse.ArgumentParser): The argument parser.

    Returns:
        argparse.Namespace: The parsed and verified command-line arguments.
    """
    clargs = parser.parse_args()

    if "all" in clargs.metrics and len(clargs.metrics) != 1:
        parser.error(message="don't include individual metrics if you include 'all'")
    elif "all" in clargs.metrics:
        clargs.metrics = METRIC_NAMES
    for metric in clargs.metrics:
        if metric not in METRIC_NAMES:
            parser.error(message=f"unknown metric {metric}")

    if "deid" in clargs.metrics:
        # if clargs.A_enc is None:
        #     parser.error(message="--A_enc has to be given with the metric 'deid'")
        # if clargs.B_enc is None:
        #     parser.error(message="--B_enc has to be given with the metric 'deid'")
        if clargs.num_frames is None:
            parser.error(message="--num_frames has to be given with the metric 'deid'")
    if "attr_pres" in clargs.metrics:
        if clargs.openface is None:
            parser.error(message="--openface has to be given with the metric 'expression'")
    if "photo_reality" in clargs.metrics:
        if clargs.model_dir is None:
            parser.error(message="--model_dir has to be given with the metric 'photo_reality'")
        if clargs.num_frames is None:
            parser.error(message="--num_frames has to be given with the metric 'photo_reality'")

    return clargs


def main(clargs) -> None:
    """
    Main function to compute the specified metrics for the input videos.

    Depending on the specified metrics, it calculates de-identification, attribute preservation,
    and photorealism metrics for the input videos and saves the results in a JSON file.

    Args:
        clargs (argparse.Namespace): The parsed and verified command-line arguments.
    """
    deid_results, attribute_preservation_results, photo_reality_results = (
        None, None, None
    )

    # Calculate de-identification metric
    if "deid" in clargs.metrics:
        if clargs.verbose:
            print(f"Calculating face recognition distances for {clargs.video_A}")
        
        # A_deid_results = fsmetric.calculate_face_recognition_distances_no_enc(
        #     vid_path=clargs.video_A,
        #     A_enc_path=clargs.A_enc,
        #     B_enc_path=clargs.B_enc,
        #     num_frames=clargs.num_frames
        # )
        # if clargs.verbose:
        #     print(f"Calculating face recognition distances for {clargs.video_B}")
        # B_deid_results = fsmetric.calculate_face_recognition_distances_no_enc(
        #     vid_path=clargs.video_B,
        #     A_enc_path=clargs.A_enc,
        #     B_enc_path=clargs.B_enc,
        #     num_frames=clargs.num_frames
        # )

        deid_results = fsmetric.calculate_face_recognition_distances_no_enc(
            original_vid_path=clargs.video_A,
            swapped_vid_path=clargs.video_B,
            num_frames=clargs.num_frames
        )

        # FOR arcface
        # if clargs.verbose:
        #     print(f"Calculating face recognition distances for {clargs.video_A} with arcface")
        # A_deid_results_arc = fsmetric.calculate_values_arcface(
        #     vid_path=clargs.video_A,
        #     A_encoding_img=clargs.A_enc,
        #     B_encoding_img=clargs.B_enc,
        #     num_frames=clargs.num_frames
        # )
        # if clargs.verbose:
        #     print(f"Calculating face recognition distances for {clargs.video_B} with arcface")
        # B_deid_results_arc = fsmetric.calculate_values_arcface(
        #     vid_path=clargs.video_B,
        #     A_encoding_img=clargs.A_enc,
        #     B_encoding_img=clargs.B_enc,
        #     num_frames=clargs.num_frames
        # )
        # A_deid_results.update(A_deid_results_arc)
        # B_deid_results.update(B_deid_results_arc)

        # deid_results = {
        #     "video_A": A_deid_results,
        #     "video_B": B_deid_results
        # }
        
    # Calculate attribute preservation metric
    if "attr_pres" in clargs.metrics:
        with tempfile.TemporaryDirectory() as temp_dir_path:
            temp_dir_path = os.path.abspath("./tmp")
            if clargs.verbose:
                print(f"Extracting action units with OpenFace for {clargs.video_A}")
            A_vid_AUs_path = fsmetric.run_openface_script(
                openface_script_path=clargs.openface,
                input_vid_path=clargs.video_A,
                output_folder_path=temp_dir_path,
                verbose=clargs.verbose
            )

            if clargs.verbose:
                print(f"Extracting action units with OpenFace for {clargs.video_B}")
            B_vid_AUs_path = fsmetric.run_openface_script(
                openface_script_path=clargs.openface,
                input_vid_path=clargs.video_B,
                output_folder_path=temp_dir_path,
                verbose=clargs.verbose
            )

            if clargs.verbose:
                print(f"Estimating RMSE and PCC for the two videos")
            attribute_preservation_results = fsmetric.get_attribute_preservation_metrics(
                A_vid_AUs_path=A_vid_AUs_path,
                B_vid_AUs_path=B_vid_AUs_path,
            )
            attribute_preservation_results = {
                "video_A_path": clargs.video_A,
                "video_B_path": clargs.video_B,
                "results": attribute_preservation_results
            }

    # Calculate photorealism metric
    if "photo_reality" in clargs.metrics:
        if clargs.verbose:
            print(f"Calculating FID for the two videos")
        
        fid = fsmetric.calculate_video_FID(
            video_path_source=clargs.video_A,
            video_path_dest=clargs.video_B,
            cascade_classifier_config_path=os.path.join(
                clargs.model_dir, fsmetric.CASCADE_CLASSIFIER_CONFIG_FILENAME
            ),
            weights_path=os.path.join(
                clargs.model_dir, fsmetric.WEIGHTS_FILENAME
            )
        )

        if clargs.verbose:
            print(f"Calculating SSIM and PSNR for the two videos")

        ssim_psnr_results = fsmetric.calculate_ssim_psnr(
            vid_path_src=clargs.video_A,
            vid_path_dst=clargs.video_B,
            cascade_classifier_config_path=os.path.join(
                clargs.model_dir, fsmetric.CASCADE_CLASSIFIER_CONFIG_FILENAME
            ),
            num_frames=clargs.num_frames
        )

        photo_reality_results = {
            "video_A_path": clargs.video_A,
            "video_B_path": clargs.video_B,
            "FID": float(fid)
        }
        photo_reality_results.update(ssim_psnr_results)

    # Combine results into a dictionary and save to the output file
    output_dict = {
        "deid_results": deid_results,
        "attribute_preservation_results": attribute_preservation_results,
        "photo_reality_results": photo_reality_results
    }

    with open(clargs.output, "w") as ostream:
        json.dump(output_dict, ostream)

    if clargs.verbose:
        print(f"Saved results to {clargs.output}")


if __name__ == "__main__":
    argument_parser = get_argument_parser()
    arguments = get_and_verify_clarg_requirements(parser=argument_parser)

    main(clargs=arguments)
