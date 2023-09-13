import argparse
import os

from slice_volume import slice_volume_from_splines
from ultrasound_fan import create_ultrasound_mask
from slice_volume import read_spline_data_from_file

if __name__ == "__main__":
    # read arguments using argparse
    parser = argparse.ArgumentParser(description='slice all of the CT data')
    # path to the batch files
    parser.add_argument('--batch_file_path', required=True, type=str, help='The path to the batch file')
    args = parser.parse_args()

    # search for batch files in the batch file folder in all subfolders and store in a list
    batch_files = []
    for root, dirs, files in os.walk(args.batch_file_path):
        for file in files:
            if "batch.txt" in file:
                # define the path to the batch file
                batch_file_path = os.path.join(root, file)
                batch_files.append(batch_file_path)

    # create the ultrasound mask
    origin = (106, 246)
    opening_angle = 70  # in degrees
    short_radius = 124  # in pixels
    long_radius = 512  # in pixels
    img_shape = (512, 512)
    _, width_, len_ = create_ultrasound_mask(origin, opening_angle, short_radius, long_radius, img_shape)
    ultrasound_spacing = (0.4, 0.4)  # in mm

    # iterate over the batch files
    for batch_file in batch_files:

        # read spline data from the batch file
        input_paths, output_paths, trans_splines, dir_splines = read_spline_data_from_file(batch_file)

        # for each input path and output path and spline, slice the volume
        for i, (input_path, output_path, trans_spline, dir_spline) in enumerate(zip(input_paths, output_paths, trans_splines, dir_splines)):

            # create the output folder
            output_path = os.path.join(output_path, "slices")
            os.makedirs(output_path, exist_ok=True)
            # create the output folder for the specific index
            output_path = os.path.join(output_path, f"slices_{i}")
            os.makedirs(output_path, exist_ok=True)

            # slice the volume
            slice_volume_from_splines(input_path, output_path, trans_spline, dir_spline,
                                               int(width_ * 1.2 * ultrasound_spacing[0]),
                                               int(len_ * 1.2 * ultrasound_spacing[1]))
