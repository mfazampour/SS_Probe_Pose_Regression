import os

import numpy as np
import SimpleITK as sitk
from scipy.interpolate import splprep, splev

from imfusion_free.ultrasound_fan import create_ultrasound_mask


def read_spline_data_from_file(file_path):
    with open(file_path, 'r') as f:
        header = f.readline()  # Read the header
        data = f.readline().strip().split(';')

        input_path = data[0].strip()
        output_path = data[1].strip()

        if output_path.endswith('.imf'):
            # set it to parent folder
            output_path = os.path.dirname(output_path)


        trans_spline_str = data[2].strip().split()
        trans_spline = [(float(trans_spline_str[i]), float(trans_spline_str[i + 1]), float(trans_spline_str[i + 2])) for
                        i in range(0, len(trans_spline_str), 3)]

        dir_spline_str = data[3].strip().split()
        dir_spline = [(float(dir_spline_str[i]), float(dir_spline_str[i + 1]), float(dir_spline_str[i + 2])) for i in
                      range(0, len(dir_spline_str), 3)]

    return input_path, output_path, trans_spline, dir_spline


def slice_volume_from_splines(input_path, output_path, transducer_spline_points, direction_spline_points, w, l,
                              target_resolution=(1.0, 1.0)):
    # Read the volume using SimpleITK
    volume = sitk.ReadImage(input_path)

    transducer_spline_interpolator, _ = splprep(np.array(transducer_spline_points).T, k=2)
    direction_spline_interpolators, _ = splprep(np.array(direction_spline_points).T, k=2)

    u_sampled = np.linspace(0, 1, 11)
    transducer_positions = splev(u_sampled, transducer_spline_interpolator)
    transducer_positions = np.array(transducer_positions).T.tolist()
    direction_points_on_spline = splev(u_sampled, direction_spline_interpolators)
    direction_points_on_spline = np.array(direction_points_on_spline).T.tolist()

    # transducer_positions = [np.poly1d(transducer_spline_points[i])(t) for i in range(3) for t in t_values]
    # direction_points_on_spline = [np.poly1d(direction_spline_points[i])(t) for i in range(3) for t in t_values]

    slices = []

    for i in range(len(transducer_positions) - 1):
        down_direction = np.array(transducer_positions[i]) - np.array(direction_points_on_spline[i])
        slice_normal = np.array(direction_points_on_spline[i + 1]) - np.array(direction_points_on_spline[i])

        # normalize slice_normal to unit vector
        slice_normal /= np.linalg.norm(slice_normal)
        # normalize down_direction to unit vector
        down_direction /= np.linalg.norm(down_direction)

        # Compute left direction
        left_direction = np.cross(slice_normal, down_direction)
        left_direction /= np.linalg.norm(left_direction)
        slice_origin = transducer_positions[i] + w * 0.5 * volume.GetSpacing()[1] * left_direction

        slice_image = interpolate_arbitrary_plane(slice_origin, slice_normal, down_direction, volume, (l, w))

        # # Extracting relevant metadata from the original volume
        # slice_image = sitk.GetImageFromArray(slice_values)
        # slice_image.SetDirection(volume.GetDirection())
        # slice_image.SetSpacing((volume.GetSpacing()[0], volume.GetSpacing()[1]))
        # slice_image.SetOrigin(slice_origin)

        # Save the slice with the updated metadata
        sitk.WriteImage(slice_image, f"{output_path}/slice_{i:04d}.nii.gz")

    return slices


def interpolate_arbitrary_plane(slice_origin, slice_normal, down_direction, volume: sitk.Image, slice_shape):
    # Extract properties from the SimpleITK Image
    spacing = volume.GetSpacing()
    size = volume.GetSize()

    # Define plane's coordinate system
    e1 = -1 * down_direction  # Just an example vector; you might need something more sophisticated.
    e1 = e1 - np.dot(e1, slice_normal) * slice_normal  # Make e1 orthogonal to the plane normal
    e1 /= np.linalg.norm(e1)  # Normalize e1 to make it a unit vector
    # e1 = down_direction

    e2 = np.cross(slice_normal, e1)  # e2 is now a unit vector orthogonal to both slice_normal and e1

    # Direction for the resampler will be (e1, e2, slice_normal) flattened
    direction = np.stack([e1, e2, slice_normal], axis=-1).flatten()

    # Use SimpleITK's resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection(direction.tolist())
    resampler.SetOutputOrigin(slice_origin)
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize((slice_shape[0], slice_shape[1], 3))
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    # Resample the volume on the arbitrary plane
    sliced_image = resampler.Execute(volume)

    slice_values = sitk.GetArrayFromImage(sliced_image)
    # print value range of slice
    print(f"Slice value range: {np.min(slice_values)} - {np.max(slice_values)}")

    return sliced_image


# Example usage:
file_path = "/mnt/projects/aorta_scan/random_simulated_ultrasound/CT001/batch.txt"
input_path, output_path, trans_spline, dir_spline = read_spline_data_from_file(file_path)

origin = (106, 246)
opening_angle = 70  # in degrees
short_radius = 124  # in pixels
long_radius = 512  # in pixels
img_shape = (512, 512)  # this is not important, just for plotting
_, width_, len_ = create_ultrasound_mask(origin, opening_angle, short_radius, long_radius, img_shape)
ultrasound_spacing = (0.4, 0.4)
slices = slice_volume_from_splines(input_path, output_path, trans_spline, dir_spline,
                                   int(width_ * 1.2 * ultrasound_spacing[0]), int(len_ * 1.2 * ultrasound_spacing[1]))
