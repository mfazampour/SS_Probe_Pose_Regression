import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset, DataLoader
import torch
import os
import nibabel as nib
import cv2
from torch.utils.data import random_split
import torchvision.transforms as transforms
from PIL import Image
import SimpleITK as sitk

from datasets.imfusion_free_slicing.ultrasound_fan import create_ultrasound_mask
from datasets.real_us_pose_dataset_with_gt_torch import get_center_pose

''' This data loader is designed to read a batch file originally created for ImFusion. Each row in the batch file has the following format:
    "volume path", "transducer spline", "direction spline", "output path"
For each row, the data loader reads the volume and the transducer spline, and returns a number of 2D slice from the volume, along with the transducer spline.
The direction spine determines the direction of the normal of the slices to be extracted from the volume. 
The dataset returns the generated slice along with the pose of the center of the slice.'''

SIZE_W = 512
SIZE_H = 512


def get_image_center_pose_as_quat(image_path, relative_to=None):
    """
    Get the 6D pose of the center of a NIfTI image.

    Parameters:
    - image_path: str, path to the .nii.gz image.

    Returns:
    - center_position: tuple, (x, y, z) coordinates of the center.
    - quaternion: array, orientation of the center in quaternion format [x, y, z, w].
    """

    # Read the NIfTI image
    image = sitk.ReadImage(image_path)

    # # 3D position of the center
    # center_pixel = [s // 2 for s in image.GetSize()]
    # center_position = image.TransformContinuousIndexToPhysicalPoint(center_pixel)
    #
    # # 3D orientation from the direction matrix
    # direction = np.array(image.GetDirection()).reshape((3, 3))

    affine = get_center_pose(image)

    if relative_to is not None:
        # relative_to is the pose of the reference volume
        # Convert the pose to the reference volume coordinate system
        affine = np.linalg.inv(relative_to) @ affine

    center_position = affine[:3, 3]
    direction = affine[:3, :3]

    # Convert the direction matrix to a quaternion
    rotation = Rotation.from_matrix(direction)
    quaternion = rotation.as_quat()

    return center_position, quaternion


class CT3DLabelmapPoseDataset(Dataset):
    def __init__(self, params):
        self.params = params
        self.n_classes = params.n_classes

        self.base_folder_data_imgs = params.base_folder_data_path

        self.total_slices = []
        # Find subfolders matching the pattern "CT*"
        for dir_name, subdirs, _ in os.walk(self.base_folder_data_imgs):
            for subdir in filter(lambda d: d.startswith('CT'), subdirs):
                subdir_path = os.path.join(dir_name, subdir)

                # Read the "whole_volume.nii.gz" using SimpleITK
                volume_path = os.path.join(subdir_path, 'whole_volume.nii.gz')
                if os.path.exists(volume_path):
                    volume = sitk.ReadImage(volume_path)
                    print(f"Read CT volume from: {volume_path}")
                    center_pose = get_center_pose(volume)
                else:
                    print(f"Could not find volume at: {volume_path}")
                    continue

                # List all ".nii.gz" files in the "slices" folder
                slices_dir = os.path.join(subdir_path, 'slices')
                if os.path.exists(slices_dir):
                    sub_slice_dirs = [f for f in os.listdir(slices_dir) if "slices" in f]

                    for dir_ in sub_slice_dirs:
                        abs_ = os.path.join(slices_dir, dir_)
                        slice_files = [f for f in os.listdir(abs_) if f.endswith('.nii.gz')]
                        for s in slice_files:
                            slice_path = os.path.join(abs_, s)
                            self.total_slices.append((slice_path, center_pose, volume))
                break

        # create the ultrasound mask
        origin = (106, 246)
        opening_angle = 70  # in degrees
        short_radius = 124  # in pixels
        long_radius = 512  # in pixels
        img_shape = (SIZE_W, SIZE_H)
        self.us_mask, _, _ = create_ultrasound_mask(origin, opening_angle, short_radius, long_radius, img_shape)

        self.transform_img = transforms.Compose([
            transforms.ToTensor(),
            # transforms.RandomAffine(degrees=(0, 30), translate=(0.2, 0.2), scale=(1.0, 2.0), fill=9),
            transforms.Resize([SIZE_W, SIZE_H], transforms.InterpolationMode.NEAREST),
            # transforms.RandomVerticalFlip()
        ])

    def __len__(self):
        if self.params.debug:
            return self.total_slices.__len__() // 100  # for debugging
        else:
            return self.total_slices.__len__()

    def read_volumes(self, full_labelmap_path):
        slice_indices = []
        volume_indices = []
        total_slices = 0
        volumes = []

        for idx, folder in enumerate(full_labelmap_path):
            labelmap = [lm for lm in sorted(os.listdir(folder)) if lm.endswith('.nii.gz') and "_" not in lm][0]
            vol_nib = nib.load(folder + labelmap)
            vol = vol_nib.get_fdata()
            # vol = vol.transpose(0,2,1)

            slice_indices.extend(np.arange(vol.shape[2]))  # append the vol indexes
            volume_indices.extend(np.full(shape=vol.shape[2], fill_value=idx, dtype=np.int))  # append the vol indexes
            total_slices += vol.shape[2]
            volumes.append(vol)

        return slice_indices, volume_indices, total_slices, volumes

    def preprocess(self, img, mask):
        if mask:
            img = np.where(img != self.params.pred_label, 0, 1)

        return img  # .astype('float64')

    def __getitem__(self, idx):
        volume: sitk.Image = None
        vol_path, ref_center_pose, volume = self.total_slices[idx]
        slice_data = sitk.ReadImage(vol_path)
        slice_data = sitk.GetArrayFromImage(slice_data)
        slice_data = slice_data.transpose(1, 2, 0)
        slice_data = slice_data[:, :, 1]
        labelmap_slice = slice_data.astype('int64')
        labelmap_slice[labelmap_slice == 0] = 9
        labelmap_slice[labelmap_slice == 15] = 4

        # print('vol_nr: ', vol_nr, 'idx: ', idx)
        # labelmap_slice = self.volumes[vol_nr][:, :, self.slice_indices[idx]].astype('int64')        #labelmap input to the US renderer
        # if self.full_labelmap_path_imgs != self.base_folder_data_masks:
        #     mask_slice = self.mask_volumes[vol_nr][:, :, self.slice_indices[idx]].astype('int64')
        # else:
        #     mask_slice = labelmap_slice

        state = torch.get_rng_state()
        labelmap_slice = self.transform_img(labelmap_slice)
        torch.set_rng_state(state)

        # us_mask = transforms.ToTensor()(self.us_mask)
        # us_mask = torch.where(us_mask > 0, 1, 0)

        position, quat = get_image_center_pose_as_quat(vol_path, relative_to=ref_center_pose)  # todo: get the relative pose to the center of the volume

        pose = np.concatenate((position, quat)).astype(np.float32)

        return labelmap_slice, pose, vol_path, sitk.GetArrayFromImage(volume), volume.GetSpacing(), volume.GetDirection(), volume.GetOrigin()


class CT3DLabelmapPoseDataLoader():
    def __init__(self, params):
        super().__init__()
        self.params = params

    def train_dataloader(self):
        full_dataset = CT3DLabelmapPoseDataset(self.params)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size

        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size,
                                                                           val_size])  # , generator=Generator().manual_seed(0))

        train_loader = DataLoader(self.train_dataset, batch_size=self.params.batch_size, shuffle=True,
                                  num_workers=self.params.num_workers)

        return train_loader, self.train_dataset, self.val_dataset

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.params.batch_size, shuffle=False,
                          num_workers=self.params.num_workers)

    # def test_dataloader(self):
    #     test_dataset = CT3DLabelmapsDataset(self.hparams)
    #     # train_dataset = AortaDataset(self.hparams, "val")
    #     return DataLoader(test_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=8)
