from pathlib import Path

import pandas as pd
import os
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset, random_split
from torchvision import transforms
import torchio as tio
import kornia.geometry.conversions as conversions
import SimpleITK as sitk
import numpy as np


def get_center_pose(image):
    # Find the physical center of the volume
    size = image.GetSize()
    center_pixel = [sz // 2 for sz in size]
    center_physical = image.TransformContinuousIndexToPhysicalPoint(center_pixel)

    # Extract the rotation matrix from the direction cosines
    rotation_matrix = np.array(image.GetDirection()).reshape((3, 3))

    affine = np.eye(4)
    affine[:3, :3] = rotation_matrix
    affine[:3, 3] = center_physical

    return affine


class PhantomPoseRegressionDataset(Dataset):
    def __init__(self, root_dir, transform=None, img_size=(512, 512), number_of_cts=1, loading_slice=False, debug=False):

        # the root directory
        self.root_dir = root_dir

        # the csv file with name tracking.csv in root_dir
        self.csv_file = os.path.join(self.root_dir, 'tracking.csv')

        # the 2d image directory
        self.us_image_dir = os.path.join(self.root_dir, '2d_images/')

        # the 3d image path
        self.ct_image_path = os.path.join(self.root_dir, 'ct_labelmap.nii')  # todo: check if nii image is fine for reading the transform

        self.poses = pd.read_csv(self.csv_file, sep='\t', header=None)

        self.us_image_dir = os.path.abspath(self.us_image_dir)
        # read all png images in us_image_dir
        self.paths_imgs = [p for p in Path(f'{self.us_image_dir}').glob(f'**/*.png')]
        # sort the paths
        self.paths_imgs = sorted(self.paths_imgs)

        self.transform = transform

        # Read the 3D CT image using sitk
        self.ct_image = sitk.ReadImage(self.ct_image_path)

        # self.ct_image = tio.LabelMap(ct_image_path)
        # self.ct_image.load()

        self.loading_slice = loading_slice
        if loading_slice:
            self.preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(img_size, transforms.InterpolationMode.NEAREST),
            ])
        else:
            self.preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(img_size),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

        self.ct_id = number_of_cts

        self.debug = debug

    def calculate_relative_pose(self, us_pose):
        # Reshape the flattened 4x4 matrix
        us_pose = us_pose.view(4, 4).t()
        # ct_pose = torch.tensor(self.ct_image.affine, dtype=torch.double) # Assuming the CT pose is stored in the affine attribute

        # find the pose of the center of the CT
        ct_center_pose = torch.tensor(get_center_pose(self.ct_image)).double()

        # Compute the relative pose as a matrix multiplication between the inverse CT pose and the US pose
        relative_pose = torch.inverse(ct_center_pose) @ us_pose

        # Extract rotation and translation components
        rotation_matrix = relative_pose[:3, :3]
        translation_vector = relative_pose[:3, 3]

        # Convert the rotation_matrix to a quaternion
        rotation = Rotation.from_matrix(rotation_matrix)
        quaternion = rotation.as_quat()

        # Concatenate the quaternion and translation to form the final pose representation
        final_pose = torch.cat([translation_vector, torch.tensor(quaternion).double()])

        return final_pose.float()

    def __len__(self):
        if self.debug:
            return 10
        else:
            return len(self.poses)

    def __getitem__(self, idx):
        us_image_path = self.paths_imgs[idx]
        if self.loading_slice:
            image_ = Image.open(us_image_path).convert('I')
        else:
            image_ = Image.open(us_image_path).convert('L')  # Load as grayscale
        image_ = self.preprocess(image_)

        if self.loading_slice:
            image_ = image_.permute(0, 2, 1)
            image_[image_ == 0] = 4
            image_[image_ == 15] = 4
            image_ = image_.to(torch.int64)

        us_pose = torch.tensor(self.poses.iloc[idx, :16].values.astype('float'))

        relative_pose = self.calculate_relative_pose(us_pose)

        if self.transform:
            image_ = self.transform(image_)

        return image_, relative_pose, us_image_path.__str__(), 0.0, 0.0, 0.0, 0.0, 0.0, self.ct_id


class PhantomPoseRegressionDataLoader():
    def __init__(self, param, root_dir, batch_size=1, transform=None, num_workers=1):
        super().__init__()
        self.params = param
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.transform = transform
        self.num_workers = num_workers

        self.dataset = PhantomPoseRegressionDataset(root_dir=self.root_dir, transform=self.transform)

    def get_dataloaders(self):

        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])  #, generator=Generator().manual_seed(0))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return train_loader, val_loader
