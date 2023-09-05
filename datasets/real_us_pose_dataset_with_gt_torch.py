from pathlib import Path

import pandas as pd
import os
import torch
from PIL import Image
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


class PoseRegressionDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        # the root directory
        self.root_dir = root_dir

        # the csv file with name tracking.csv in root_dir
        self.csv_file = os.path.join(self.root_dir, 'tracking.csv')

        # the 2d image directory
        self.us_image_dir = os.path.join(self.root_dir, '2d_images/')

        # the 3d image path
        self.ct_image_path = os.path.join(self.root_dir, 'ct_seg.mhd')

        self.poses = pd.read_csv(self.csv_file, sep='\t')

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

        # Preprocessing for grayscale images (resize not included as it will depend on your specific needs)  # todo: resize
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])

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

        # Convert the rotation matrix to a quaternion
        quaternion = conversions.rotation_matrix_to_quaternion(rotation_matrix)

        # Concatenate the quaternion and translation to form the final pose representation
        final_pose = torch.cat([quaternion, translation_vector])

        return final_pose.float()

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        us_image_path = self.paths_imgs[idx]
        us_image = Image.open(us_image_path).convert('L')  # Load as grayscale
        us_image = self.preprocess(us_image)

        us_pose = torch.tensor(self.poses.iloc[idx, :16].values.astype('float'))

        relative_pose = self.calculate_relative_pose(us_pose)

        if self.transform:
            us_image = self.transform(us_image)

        return us_image, relative_pose


class RealUSPoseRegressionDataLoader():
    def __init__(self, param, root_dir, batch_size=1, transform=None, num_workers=1):
        super().__init__()
        self.params = param
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.transform = transform
        self.num_workers = num_workers

    def get_dataloaders(self):
        dataset = PoseRegressionDataset(root_dir=self.root_dir, transform=self.transform)
       
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])  #, generator=Generator().manual_seed(0))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return train_loader, val_loader
