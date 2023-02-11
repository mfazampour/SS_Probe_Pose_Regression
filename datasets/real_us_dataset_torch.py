from torch.utils.data import Dataset, DataLoader
import torch
import os
from PIL import Image
# from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import random_split


class RealUSDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg') or f.endswith('.png')]
        print("len(self.image_files): ", len(self.image_files))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(image_path).convert('L')

        if self.transform:
            image = self.transform(image)

        return image


class RealUSDataLoader():
    def __init__(self, param, root_dir, batch_size=1, transform=None, num_workers=1):
        super().__init__()
        self.params = param
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.transform = transform
        self.num_workers = num_workers

    def get_dataloaders(self):
        dataset = RealUSDataset(root_dir=self.root_dir, transform=self.transform)
       
        # Define the indices for the training and validation datasets
        # dataset_size = len(dataset)
        # indices = list(range(dataset_size))
        # split = int(0.8 * dataset_size)
        # train_indices, val_indices = indices[:split], indices[split:]

        # Define the samplers for the training and validation datasets
        # train_sampler = SubsetRandomSampler(train_indices)
        # val_sampler = SubsetRandomSampler(val_indices)

        # Create the training and validation data loaders
        # train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, sampler=train_sampler, shuffle=True, num_workers=self.num_workers)
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])#, generator=Generator().manual_seed(0))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


        return train_loader, val_loader