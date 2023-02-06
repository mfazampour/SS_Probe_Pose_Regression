import numpy as np
from torch.utils.data import Dataset, DataLoader

import os
import nibabel as nib
import cv2
# import pytorch_lightning as pl
from torch.utils.data import random_split
# from PIL import Image


SIZE_W = 256
SIZE_H = 256




''' This data loader is designed to work with a dictionary of volumes,
where the keys are the volume names and the values are the 3D volume data 
represented as numpy arrays. The data loader returns a randomly selected 
2D slice from each volume, along with the name of the volume the slice came from. '''

class CT3DLabelmapsDataset(Dataset):
    def __init__(self, params):
        # self.device = params.device
        self.n_classes = params.n_classes

        self.base_folder_data = params.base_folder_data_path
        self.labelmap_path = params.labelmap_path

        self.sub_folder_CT = [sub_f for sub_f in sorted(os.listdir(self.base_folder_data))]
        self.full_labelmap_path = [self.base_folder_data + s + self.labelmap_path for s in self.sub_folder_CT]

        self.slice_indices = []
        # self.volume_indices = {}
        self.volume_indices = []
        self.total_slices = 0
        self.volumes = []


        for idx, folder in enumerate(self.full_labelmap_path):
            labelmap = [lm for lm in sorted(os.listdir(folder)) if lm.endswith('.nii.gz') and "_" not in lm][0]
            vol_nib = nib.load(folder + labelmap)
            vol = vol_nib.get_fdata()

            self.slice_indices.extend(np.arange(vol.shape[2]))  #append the vol indexes
            self.volume_indices.extend(np.full(shape=vol.shape[2], fill_value=idx, dtype=np.int))  #append the vol indexes
            self.total_slices += vol.shape[2]
            self.volumes.append(vol)


    def __len__(self):
        return self.total_slices


    @classmethod
    def preprocess(cls, img, mask):
        if mask:
            img = np.where(img != 4, 0, 1)

        return img      #.astype('float64')

    
    def __getitem__(self, idx):

        vol_nr = self.volume_indices[idx]
        # print('vol_nr: ', vol_nr, 'idx: ', idx)
        slice = self.volumes[vol_nr][:, :, self.slice_indices[idx]].astype('int64')

        # resized_slice = cv2.resize(slice, (SIZE_W, SIZE_H), cv2.INTER_LINEAR_EXACT)
        mask = self.preprocess(slice, mask=True)

        return slice, mask, str(vol_nr) + '_' + str(self.slice_indices[idx])


class CT3DLabelmapsDataLoader():
    def __init__(self, params):
        super().__init__()
        self.params = params

    def train_dataloader(self):
        full_dataset = CT3DLabelmapsDataset(self.params)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size

        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])#, generator=Generator().manual_seed(0))

        train_loader = DataLoader(self.train_dataset, batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_workers)
        
        return train_loader, self.train_dataset, self.val_dataset 

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.num_workers)

    # def test_dataloader(self):
    #     test_dataset = CT3DLabelmapsDataset(self.hparams)
    #     # train_dataset = AortaDataset(self.hparams, "val")
    #     return DataLoader(test_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=8)
