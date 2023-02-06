import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import nibabel as nib
import cv2
import pytorch_lightning as pl
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
        self.device = params.device
        self.n_classes = params.n_classes

        self.base_folder_data = params.base_folder_data_path
        self.labelmap_path = params.labelmap_path

        self.sub_folder_CT = [sub_f for sub_f in sorted(os.listdir(self.base_folder_data))]
        self.full_labelmap_path = [self.base_folder_data + s + self.labelmap_path for s in self.sub_folder_CT]

        # # self.volumes = []
        # self.volume_dict = {}
        # self.total_slices = 0
        # for folder in self.full_labelmap_path:
        #     labelmap = [lm for lm in sorted(os.listdir(folder)) if lm.endswith('.nii.gz') and "_" not in lm][0]
        #     vol_nib = nib.load(folder + labelmap)
        #     vol = vol_nib.get_fdata()

        #     # indices = np.arange(vol.shape[2])
        #     self.total_slices += vol.shape[2]
        #     vol_name = labelmap.split('.')[0]
        #     self.volume_dict[vol_name] = vol

        # self.volume_names = list(self.volume_dict.keys())
        
        # # print('self.volumes: ', self.volumes)
        # print('self.volume_names: ', self.volume_names)

        # self.slice_indices = {}
        self.slice_indices = []
        # self.volume_indices = {}
        self.volume_indices = []
        self.total_slices = 0
        self.volumes = []


        for idx, folder in enumerate(self.full_labelmap_path):
            labelmap = [lm for lm in sorted(os.listdir(folder)) if lm.endswith('.nii.gz') and "_" not in lm][0]
            vol_nib = nib.load(folder + labelmap)
            vol = vol_nib.get_fdata()

            # self.slice_indices[idx] = np.arange(vol.shape[2])
            # self.volume_indices[idx] = np.full((vol.shape[2],), idx)
            # self.volume_indices += np.full((vol.shape[2],), idx)   
            self.slice_indices.extend(np.arange(vol.shape[2]))  #append the vol indexes
            self.volume_indices.extend(np.full(shape=vol.shape[2], fill_value=idx, dtype=np.int))  #append the vol indexes
            self.total_slices += vol.shape[2]
            self.volumes.append(vol)

        # print('self.slice_indices: ', self.slice_indices)
        # x=np.array(x)
        # self.idx_arr = np.arange(0, self.total_slices)

    def __len__(self):
        return self.total_slices


    @classmethod
    def preprocess(cls, img, mask):
        # if not mask:
        #     pil_img = pil_img.convert('L')

        # img_nd = np.array(pil_img)
        if mask:
            img = np.where(img != 4, 0, 1)

        # if len(img_nd.shape) == 2:
        #     img_nd = np.expand_dims(img_nd, axis=2)

        # # HWC to CHW
        # img_trans = img_nd.transpose((2, 0, 1))
        # if img_trans.max() > 1:
        #     img_trans = img_trans / 255

        return img      #.astype('float64')

    
    def __getitem__(self, idx):

        # volume_idx = self.volume_indices[idx // len(self.slice_indices[idx % len(self.volumes)])]
        # slice_idx = self.slice_indices[volume_idx][idx % len(self.slice_indices[volume_idx])]
        # vol = self.volumes[volume_idx]

        vol_nr = self.volume_indices[idx]
        # print('vol_nr: ', vol_nr, 'idx: ', idx)
        slice = self.volumes[vol_nr][:, :, self.slice_indices[idx]].astype('int64')


        # resized_slice = cv2.resize(slice, (SIZE_W, SIZE_H), cv2.INTER_LINEAR_EXACT)
        mask = self.preprocess(slice, mask=True)

        return slice, mask, str(vol_nr) + '_' + str(self.slice_indices[idx])

        # slice = torch.from_numpy(slice)
        # return slice    #, slice_idx, volume_idx





        vol_name = self.volume_names[idx]
        vol = self.volume_dict[vol_name]
        slice_idx = np.random.randint(vol.shape[2])
        slice = vol[:, :, slice_idx]
        resized_slice = cv2.resize(slice, (SIZE_W, SIZE_H), cv2.INTER_LINEAR)

        mask = self.preprocess(resized_slice, mask=True)

        return resized_slice, mask, vol_name

# torch.from_numpy(ct_3d_labelmap).type(torch.FloatTensor).to(self.device)



class CT3DLabelmapsModule(pl.LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def train_dataloader(self):
        full_dataset = CT3DLabelmapsDataset(self.params)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size

        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])#, generator=Generator().manual_seed(0))

        return DataLoader(self.train_dataset, batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.num_workers)

    # def test_dataloader(self):
    #     test_dataset = CT3DLabelmapsDataset(self.hparams)
    #     # train_dataset = AortaDataset(self.hparams, "val")
    #     return DataLoader(test_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=8)



# dataset = CT3DLabelmapsDataset(volumes)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# for data, name in dataloader:
#     # data is a randomly selected 2D slice from one of the volumes
#     # name is the name of the volume the slice came from
#     pass


