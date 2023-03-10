from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset, ConcatDataset
import logging
from PIL import Image
import matplotlib.pyplot as plt
SIZE_W = 256
SIZE_H = 256


class AortaDataset(Dataset):
    def __init__(self, hparams, type):
        self.device = hparams.device
        self.n_classes = hparams.n_classes
        print('hparams.cactuss_data_dir: ', hparams.cactuss_data_dir)

        dir_root = hparams.cactuss_data_dir + "/" + type + "/"     # type is the name of the subfolder containing the imgs
        print('dir_root:', dir_root)
        self.imgs_dir = dir_root + "imgs/"
        self.masks_dir = dir_root + "masks/"

        self.resize = True
        self.scale = 1
        self.mask_suffix = ''
        assert 0 < self.scale <= 1, 'Scale must be between 0 and 1'
        # self.imgs_dir = glob(self.imgs_dir + "*/")
        # temp  = []
        # for element in self.imgs_dir:
        #     if "@eaDir" not in element:
        #         temp.append(element)
        # self.imgs_dir = temp
        # print(self.imgs_dir)
        self.ids = [splitext(file)[0] for file in listdir(self.imgs_dir)
                    if not (file.startswith('.') or file.startswith('@'))]

        # self.imgs_ids, self.masks_ids = ([] for i in range(2))
        # img_type = ["imgs/", "masks/"]
        # for fold in datasets:
        #     dir = hparams.data_root + "/" + fold
        #     [self.imgs_ids.append(dir + img_type[0] + file) for file in listdir(dir + img_type[0])]
        #     [self.masks_ids.append(dir + img_type[1] + file) for file in listdir(dir + img_type[1])]

        # logging.info(f'Creating dataset with {len(self.imgs_ids)} examples')
        print(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)    #/ 2

    def add_dataset_specific_args(parser):
        parser.add_argument("--testing_this", type=float, default=1)

    @classmethod
    def change_size(cls, pil_img, resize, scale):
        if resize:
            pil_img = pil_img.resize((SIZE_W, SIZE_H))
        w, h = pil_img.size
        # newW, newH = int(scale * w), int(scale * h)
        # assert newW > 0 and newH > 0, 'Scale is too small'
        # pil_img = pil_img.resize((newW, newH))
        return pil_img

    @classmethod
    def preprocess(cls, pil_img, mask):
        if not mask:
            pil_img = pil_img.convert('L')

        img_nd = np.array(pil_img)
        if mask:
            # img_nd = np.where(img_nd == img_nd.min(), 1, 0)
            # img_nd = np.where(img_nd == 23, 255, img_nd)
            # img_nd = np.where(img_nd == 229, 255, img_nd)
            img_nd = np.where(img_nd == 2, 0, img_nd)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        # img = Image.open(self.imgs_ids[i])
        # mask = Image.open(self.masks_ids[i])

        idx = self.ids[i]
        img_file = glob(self.imgs_dir + idx + '.*')
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')

        file_name_img = img_file[0].rsplit("/", 1)[1]
        file_name_masks = mask_file[0].rsplit("/", 1)[1]
        # print(f'imgs_id: ', file_name_img, 'masks_id: ', file_name_masks)
        assert file_name_img == file_name_masks, \
            f'Image and mask NOT SAME'

        img = Image.open(img_file[0])
        mask = Image.open(mask_file[0])
        # print(f'imgs_name: ', img_file[0], 'masks_name: ', mask_file[0])

        img = self.change_size(img, self.resize, self.scale)
        mask = self.change_size(mask, self.resize, self.scale)
        # print(f'Img size: {img.size}, mask size: {mask.size}')

        assert img.size == mask.size, \
            f'Image and mask {i} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, mask=False)
        # print(f'Img min: {img.min()}, max: {img.max()}')

        mask = self.preprocess(mask, mask=True)
        # print(f'Mask min: {mask.min()}, max: {maAsk.max()}')

        # plt.imshow(mask, aspect="auto")
        # plt.show()

        return torch.from_numpy(img).type(torch.FloatTensor).to(self.device), \
               torch.from_numpy(mask).type(torch.FloatTensor).to(self.device), \
               img_file[0]
               # self.imgs_ids[i], self.masks_ids[i]

    def add_dataset_specific_args(parser):  # pragma: no cover
        specific_args = parser.add_argument_group(title='elipses database specific args options')
        #     specific_args.add_argument("--n_classes", default=1, type=int)
        specific_args.add("--dimensions", default=1, type=int)
        return parser
