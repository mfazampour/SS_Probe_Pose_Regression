import os
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torch import optim
from models import tiramisu
import numpy as np
from utils.Elipses import imgs
from torch.utils.data.distributed import DistributedSampler
import logging
import cv2
# from monai.losses import DiceLoss
from pytorch_lightning.metrics import Accuracy
import torch.nn as nn
# from Losses.dice_loss_unet import DiceLoss
from Losses.losses import DiceLoss

import wandb
from Losses.bce_dice_loss import DiceBCELoss

# torch.set_printoptions(profile="full")
THRESHOLD = 0.5


class Segmentation(pl.LightningModule):
    def __init__(self, hparams, model, inner_model):
        # super().__init__()
        super(Segmentation, self).__init__()
        self.hparams = hparams
        self.UnetModel = model.to(hparams.device)
        self.tiramisuModel = inner_model.to(hparams.device)
        # self.tiramisuModel.freeze()   #freeze should be called here

        self.accuracy = Accuracy()
        self.criterion = DiceLoss() #hparams.device)
        # self.loss_function_no_background = DiceLoss(include_background=False)
        # self.criterion = torch.nn.BCEWithLogitsLoss()
        # self.criterion = DiceBCELoss()

        # self.activation = nn.Sigmoid()
        # self.activation = nn.Softmax(dim=2)


        # self.criterion = criterion
        # self.batch_size = hparams.batch_size
        self.current_train_fold = hparams.current_train_fold
        print('UnetModel On cuda?: ', next(self.UnetModel.parameters()).is_cuda)
        print('tiramisuModel On cuda?: ', next(self.tiramisuModel.parameters()).is_cuda)


    def normalize(self, img):
        return (img - torch.min(img)) / (torch.max(img) - torch.min(img))

    def forward(self, input_n):
        z = self.UnetModel(input_n)
        print('AFTER UnetModel.forward. ')
        z_norm = self.normalize(z)

        # idx = z_norm[0, 0, :, :]
        # print('min: ', torch.min(idx), ' max: ', torch.max(idx), 'idx: ', idx)

        # with torch.no_grad():         #only for improving performance?
        z_hat = self.tiramisuModel(z_norm)
        print('AFTER tiramisuModel.forward. ')

        # idx = z_hat[0, 0, :, :]
        # print('min: ', torch.min(idx), ' max: ', torch.max(idx), 'idx: ', idx)

        # z_hat = z_hat.view(1,1,256*256)
        # z_hat_probs = self.activation(z_hat)
        # print('min: ', torch.min(z_hat_probs), ' max: ', torch.max(z_hat_probs), 'idx: ', z_hat_probs)

        # idx_p = z_hat_probs[0, 0, :]
        # print('min: ', torch.min(idx_p), ' max: ', torch.max(idx_p), 'idx: ', idx_p)

        # print('X min: ', torch.min(z_hat), ' max: ', torch.max(z_hat))

        return z_hat

    def step(self, batch, batch_idx):
        print('STEPP')

        input, gt_mask, file_name = batch
        print('filename: ', file_name)
        # self.current_train_fold = (file_name[0][0].rsplit("/", 1)[1]).split("_")[0]
        input_n = self.normalize(input)

        z_hat = self.forward(input_n)
        # z_hat = torch.sigmoid(z_hat)
        loss = self.criterion(z_hat, gt_mask)
        print('AFTER loss: ')

        # self.log("train_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        # result = pl.TrainResult(loss, checkpoint_on=loss)
        return loss, gt_mask, z_hat                 #{"loss": loss}

    def training_step(self, batch, batch_idx):
        for b in batch:
            loss, gt_mask, z_hat = self.step(b, batch_idx)
            print('TRAINING STEP ')
            # self.log_dict({f"train_{k}": v for k, v in logs.items()})
            # self.log("train_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
            self.log('metric_to_track', loss)
            # self.log(self.current_train_fold + "_" + "train_loss", loss)

            z_hat_pred = torch.sigmoid(z_hat)
            accuracy = self.accuracy(z_hat_pred, gt_mask.int())
            print('TRAINING STEP END ')


############################# LOG WEIGHTS ###############################################
        # for name, param in self.UnetModel.named_parameters():
        #     # pl_module.logger.experiment.summary[name] = param.data
        #     self.logger.experiment.log({"Unet." + name: wandb.Histogram(param.data.numpy())})
        #
        # for name, param in self.tiramisuModel.named_parameters():
        #     # pl_module.logger.experiment.summary[name] = param.data
        #     self.logger.experiment.log({"Tiramisu." + name: wandb.Histogram(param.data.numpy())})
##########################################################################################

        return {'loss': loss, 'accuracy': accuracy}         #loss

    # def training_step_end(self, train_step_output):
    #return {'loss': train_step_output['loss']}

    def training_epoch_end(self, outputs):
        print('-------------TRAINING EPOCH ENDED--------------')

        # print('training_epoch_end: ', outputs)
        train_loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        train_acc_mean = torch.stack([x['accuracy'] for x in outputs]).mean()
        self.log(''.join(self.current_train_fold) + "_" + "train_loss", train_loss_mean, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        self.logger.experiment.log({self.current_train_fold + "_" + "train_loss": train_loss_mean,
                                    self.current_train_fold + "_" + "train_acc": train_acc_mean})

        # dict = {f"train_loss": train_loss_mean, 'current_train_fold': self.current_train_fold}
        # return {"train_loss": train_loss_mean}

    def test_step(self, batch, batch_idx, dataloader_idx):

        input, gt_mask, file_name = batch
        # print(' min: ', torch.min(input), ' max: ', torch.max(input))
        input_n = self.normalize(input)
        # print(' min: ', torch.min(input_n), ' max: ', torch.max(input_n))

        current_test_fold = self.test_dataloader()[dataloader_idx].dataset.imgs_dir.rsplit("/", 3)[1]
        print(current_test_fold)

        z = self.UnetModel(input_n)
        # print(' min: ', torch.min(z), ' max: ', torch.max(z))

        z_norm = self.normalize(z)
        # print(' min: ', torch.min(z_norm), ' max: ', torch.max(z_norm))

        z_hat = self.tiramisuModel(z_norm)
        loss = self.criterion(z_hat, gt_mask)
        # print(' min: ', torch.min(z_hat), ' max: ', torch.max(z_hat))

        dict = {f"test_images_unet": (input_n, z_norm)}
        dict['file_name'] = file_name
        dict['current_test_fold'] = current_test_fold
        print('current_test_fold: ', current_test_fold)

        z_hat = torch.sigmoid(z_hat)    #just for the visualization
        dict['test_images_end'] = (gt_mask, z_hat)
        # z_hat_thresh = torch.ge(z_hat, THRESHOLD).float()       # calc the loss on tht thresholded prediction
        # loss_thresholded = self.criterion(z_hat_thresh, gt_mask)
        dict['test_loss'] = loss
        # dict['test_loss_thresh'] = loss_thresholded

        return dict

    def test_epoch_end(self, outputs):

        for test_fold_dict in outputs:
            # for test_fold_dict in dataloader_outputs:
            test_fold_loss_stack = torch.stack([x['test_loss'] for x in test_fold_dict])
            # test_fold_loss_thresh_stack = torch.stack([x['test_loss_thresh'] for x in test_fold_dict])
            test_fold = test_fold_dict[0]['current_test_fold']

            test_fold_loss_mean = test_fold_loss_stack.mean()
            test_fold_loss_std = test_fold_loss_stack.std()
            print('test_fold_loss_mean: ', test_fold_loss_mean)
            print('test_fold_loss_std: ', test_fold_loss_std)
            self.logger.experiment.log({"train_" + self.current_train_fold + "_" + "test_loss_mean_" + test_fold: test_fold_loss_mean,
                                        # "train_" + self.current_train_fold + "_" + "test_loss_thresh_mean_" + test_fold: test_fold_loss_thresh_stack.mean(),
                                        "train_" + self.current_train_fold + "_" + "test_loss_std_" + test_fold: test_fold_loss_std})

        # self.log("test_loss", test_loss_mean, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        # self.log("test_loss_std", test_loss_std, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        # return {'test_loss': test_loss_mean.item(), 'test_loss_std': test_loss_std.item()}

    def validation_step(self, batch, batch_idx, dataset_idx):
        print('IN VALIDATION... ')
        print('dataset_idx: ', dataset_idx)

        input, gt_mask, file_name = batch
        input_n = self.normalize(input)

        # imin=torch.min(input_n);    imax=torch.max(input_n) ;     istd=torch.std(input_n);   ivar = torch.var(input_n)

        z = self.UnetModel.forward(input_n)
        z_norm = self.normalize(z)

        z_hat = self.tiramisuModel.forward(z_norm)

        dict = {f"val_images_unet": (input, z_norm)}
        dict['file_name'] = file_name
        dict['val_images_end'] = (gt_mask, z_hat)

        # self.log_dict({f"val_{k}": v for k, v in logs.items()})
        loss = self.criterion(z_hat, gt_mask)
        dict['val_loss'] = loss

        # self.log("val_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)

        return dict

    # def validation_epoch_end(self, outputs):
    def validation_end(self, outputs):

        print('validation_epoch_end: ', outputs)
        # With multiple dataloaders, outputs will be a list of lists
        # val_loss_mean = 0
        # i = 0
        # for dataloader_outputs in outputs:
        #     for output in dataloader_outputs:
        #         val_loss_mean += output['val_loss']
        #         i += 1
        #
        # val_loss_mean /= i
        # print("val_loss:", val_loss_mean.item())
        # self.log("val_loss", val_loss_mean.item(), prog_bar=True, logger=True, on_step=False, on_epoch=True)
        # self.logger.experiment.log({"val_loss": val_loss_mean.item()})

        #########################################
        ######## another way:
        # for dataloader_outputs in outputs:
        #     for val_fold in dataloader_outputs:
        #         val_loss_stack = torch.stack([x['val_loss'] for x in val_fold])
        #
        # print('val_loss_mean: ', val_loss_stack.mean())
        # print('val_loss_std: ', val_loss_stack.std())
        #
        # self.log("val_loss", val_loss_mean, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        #
        # # self.logger.experiment.log({"val_loss": val_loss_mean})
        # return {'val_loss': val_loss_mean.item()}



    # def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
    #     # import pdb; pdb.set_trace()
    #     print('val outputs: {}'.format(outputs))

    # def validation_epoch_end(self, outputs):
    #     """
    #     Called at the end of validation to aggregate outputs
    #     :param outputs: list of individual outputs of each validation step
    #     :return:
    #     """
    #     # val_acc_epoch = torch.stack([o["val_acc"] for o in outputs]).mean()
    #     #
    #     # if val_acc_epoch > self.max_acc["acc"]:
    #     #     self.max_acc["acc"] = val_acc_epoch
    #     #     self.max_acc["epoch"] = self.current_epoch
    #     #
    #     # self.log("val: max acc", self.max_acc["acc"])
    #     # print (outputs)
    #     # outputs, step="val")


    def configure_optimizers(self):
        # LR = 1e-4
        # optimizer = optim.SGD(self.UnetModel.parameters(), lr=0.00001, momentum=0.9)
        # return optimizer

        LR = 1e-4
        LR_DECAY = 0.995
        DECAY_EVERY_N_EPOCHS = 5
        N_EPOCHS = 2
        torch.cuda.manual_seed(0)
        optimizer = torch.optim.RMSprop(self.UnetModel.parameters(), lr=1e-4, weight_decay=1e-4)

        # return optimizer
        # optimizer = torch.optim.Adam(self.UnetModel.parameters(), lr=1e-3)
        # optimizer = torch.optim.Adam(self.UnetModel.parameters(), lr=1e-4)

        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)
        # return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler,  'monitor': 'metric_to_track'},

        scheduler = {'scheduler': lr_scheduler, 'interval': 'epoch', 'monitor': 'metric_to_track'}
        return [optimizer], [scheduler]


    @staticmethod
    def add_module_specific_args(parser):  # pragma: no cover
        unet_module = parser.add_argument_group(
            title='tiramisu_module  specific args options')
        # tiramisu_module.add_argument("--learning_rate",
        #                            default=0.001,
        #                            type=float)
        # tiramisu_module.add_argument("--optimizer_name",
        #                            default="adam",
        #                            type=str)
        # unet_module.add_argument("--batch_size", default=1, type=int)
        unet_module.add("--in_channels", default=1, type=int)
        unet_module.add("--out_channels", default=1, type=int)

        return parser


    # def train_dataloader(self):
    #     dataloader = self.__dataloader(split="train")
    #     logging.info("training data loader called - size: {}".format(
    #         len(dataloader.dataset)))
    #     return dataloader
    #
    # def val_dataloader(self):
    #     dataloader = self.__dataloader(split="val")
    #     logging.info("validation data loader called - size: {}".format(
    #         len(dataloader.dataset)))
    #     return dataloader
    #
    # def test_dataloader(self):
    #     dataloader = self.__dataloader(split="test")
    #     logging.info("test data loader called  - size: {}".format(
    #         len(dataloader.dataset)))
    #     return dataloader
