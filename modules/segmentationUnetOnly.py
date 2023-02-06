import os
import torch
import torchvision
import torch.nn.functional as f
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import logging
# from pytorch_lightning.metrics import Accuracy
from torchmetrics import Accuracy
# from Losses.dice_loss_unet import DiceLoss
import numpy as np
from ..Losses.losses import SoftDiceLoss, DiceLoss

# from Losses.bce_dice_loss import DiceBCELoss

# torch.set_printoptions(profile="full")
THRESHOLD = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Segmentation(pl.LightningModule):
    def __init__(self, hparams, model, inner_model):
        # super().__init__()
        super(Segmentation, self).__init__()
        self.hparams = hparams
        self.UnetModel = model.to(device)
        self.accuracy = Accuracy()
        self.criterion = SoftDiceLoss()  # hparams.device)
        self.criterion_thresh = DiceLoss()#hparams.device)
        # self.current_train_fold = hparams.current_train_fold
        print('UnetModel On cuda?: ', next(self.UnetModel.parameters()).is_cuda)

    def normalize(self, img):
        return (img - torch.min(img)) / (torch.max(img) - torch.min(img))

    def create_return_dict(self, type, loss, input, file_name, gt_mask, z_hat_pred, z_hat_pred_thresh):
        dict = {f"{type}_images_unet": (input.detach()),
                'file_name': file_name,
                f'{type}_images_gt': gt_mask.detach(),
                f'{type}_images_pred': z_hat_pred.detach(),
                f'{type}_images_pred_thresh': z_hat_pred_thresh.detach(),
                f'{type}_loss': loss.detach(),
                'epoch': self.current_epoch}
        # dict[f'current_{type}_fold'] = current_fold
        # dict['val_acc'] = accuracy

        return dict

    def forward(self, input):
        # print('STEPP')
        input_n = self.normalize(input)
        z_hat = self.UnetModel(input_n)

        return z_hat

    def step(self, input, gt_mask):
        z_hat = self.forward(input)
        # z_hat = self.activation(z)
        loss, z_hat_pred = self.criterion(z_hat, gt_mask)
        # print('AFTER loss: ')

        z_hat_pred_thresh = torch.ge(z_hat_pred.data, 0.9).float()

        # z_hat_pred = torch.sigmoid(z_hat)
        accuracy = self.accuracy(z_hat_pred, gt_mask.int())

        return loss, z_hat_pred, z_hat_pred_thresh, accuracy  # {"loss": loss}

    def step_thresh(self, input, gt_mask):
        z_hat = self.forward(input)
        loss, z_hat_pred_thresh = self.criterion_thresh(z_hat, gt_mask, threshold=0.9)

        # z_hat_pred = torch.sigmoid(z_hat)
        # accuracy = self.accuracy(z_hat_pred_thresh, gt_mask.int())

        return loss, z_hat_pred_thresh

    def training_step(self, batch, batch_idx):
        print('TRAINING STEP ')
        input, gt_mask, file_name = batch
        print('filename: ', file_name[0])

        loss, z_hat_pred, z_hat_pred_thresh, accuracy = self.step(input, gt_mask)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True,
                 on_epoch=True)
        dict = self.create_return_dict('train', loss, input, file_name, gt_mask, z_hat_pred, z_hat_pred_thresh)

        return {'loss': loss, 'accuracy': accuracy, 'dict': dict}  # loss

    def validation_step(self, batch, batch_idx):
        print('IN VALIDATION... ')
        input, gt_mask, file_name = batch
        # print(f'filename: {file_name}')
        loss, z_hat_pred, z_hat_pred_thresh, accuracy = self.step(input, gt_mask)
        self.log('val_loss', loss, prog_bar=True, logger=True, on_step=True,
                 on_epoch=True)

        dict = self.create_return_dict('val', loss, input, file_name, gt_mask, z_hat_pred, z_hat_pred_thresh)
        return dict

    def test_step(self, batch, batch_idx):
        print('IN TEST... ')
        input, gt_mask, file_name = batch
        # print(f'filename: {file_name}')
        loss, z_hat_pred, z_hat_pred_thresh, accuracy = self.step(input, gt_mask)
        self.log('test_loss', loss, prog_bar=True, logger=True, on_step=True,
                 on_epoch=True)

        dict = self.create_return_dict('test', loss, input, file_name, gt_mask, z_hat_pred, z_hat_pred_thresh)
        return dict

        # print('filename: ', file_name)
        # current_val_fold = file_name[0].rsplit("/", 3)[1]
        #
        # loss, z_hat_pred, accuracy = self.step(input, gt_mask)
        # dict = {f"val_images_unet": (input, input)}
        # dict['file_name'] = file_name
        # dict['current_val_fold'] = current_val_fold
        # print('current_val_fold: ', current_val_fold)
        #
        # # z_hat = torch.sigmoid(z_hat)    #just for the visualization
        # dict['val_images_pred'] = (gt_mask, z_hat_pred)
        #
        # # z_hat_thresh = (z_hat > THRESHOLD).float()  # calc the loss on the thresholded prediction
        # # loss_thresholded = self.criterion(z_hat_thresh, gt_mask)
        # dict['val_loss'] = loss
        # dict['epoch'] = self.current_epoch
        #
        # dict['val_acc'] = accuracy
        # # dict['test_loss_thresh'] = loss_thresholded

    def configure_optimizers(self):
        # optimizer = optim.SGD(self.UnetModel.parameters(), lr=1e-4, momentum=0.9)
        # return optimizer

        # optimizer = torch.optim.Adam(self.UnetModel.parameters(), lr=1e-4)
        optimizer = torch.optim.Adam(self.UnetModel.parameters(), lr=self.hparams.learning_rate)
        print('LR is: ', self.hparams.learning_rate)
        # return optimizer

        print('##############  ReduceLROnPlateau ########################')
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler,  'monitor': 'val_loss'}

        # LR_DECAY = 0.995
        # DECAY_EVERY_N_EPOCHS = 5
        # N_EPOCHS = 2
        # torch.cuda.manual_seed(0)
        # optimizer = torch.optim.RMSprop(self.UnetModel.parameters(), lr=1e-4, weight_decay=1e-4)



        # scheduler = {'scheduler': lr_scheduler, 'interval': 'epoch', 'monitor': 'metric_to_track'}
        # return [optimizer], [scheduler]



    # def training_epoch_end(self, outputs):
    #     print('-------------TRAINING EPOCH ENDED--------------')
    #
    #     # print('training_epoch_end: ', outputs)
    #     train_loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
    #     train_acc_mean = torch.stack([x['accuracy'] for x in outputs]).mean()
    #
    #     # self.log(''.join(self.current_train_fold) + "_" + "train_loss", train_loss_mean, prog_bar=True, logger=True, on_step=False, on_epoch=True)
    #
    #     # self.log("train_loss", train_loss_mean, prog_bar=True, logger=True, on_step=False, on_epoch=True)
    #     # self.logger.experiment.log({"train_loss": train_loss_mean, "train_acc": train_acc_mean})
    #
    #     self.logger.experiment.log({self.current_train_fold + "_" + "train_loss": train_loss_mean,
    #                                 self.current_train_fold + "_" + "train_acc": train_acc_mean})
    #
    #     # dict = {f"train_loss": train_loss_mean, 'current_train_fold': self.current_train_fold}
    #     # return {"train_loss": train_loss_mean}

    # # def validation_end(self, outputs):
    # def validation_epoch_end(self, outputs):
    #     val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     val_loss_std = torch.stack([x['val_loss'] for x in outputs]).std()
    #     # val_acc_stack = torch.stack([x['val_acc'] for x in outputs])
    #
    #     self.log("val_loss", val_loss_mean, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
    #     self.logger.experiment.log(
    #             {"train_" + self.current_train_fold + "_" + "val_loss_mean": val_loss_mean,
    #              "train_" + self.current_train_fold + "_" + "val_loss_std": val_loss_std})
    #
    #     # val_loss_mean = []
    #     # val_loss_std = []
    #     # # for val_fold_dict in outputs:
    #     #     # for test_fold_dict in dataloader_outputs:
    #     #     val_fold_loss_stack = torch.stack([x['val_loss'] for x in val_fold_dict])
    #     #     # val_fold_acc_stack = torch.stack([x['val_acc'] for x in val_fold_dict])
    #     #     # test_fold_loss_thresh_stack = torch.stack([x['test_loss_thresh'] for x in test_fold_dict])
    #     #     val_fold = outputs[0]['current_val_fold']
    #     #
    #     #     val_fold_loss_mean = val_fold_loss_stack.mean()
    #     #     val_fold_loss_std = val_fold_loss_stack.std()
    #     #     val_loss_mean.append(val_fold_loss_mean)
    #     #     val_loss_std.append(val_fold_loss_std)
    #     #     print('val_fold_loss_mean: ', val_fold_loss_mean)
    #     #     print('val_fold_loss_std: ', val_fold_loss_std)
    #     #     self.logger.experiment.log(
    #     #         {"train_" + self.current_train_fold + "_" + "val_loss_mean_" + val_fold: val_loss_mean,
    #     #          "train_" + self.current_train_fold + "_" + "val_loss_std_" + val_fold: val_loss_std})
    #     #
    #     # val_loss_mean = torch.mean(torch.stack(val_loss_mean))
    #     # val_loss_std = torch.mean(torch.stack(val_loss_std))
    #     # self.log("val_loss", val_loss_mean, prog_bar=True, logger=True, on_step=False, on_epoch=True)
    #     # self.log("val_std", val_loss_std, prog_bar=True, logger=True, on_step=False, on_epoch=True)


    @staticmethod
    def add_module_specific_args(parser):  # pragma: no cover
        unet_module = parser.add_argument_group(
            title='module  specific args options')
        # tiramisu_module.add_argument("--learning_rate",
        #                            default=0.001,
        #                            type=float)
        # tiramisu_module.add_argument("--optimizer_name",
        #                            default="adam",
        #                            type=str)
        # unet_module.add_argument("--batch_size", default=64, type=int)
        unet_module.add("--in_channels", default=1, type=int)
        unet_module.add("--out_channels", default=1, type=int)

        return parser

    # def training_step_end(self, train_step_output):
    #
    #     for name, param in self.UnetModel.named_parameters():
    #         # pl_module.logger.experiment.summary[name] = param.data
    #         self.logger.experiment.log({"Unet." + name: wandb.Histogram(param.data.numpy())})
    #
    #     for name, param in self.tiramisuModel.named_parameters():
    #         # pl_module.logger.experiment.summary[name] = param.data
    #         self.logger.experiment.log({"Tiramisu." + name: wandb.Histogram(param.data.numpy())})
    #
    #     # return {'loss': train_step_output['loss']}

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
