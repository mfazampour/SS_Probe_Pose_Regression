"""
Adapted from
https://gitlab.lrz.de/CAMP_IFL/covid19_seg/-/blob/master/modules/seg.py
See the monai docs
https://docs.monai.io/en/latest/
"""
import pytorch_lightning as pl
from torch import optim, sigmoid, ge
from pytorch_lightning.metrics import Accuracy
from monai.losses import DiceLoss
from utils.utils import get_argparser_group
import torch

EPS = 1e-5


class Segmentation(pl.LightningModule):
    def __init__(self, hparams, UnetModel, inner_model, criterion):
        super(Segmentation, self).__init__()
        self.hparams = hparams
        self.model = UnetModel
        self.tiramisuModel = inner_model.to(hparams.device)
        self.example_input_array = torch.zeros(1, 1, 196, 196)
        self.accuracy = Accuracy()
        # instead of implementing our own dice loss, use the monai one
        # self.loss_function = DiceLoss()
        self.criterion = criterion

    # 1: Forward step (forward hook), Lightning calls this inside the training loop
    def forward(self, x):
        x = self.model.forward(x)
        return x

    # 2: Optimizer (configure_optimizers hook)
    # see https://pytorch-lightning.readthedocs.io/en/latest/optimizers.html
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    # 4: Training Loop (this is where the magic happens)(training_step hook)
    def training_step(self, train_batch, batch_idx):
        # Note: dyndata returns a dict with keys (['image', 'label', 'pat_id', 'vol_idx'])
        x = train_batch['image']
        # train_batch['label'] is loaded as LongTensor but FloatTensor is required later on for the loss function
        y_true = train_batch['label']
        # forward pass. y_pred contains probabilities per pixel and class / channel
        z = self.UnetModel.forward(x)
        z_hat = self.tiramisuModel.forward(z)

        # set all values in y_pred >= .5 to 1 and all values < .5 to 0 to obtain a clear differentiation between classes
        # for more than two classes use one hot encoding!
        # y_pred_01 = torch.ge(y_pred, 0.5).float()
        # y_pred_dim1 = torch.argmax(y_pred, dim=1)
        # display a few images
        if batch_idx == 0 or batch_idx % 20 == 0:
            # add a little padding / buffer between the three images
            buffer = torch.ones(x.size()[2], 10, 1)
            images = torch.cat((torch.movedim(x[0, :, :, :], 0, 2), buffer, torch.movedim(y_true[:1, 1, :, :], 0, 2),
                                buffer, torch.movedim(torch.argmax(y_pred[:1, :, :, :], dim=1), 0, 2)), dim=1)
            self.logger.experiment[0].add_images(f'Epoch: {self.current_epoch}, Batch: {batch_idx}, '
                                                 f'Vol: {train_batch["vol_idx"][0]}, '
                                                 f'Patient: {train_batch["pat_id"][0]}  \n '
                                                 f'(CT Image - Ground Truth Label - Predicted Label)', images, 0,
                                                 dataformats='HWC')
        # calculate loss
        # train_loss = self.loss_function(y_pred, y_true)
        train_loss = self.criterion(z_hat, y_true)
        return {'loss': train_loss, 'y_true': y_true, 'y_pred': z_hat}

    # If using metrics in data parallel mode (dp), the metric update/logging should be done in the training_step_end
    # This is due to metric states else being destroyed after each forward pass, leading to wrong accumulation.
    def training_step_end(self, train_step_output):
        accuracy = self.accuracy(train_step_output['y_pred'], train_step_output['y_true'])
        self.log('Train DICE Loss', train_step_output['loss'], on_step=True, on_epoch=True)
        self.log('Train Accuracy', accuracy, on_step=False, on_epoch=True)
        return train_step_output

    # 5 Validation Loop
    def validation_step(self, val_batch, batch_idx):
        # x = val_batch['image']
        # y_true = val_batch['label']
        x, y_true = val_batch

        # forward pass
        y_pred = self.forward(x)
        # ge(probs, 0.5) sets all values >= 0.5 to True and all <0 to False. .float() turns True to 1 and False to 0.
        # y_pred = torch.ge(probs, 0.5).float()
        # calculate loss
        val_loss = self.criterion(y_pred[0], y_true)
        return {'val_loss': val_loss, 'y_true': y_true, 'y_pred': y_pred}

    # If using metrics in data parallel mode (dp), the metric update/logging should be done in the validation_step_end
    # This is due to metric states else being destroyed after each forward pass, leading to wrong accumulation.
    def validation_step_end(self, val_step_output):
        accuracy = self.accuracy(val_step_output['y_pred'], val_step_output['y_true'])
        self.log('Validation DICE Loss (monai)', val_step_output['val_loss'], on_step=True, on_epoch=True)
        self.log('Validation Accuracy', accuracy, on_step=False, on_epoch=True)
        return val_step_output

    # 6 Test Loop
    def test_step(self, test_batch, batch_idx):
        x = test_batch['image']
        y_true = test_batch['label']
        # forward pass
        y_pred = self.forward(x)
        # calculate loss
        test_loss = self.criterion(y_pred, y_true)
        return {'test_loss': test_loss, 'y_pred': y_pred, 'y_true': y_true}

    def test_step_end(self, test_step_output):
        self.log('Test DICE Loss', test_step_output['test_loss'], on_step=True, on_epoch=True)
        return test_step_output

    @staticmethod
    def add_module_specific_args(parser):
        specific_args = get_argparser_group(title="Model options", parser=parser)
        # specific_args.add_argument('--input_channels', default=1, type=int)
        # specific_args.add_argument('--model_dim', default=2, type=int)
        # specific_args.add_argument('--model_needs_activation', default=1, type=int)
        # specific_args.add_argument('--unet_up_mode', choices=['upconv', 'upsample'], default='upconv')
        # specific_args.add("--unet_depth", default=5, type=int)
        specific_args.add_argument("--batch_size", default=1, type=int)
        specific_args.add_argument("--learning_rate", default=0.001, type=int)

        return parser

