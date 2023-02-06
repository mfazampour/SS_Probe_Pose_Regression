import os
import torch
from torchvision import transforms
import pytorch_lightning as pl
# from torch.metrics.fun import Accuracy
# from torchmetrics.functional import accuracy
from models.us_rendering import UltrasoundRendering
# import torch.nn.functional as F
import torchvision.transforms.functional as F

from Losses.losses import SoftDiceLoss, DiceLoss
import monai
# torch.set_printoptions(profile="full")
THRESHOLD = 0.5


class Segmentation(pl.LightningModule):
    def __init__(self, params, outer_model, inner_model):
        super(Segmentation, self).__init__()
        # self.automatic_optimization = False
        self.params = params
        # self.UnetModel = outer_model.to(params.device)

        self.UnetModel = monai.networks.nets.UNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            ).to(params.device)

        self.USRenderingModel = inner_model.to(params.device)

        # self.accuracy = Accuracy()
        self.criterion = SoftDiceLoss()  # hparams.device)
        self.criterion_thresh = DiceLoss()#hparams.device)
                # self.current_train_fold = hparams.current_train_fold
        print(f'UnetModel On cuda?: ', next(self.UnetModel.parameters()).is_cuda)
        print('USRenderingModel On cuda?: ', self.USRenderingModel.device.type)

    def normalize(self, img):
        return (img - torch.min(img)) / (torch.max(img) - torch.min(img))

    def create_return_dict(self, type, loss, input, file_name, gt_mask, z_hat_pred, us_sim):
        dict = {f"{type}_images_unet": (input.detach()),
                'file_name': file_name,
                f'{type}_images_gt': gt_mask.detach(),
                f'{type}_images_pred': z_hat_pred.detach(),
                f'{type}_images_us_sim': us_sim.detach(),
                f'{type}_loss': loss.detach(),
                'epoch': self.current_epoch}
        # dict[f'current_{type}_fold'] = current_fold
        # dict['val_acc'] = accuracy

        return dict


    def step(self, input, gt_mask):
        # print('STEPP')
        # UltrasoundRendering().plot_fig(input.squeeze(), "input", False)

        us_sim = self.USRenderingModel(input.squeeze()) 
        # UltrasoundRendering().plot_fig(us_sim, "us_sim", True)
        # us_sim_resized = F.interpolate(us_sim.unsqueeze(0).unsqueeze(0), size=(256,256))

        us_sim_resized = F.resize(us_sim.unsqueeze(0).unsqueeze(0), (256,256))

        z_hat = self.UnetModel(us_sim_resized)
        # UltrasoundRendering().plot_fig(z_hat.squeeze(), "z_hat", True)

        # z_norm = self.normalize(z)
        loss, z_hat_pred = self.criterion(z_hat, gt_mask)

        # accuracy = self.accuracy(z_hat_pred, gt_mask.int())

        return loss, us_sim_resized, z_hat_pred                #{"loss": loss}


    def training_step(self, batch, batch_idx):
        # opt = self.optimizers()
        # opt.zero_grad()
        
        input, gt_mask, file_name = batch   #gives 3D labelmap volume
        print('FILENAME: ' + file_name)

        # mask = F.interpolate(gt_mask.unsqueeze(0).unsqueeze(0), size=(1,256,256), mode="bilinear")
        gt_mask = F.resize(gt_mask, (256,256))

        loss, us_sim, z_hat_pred = self.step(input, gt_mask)

        # self.manual_backward(loss)
        # opt.step()


        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        input = F.resize(input, (256,256))

        dict = self.create_return_dict('train', loss, input, file_name, gt_mask, z_hat_pred, us_sim)

        return {'loss': loss, 'dict': dict}        #loss
    

    def validation_step(self, batch, batch_idx):
        print('IN VALIDATION... ')

        input, gt_mask, file_name = batch   #gives 3D labelmap volume
        gt_mask = F.resize(gt_mask, (256,256))

        loss, us_sim, z_hat_pred = self.step(input, gt_mask)

        self.log('val_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        input = F.resize(input, (256,256))

        dict = self.create_return_dict('val', loss, input, file_name, gt_mask, z_hat_pred, us_sim)
        return dict


    def configure_optimizers(self):
        # optimizer = optim.SGD(self.UnetModel.parameters(), lr=1e-4, momentum=0.9)
        # return optimizer

        params = list(self.USRenderingModel.parameters()) + list(self.UnetModel.parameters())
        # params = list(self.UnetModel.parameters())
        optimizer = torch.optim.Adam(params, lr=self.params.learning_rate)
        print('LR is: ', self.params.learning_rate)
        return optimizer

        # print('##############  ReduceLROnPlateau ########################')
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)
        # return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler,  'monitor': 'val_loss'}

        # LR_DECAY = 0.995
        # DECAY_EVERY_N_EPOCHS = 5
        # N_EPOCHS = 2
        # torch.cuda.manual_seed(0)
        # optimizer = torch.optim.RMSprop(self.UnetModel.parameters(), lr=1e-4, weight_decay=1e-4)



        # scheduler = {'scheduler': lr_scheduler, 'interval': 'epoch', 'monitor': 'metric_to_track'}
        # return [optimizer], [scheduler]

 


    @staticmethod
    def add_module_specific_args(parser): 
        unet_module = parser.add_argument_group(
            title='model  specific args options')
        # tiramisu_module.add_argument("--learning_rate", default=0.001, type=float)
        # tiramisu_module.add_argument("--optimizer_name", default="adam", type=str)
        # unet_module.add_argument("--batch_size", default=1, type=int)
        unet_module.add("--in_channels", default=1, type=int)
        unet_module.add("--out_channels", default=1, type=int)

        return parser

