import numpy as np
import timm
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from PIL import Image
import monai
from torch import nn

from Losses.losses import SoftDiceLoss, DiceLoss
from utils.helpers import AddGaussianNoise
from models.unet_2d_github import dice_loss
from torch.optim.lr_scheduler import StepLR

# torch.set_printoptions(profile="full")
THRESHOLD = 0.5
SIZE_W = 256
SIZE_H = 256


class PoseRegressionSim(torch.nn.Module):
    def __init__(self, params, inner_model=None):
        super(PoseRegressionSim, self).__init__()
        self.params = params

        # negative_penalty = negative_alpha * torch.sum(torch.nn.functional.relu(-1 * input))
        self.USRenderingModel = inner_model.to(params.device)

        # define transforms for image
        self.rendered_img_masks_random_transf = transforms.Compose([
            transforms.Resize([286, 286], transforms.InterpolationMode.NEAREST),  # todo: check if 286 is ok
            transforms.RandomCrop(256),
        ])

        self.rendered_img_random_transf = transforms.Compose([
            transforms.RandomApply([AddGaussianNoise(0., 0.02)], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(9, 9), sigma=(0.1, 3))], p=0.5),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),    #???
        ])

        if params.seg_net_input_augmentations_noise_blur:  # todo: change the names later
            print(self.rendered_img_random_transf)

        # if params.outer_model_monai:
        #     # channels = (16, 32, 64, 128, 256)
        #     # channels = (64, 128, 256, 512, 1024)
        #     channels = (32, 64, 128, 256, 512)
        #     print('MONAI channels: ', channels, 'DROPOUT: ', params.dropout_ratio)
        #     self.outer_model = monai.networks.nets.UNet(
        #         spatial_dims=2,
        #         in_channels=1,
        #         out_channels=1,
        #         channels=channels,
        #         strides=(2, 2, 2, 2),
        #         num_res_units=2,
        #         dropout = params.dropout_ratio
        #     ).to(params.device)
        #
        #     self.loss_function = monai.losses.DiceLoss(sigmoid=True)
        #
        # elif self.params.outer_model_github:
        #     self.outer_model = outer_model.to(params.device)
        # else:
        #     self.outer_model = outer_model.to(params.device)
        #     self.loss_function = SoftDiceLoss()  # without thresholding is soft DICE loss
        #     # self.loss_function = DiceLoss()     #default threshold is 0.5

        self.outer_model = PoseRegressionNet().to(params.device)

        # translation normalization coeff
        self.translation_normalization_coeff = np.array((SIZE_W, SIZE_H)).mean() / 2

        self.optimizer, self.scheduler = self.configure_optimizers(self.USRenderingModel, self.outer_model)
        # self.optimizer = self.configure_optimizers(self.USRenderingModel, self.outer_model)

        self.loss_function = self.pose_loss

        print(f'OuterModel On cuda?: ', next(self.outer_model.parameters()).is_cuda)
        print('USRenderingModel On cuda?: ', next(self.USRenderingModel.parameters()).is_cuda)

    def geodesic_loss(self, q1, q2):
        # Calculate the dot product between the two quaternions
        dot_product = (q1 * q2).sum(dim=1)
        # Clamp the values to avoid numerical instability
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        # Calculate the angle between the two quaternions
        angle = torch.acos(dot_product)
        return angle.mean()

    def pose_loss(self, predicted_poses, gt_poses):
        normalized_rot = torch.nn.functional.normalize(predicted_poses[:, :4], p=2, dim=1)
        loss_rot = self.geodesic_loss(normalized_rot, gt_poses[:, :4])
        loss_trans = torch.nn.functional.mse_loss(predicted_poses[:, 4:], gt_poses[:, 4:] / self.translation_normalization_coeff)
        loss = loss_rot + loss_trans
        # self.log('train_loss', loss, on_step=True, on_epoch=True)  # todo: add logging later
        return loss

    def normalize(self, img):
        return (img - torch.min(img)) / (torch.max(img) - torch.min(img))

    def create_return_dict(self, type, loss, input, file_name, gt_mask, z_hat_pred, us_sim, epoch):
        dict = {f"{type}_images_unet": (input.detach()),
                'file_name': file_name,
                f'{type}_images_gt': gt_mask.detach(),
                f'{type}_images_pred': z_hat_pred.detach(),
                f'{type}_images_us_sim': us_sim.detach(),
                f'{type}_loss': loss.detach(),
                'epoch': epoch}

        return dict

    def rendering_forward(self, input):
        us_sim = self.USRenderingModel(input.squeeze())
        us_sim_resized = F.resize(us_sim.unsqueeze(0).unsqueeze(0), (SIZE_W, SIZE_H)).float()
        # self.USRenderingModel.plot_fig(us_sim.squeeze(), "us_sim", True)

        return us_sim_resized


    def pose_forward(self, us_sim, pose_gt):
        output = self.outer_model.forward(us_sim)
        loss = self.loss_function(output, pose_gt)
        pred = output

        return loss, pred

    def step(self, input, pose_gt):
        us_sim_resized = self.rendering_forward(input)
        loss, pred = self.pose_forward(self, us_sim_resized, pose_gt)
        # z_norm = self.normalize(z)

        return loss, us_sim_resized, pred

    def get_data(self, batch_data):
        input, pose_gt, file_name = batch_data[0].to(self.params.device), batch_data[1].to(self.params.device), \
        batch_data[2]

        return input, pose_gt, file_name

    def validation_step(self, batch_data, epoch, batch_idx=None):
        # print('IN VALIDATION... ')
        # input, label, file_name = batch_data[0].to(self.params.device), batch_data[1].to(self.params.device), batch_data[2]
        # label = self.label_preprocess(label)

        input, label, file_name = self.get_data(batch_data)

        loss, us_sim_resized, pred = self.step(input, label)

        dict = self.plot_val_results(input, loss, file_name, label, pred, us_sim_resized, epoch)

        return pred, us_sim_resized, loss, dict

    def plot_val_results(self, input, loss, file_name, label, pred, us_sim_resized, epoch):  # todo: this will not work for now
        return {}

        val_images_plot = F.resize(input, (SIZE_W, SIZE_H)).float().unsqueeze(0)
        dict = self.create_return_dict('val', loss, val_images_plot, file_name[0], label, pred, us_sim_resized, epoch)

        return dict

    def configure_optimizers(self, inner_model, outer_model):

        if inner_model is None:
            optimizer = torch.optim.Adam(outer_model.parameters(), self.params.outer_model_learning_rate)
        else:
            # params = list(outer_model.parameters()) + list(inner_model.parameters())
            # optimizer = torch.optim.Adam(params, self.params.learning_rate)

            optimizer = torch.optim.Adam(
                [
                    {"params": outer_model.parameters(), "lr": self.params.outer_model_learning_rate},
                    {"params": inner_model.parameters(), "lr": self.params.inner_model_learning_rate},
                ],
                # lr=self.params.global_learning_rate,
            )

        if self.params.scheduler:
            scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
        else:
            scheduler = None

        return optimizer, scheduler


class PoseRegressionNet(nn.Module):
    def __init__(self):
        super(PoseRegressionNet, self).__init__()

        # Initialize the EfficientNet model
        efficientnet_pretrained = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)

        # Get the weights from the first convolutional layer
        pretrained_weights = efficientnet_pretrained.conv_stem.weight

        # Average the weights across the three input channels
        new_weights = pretrained_weights.mean(dim=1, keepdim=True)

        # Create a new convolutional layer with one input channel
        new_conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

        # Set the weights of the new convolutional layer
        new_conv.weight = nn.Parameter(new_weights)

        # Replace the first convolutional layer in the model
        efficientnet_pretrained.conv_stem = new_conv

        self.efficientnet = efficientnet_pretrained

        # Get the output feature size of EfficientNet-B0
        num_features = self.efficientnet.num_features

        # Fully connected layers for pose regression
        self.fc_layers = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        # Pass input through EfficientNet
        x = self.efficientnet(x)

        # Pass output through fully connected layers
        x = self.fc_layers(x)

        return x
