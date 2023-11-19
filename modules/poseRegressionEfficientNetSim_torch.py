import numpy as np
import timm
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from PIL import Image
import monai
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from skimage.transform import resize
from kornia.filters.median import MedianBlur
from torch import nn
from utils.helpers import AddGaussianNoise
from models.unet_2d_github import dice_loss
from torch.optim.lr_scheduler import StepLR
from kornia.geometry.conversions import euler_from_quaternion

from datasets.imfusion_free_slicing.slice_volume import interpolate_arbitrary_plane
from datasets.imfusion_free_slicing.ultrasound_fan import create_ultrasound_mask
from datasets.real_us_pose_dataset_with_gt_torch import get_center_pose
from models.us_rendering_model_torch import warp_img2
from Losses.pytorch_similarity.torch_similarity.modules.gradient_correlation_loss import GradientCorrelationLoss2d, \
    GradientCorrelation2d

# torch.set_printoptions(profile="full")
THRESHOLD = 0.5
SIZE_W = 256
SIZE_H = 256

# Constants
ORIGIN = (106, 246)
OPENING_ANGLE = 70  # in degrees
SHORT_RADIUS = 124  # in pixels
LONG_RADIUS = 512  # in pixels
IMG_SHAPE = (512, 512)
US_SPACING = (0.4, 0.4)  # in mm
SLICED_IMG_SHAPE = (int(IMG_SHAPE[0] * 1.2 * US_SPACING[0]), int(IMG_SHAPE[1] * 1.2 * US_SPACING[1]))


def lcc_loss(I, J, window_size=5):
    assert I.shape == J.shape, "Input images must have the same shape"

    # Define the local window
    window = torch.ones((1, 1, window_size, window_size)) / (window_size ** 2)
    window = window.to(I.device)

    # Compute local means of I and J
    I_mean = torch.nn.functional.conv2d(I, window, padding=window_size // 2)
    J_mean = torch.nn.functional.conv2d(J, window, padding=window_size // 2)

    # Compute cross-correlation numerator
    cross_corr = torch.nn.functional.conv2d(I * J, window, padding=window_size // 2) - (I_mean * J_mean)

    # Compute the denominators
    I_var = torch.nn.functional.conv2d(I * I, window, padding=window_size // 2) - (I_mean * I_mean)
    J_var = torch.nn.functional.conv2d(J * J, window, padding=window_size // 2) - (J_mean * J_mean)

    lcc = cross_corr / (torch.sqrt((I_var * J_var).abs()) + 1e-5)

    # Here, we're taking the negative mean to treat it as a loss (the closer lcc is to 1, the better)
    return -torch.mean(lcc)


def create_return_dict(type, loss, input, file_name, gt_mask, z_hat_pred, us_sim, epoch):
    dict = {f"{type}_images_unet": (input.detach()),
            'file_name': file_name,
            f'{type}_images_gt': gt_mask.detach(),
            f'{type}_images_pred': z_hat_pred.detach(),
            f'{type}_images_us_sim': us_sim.detach(),
            f'{type}_loss': loss.detach(),
            'epoch': epoch}

    return dict


def plot_val_results(input, loss, file_name, label, pred, us_sim_resized,
                     epoch):  # todo: this will not work for now
    val_images_plot = F.resize(input, (SIZE_W, SIZE_H)).float().unsqueeze(0)
    dict = create_return_dict('val', loss, val_images_plot, file_name[0], label, pred, us_sim_resized, epoch)

    return dict


def geodesic_loss(q1, q2):

    # theta = cos^{-1}(2 <q_1,q_2>^2 -1)
    # https://math.stackexchange.com/questions/90081/quaternion-distance
    # Or approximated by:
    # d(q1,q2) = 1 - <q1,q2>^2

    # Calculate the dot product between the two quaternions
    dot_product = (q1 * q2).sum(dim=1)

    # Clamp the values to avoid numerical instability
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    distance = 1 - dot_product ** 2
    return distance.mean()  # todo check this loss

    # Calculate the angle between the two quaternions
    # angle = torch.acos(2 * (dot_product ** 2) - 1)
    # return angle.mean()  # todo check this loss


def slice_volume_from_pose(pose, ct_volume: sitk.Image, us_sim, spacing: tuple,
                           plot: bool = False, origina_volume=False) -> torch.Tensor:
    """
    Slice the volume from the pose and return the slice.

    Args:
        origina_volume:
        pose: The pose information.
        ct_volume: The CT volume to slice.
        us_sim: Used for plotting.
        spacing: Spacing information.
        plot: If True, plot the slice.

    Returns:
        torch.Tensor: The warped slice from the volume.
    """

    ct_center = get_center_pose(image=ct_volume)

    normalized_rot = torch.nn.functional.normalize(pose[:, -4:].detach().cpu(), p=2, dim=1)
    r = Rotation.from_quat(normalized_rot)
    pose_mat = np.eye(4)
    pose_mat[:3, :3] = r.as_matrix()
    pose_mat[:3, 3] = pose[0, :3].detach().cpu()

    center_world_pose = ct_center @ pose_mat
    center_position = center_world_pose[:3, 3]

    # get the normal vectors of the image plane
    normal_vector = center_world_pose[2, :3]
    down_vector = center_world_pose[1, :3]
    right_vector = center_world_pose[0, :3]

    # get the top left corner of the image from the center position
    top_left_corner = center_position - (SLICED_IMG_SHAPE[1] / 2) * spacing[0] * right_vector - (
            SLICED_IMG_SHAPE[0] / 2) * spacing[1] * down_vector

    sliced = interpolate_arbitrary_plane(top_left_corner, slice_normal=-1 * normal_vector, volume=ct_volume,
                                         down_direction=-1 * down_vector, slice_shape=SLICED_IMG_SHAPE)
    sliced = sitk.GetArrayFromImage(sliced)

    # average over the z axis
    sliced = sliced.transpose(1, 2, 0)[:, :, 1]
    sliced = resize(sliced, IMG_SHAPE, preserve_range=True, order=0)[:, ::-1]

    if not origina_volume:
        sliced[sliced == 0] = 9
        sliced[sliced == 15] = 4

    sliced_t = torch.tensor(sliced.copy()).float().to(pose.device)
    sliced_t = torch.rot90(sliced_t, 1)
    warped = warp_img2(sliced_t)
    warped = torch.rot90(warped, 3)

    if plot:
        # plot both the sliced and the us_sim image in matplotlib
        plt.imshow(sliced, cmap='gray')
        plt.show()
        plt.imshow(us_sim.detach().squeeze().cpu(), cmap='gray')
        plt.show()
        plt.imshow(warped.detach().squeeze().cpu(), cmap='gray')
        plt.show()

    return warped


class PoseRegressionSim(torch.nn.Module):
    def __init__(self, params, inner_model=None, number_of_cts=1):
        super(PoseRegressionSim, self).__init__()
        self.params = params

        # negative_penalty = negative_alpha * torch.sum(torch.nn.functional.relu(-1 * input))
        self.USRenderingModel = inner_model.to(params.device)

        # define transforms for image
        self.rendered_img_masks_random_transf = transforms.Compose([
            transforms.Resize([286, 286], transforms.InterpolationMode.NEAREST),
            transforms.RandomCrop(256),
        ])

        self.rendered_img_random_transf = transforms.Compose([
            transforms.RandomApply([AddGaussianNoise(0., 0.02)], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(9, 9), sigma=(0.1, 3))], p=0.5),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),    #???
        ])

        if params.net_input_augmentations_noise_blur:
            print(self.rendered_img_random_transf)

        self.outer_model = PoseRegressionNet(number_of_cts=number_of_cts).to(params.device)

        self.gradient_correlation_loss = GradientCorrelationLoss2d(gauss_sigma=None).to(params.device)
        self.gradient_correlation = GradientCorrelation2d().to(params.device)
        self.median_blur = MedianBlur(9)

        # translation normalization coeff
        self.translation_normalization_coeff = np.array((SIZE_W, SIZE_H)).mean() / 20

        self.optimizer, self.scheduler = self.configure_optimizers(self.USRenderingModel, self.outer_model)
        # self.optimizer = self.configure_optimizers(self.USRenderingModel, self.outer_model)

        # self.loss_function = self.pose_loss

        print(f'OuterModel On cuda?: ', next(self.outer_model.parameters()).is_cuda)
        print('USRenderingModel On cuda?: ', next(self.USRenderingModel.parameters()).is_cuda)

    def pose_loss(self, predicted_poses, gt_poses):
        norm_ = torch.norm(predicted_poses[:, -4:], p=2, dim=1)
        loss_norm = torch.pow(norm_, 2).mean()
        normalized_rot = predicted_poses[:, -4:] / norm_.detach()  #torch.nn.functional.normalize(predicted_poses[:, -4:], p=2, dim=1)
        loss_rot = geodesic_loss(normalized_rot, gt_poses[:, -4:])

        # euler angles loss
        predicted_euler = euler_from_quaternion(x=normalized_rot[0, 0], y=normalized_rot[0, 1], z=normalized_rot[0, 2], w=normalized_rot[0, 3])
        gt_euler = euler_from_quaternion(x=gt_poses[0, -4], y=gt_poses[0, -3], z=gt_poses[0, -2], w=gt_poses[0, -1])
        [loss_yaw, loss_pitch, loss_roll] = (torch.tensor(predicted_euler) - torch.tensor(gt_euler)).abs()

        loss_trans = torch.nn.functional.mse_loss(predicted_poses[:, :3],
                                                  gt_poses[:, :3]) / self.translation_normalization_coeff ** 2
        losses = {'loss_rot': loss_rot, 'loss_trans': loss_trans, 'loss_norm': loss_norm,
                   'loss_yaw': loss_yaw, 'loss_pitch': loss_pitch, 'loss_roll': loss_roll}
        return losses

    def normalize(self, img):
        return (img - torch.min(img)) / (torch.max(img) - torch.min(img))

    def rendering_forward(self, input):
        us_sim = self.USRenderingModel(input.squeeze())
        us_sim_resized = F.resize(us_sim.unsqueeze(0).unsqueeze(0), (SIZE_W, SIZE_H)).float()
        # self.USRenderingModel.plot_fig(us_sim.squeeze(), "us_sim", True)

        return us_sim_resized

    def initiate_pose_net_for_phantom_test(self):

        # 1. Accumulate the weights
        accumulated_weights = {}
        for model in self.outer_model.fc_layers[:-1]:
            for name, param in model.named_parameters():
                if name not in accumulated_weights:
                    accumulated_weights[name] = param.data.clone()
                else:
                    accumulated_weights[name] += param.data

        # 2. Compute the average
        num_models = len(self.outer_model.fc_layers) - 1
        for name in accumulated_weights:
            accumulated_weights[name] /= num_models

        # 3. Initialize a new model with the averaged weights
        for name, param in self.outer_model.fc_layers[-1].named_parameters():
            param.data.copy_(accumulated_weights[name])

    def pose_forward(self, us_sim, pose_gt, slice_volume=False, volume_data=None, spacing=None, direction=None,
                     origin=None, us_sim_orig=None, ct_data=None, ct_id=0):
        output = self.outer_model.forward(us_sim, ct_id=ct_id)
        losses = self.pose_loss(output, pose_gt)

        slice_volume = False  # todo: remove this to enable volume slicing
        if slice_volume:
            ct_volume = sitk.GetImageFromArray(ct_data.squeeze())
            spacing = [s.numpy().item() for s in spacing]
            ct_volume.SetSpacing(spacing)
            direction = [d.numpy().item() for d in direction]
            ct_volume.SetDirection(direction)
            origin = [o.numpy().item() for o in origin]
            ct_volume.SetOrigin(origin)

            sliced = slice_volume_from_pose(pose_gt, ct_volume, us_sim_orig, spacing, origina_volume=True)
            lcc_gt = lcc_loss(us_sim.detach(), sliced.view_as(us_sim))
            # gc_gt = self.gradient_correlation_loss(self.median_blur(us_sim), sliced.view_as(us_sim))
            gc_gt = self.gradient_correlation_loss(self.median_blur(us_sim_orig), self.median_blur(sliced.view_as(us_sim)).detach())

            sliced = slice_volume_from_pose(output.detach(), ct_volume, us_sim_orig, spacing, origina_volume=True)
            lcc_pred = lcc_loss(us_sim.detach(), sliced.view_as(us_sim))
            gc_pred = self.gradient_correlation_loss(self.median_blur(us_sim), self.median_blur(sliced.view_as(us_sim)).detach())
            # the backward only affects the simulation model and not the pose model
            # since the slicing is not differentiable, and we need to detach the pose model output
            lcc_ratio = lcc_pred / lcc_gt
            gc_ratio = gc_pred.detach() / gc_gt

            sum_loss = (losses['loss_rot'] + losses['loss_trans']) * 10.0 + losses['loss_norm'] + gc_pred + gc_gt
            # slicing_losses = {'lcc_gt': lcc_gt, 'lcc_pred': lcc_pred, 'gc_gt': gc_gt, 'gc_pred': gc_pred,
            #                   'lcc_ratio': lcc_ratio, 'gc_ratio': gc_ratio, 'sum_loss': sum_loss}
            # losses.update(slicing_losses)
            losses.update({'sum_loss': sum_loss})
        else:
            sum_loss = (losses['loss_rot'] + losses['loss_trans']) * 10.0 + losses['loss_norm']
            losses.update({'sum_loss': sum_loss})

        pred = output

        return losses, pred

    # def step(self, input, pose_gt):
    #     us_sim_resized = self.rendering_forward(input)
    #     loss, pred = self.pose_forward(self, us_sim_resized, pose_gt)
    #     # z_norm = self.normalize(z)
    #
    #     return loss, us_sim_resized, pred

    def get_data(self, batch_data):
        input, pose_gt, file_name, volume, spacing, direction, origin, ct, ct_id = batch_data[0].to(self.params.device), \
        batch_data[1].to(self.params.device), \
            batch_data[2], batch_data[3], batch_data[4], batch_data[5], batch_data[6], batch_data[7], batch_data[8]

        return input, pose_gt, file_name, volume, spacing, direction, origin, ct, ct_id

    # def validation_step(self, batch_data, epoch, batch_idx=None):
    #     # print('IN VALIDATION... ')
    #     # input, label, file_name = batch_data[0].to(self.params.device), batch_data[1].to(self.params.device), batch_data[2]
    #     # label = self.label_preprocess(label)
    #
    #     input, label, filename, volume, spacing, direction, origin, ct, ct_id = self.get_data(batch_data)
    #
    #     us_sim_resized = self.rendering_forward(input)
    #     losses, pred = self.pose_forward(self, us_sim_resized, label)
    #     loss = losses['sum_loss']
    #     # z_norm = self.normalize(z)
    #     # loss, us_sim_resized, pred = self.step(input, label)
    #
    #     dict = plot_val_results(input, loss, filename, label, pred, us_sim_resized, epoch)
    #
    #     return pred, us_sim_resized, loss, dict

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
    def __init__(self, number_of_cts=1):
        super(PoseRegressionNet, self).__init__()

        # Initialize the EfficientNet model
        efficientnet_pretrained = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)

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
        self.fc_layers = nn.ModuleList()
        for i in range(number_of_cts):
            self.fc_layers.append(nn.Sequential(
                nn.Linear(num_features, 512, bias=False),
                nn.ReLU(),
                nn.Linear(512, 256, bias=False),
                nn.ReLU(),
                nn.Linear(256, 7, bias=False)
            ))

    def forward(self, x, ct_id=0):
        # Pass input through EfficientNet
        x = self.efficientnet(x)

        # Pass output through fully connected layers
        x = self.fc_layers[ct_id](x)

        return x
