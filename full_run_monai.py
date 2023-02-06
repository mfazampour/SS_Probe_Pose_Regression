# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import tempfile
from glob import glob

import torch
import wandb
from tensorboardX import SummaryWriter

from PIL import Image
# from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import ArrayDataset, create_test_image_2d, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImage,
    RandRotate90,
    RandSpatialCrop,
    ScaleIntensity,
)
from monai.visualize import plot_2d_or_3d_image
import configargparse
from utils.configargparse_arguments import build_configargparser
from datasets.ct_3d_labemaps_dataset_torch import CT3DLabelmapsDataLoader
import torchvision.transforms.functional as F
from utils.plotter_torch import Plotter
from models.us_rendering_model_torch import UltrasoundRendering

wandb.init(project="cactuss_end2end-Segmentation")

tb_logger = SummaryWriter('log_dir/cactuss_end2end')
    
def log_gradients_in_model(model, step):
    for tag, value in model.named_parameters():
        if value.grad is not None:
            tb_logger.add_histogram(tag + "/grad", value.grad.cpu(), step)

def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('-c', is_config_file=True, help='config file path')
    parser, hparams = build_configargparser(parser)



    dataloader = CT3DLabelmapsDataLoader(hparams)
    train_loader, train_dataset, val_dataset  = dataloader.train_dataloader()
    val_loader = dataloader.val_dataloader()
    
    plotter = Plotter()

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outer_model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    # wandb.watch(outer_model, log_freq=100)

    inner_model = UltrasoundRendering()
    wandb.watch(inner_model, log='all', log_graph=True, log_freq=10)
    # wandb.watch([inner_model, outer_model], log='all', log_graph=True, log_freq=100)


    loss_function = monai.losses.DiceLoss(sigmoid=True)

    # params = list(outer_model.parameters()) + list(inner_model.parameters())
    # optimizer = torch.optim.Adam(params, hparams.learning_rate)
    optimizer = torch.optim.Adam(outer_model.parameters(), hparams.learning_rate)

    # start a typical PyTorch training
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    # epoch_loss_values = list()
    metric_values = list()
    # writer = SummaryWriter()
    for epoch in range(10):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{10}")
        outer_model.train()
        inner_model.train()
        epoch_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            labels = F.resize(labels, (128,128)).float().unsqueeze(0)

            optimizer.zero_grad()

            us_sim = inner_model(inputs.squeeze()) 
            us_sim_resized = F.resize(us_sim.unsqueeze(0).unsqueeze(0), (128,128)).float()
            # us_sim_resized.register_hook(lambda grad: print(grad))

            outputs = outer_model(us_sim_resized)
            
            loss = loss_function(outputs, labels)
            # outputs.register_hook(lambda grad: print(grad))
            loss.backward()
            log_gradients_in_model(inner_model, step)

            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_dataset) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            wandb.log({"train_loss_step": loss.item()})
            # writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

        epoch_loss /= len(train_loader)
        # epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        wandb.log({"train_loss_epoch": epoch_loss})


        if (epoch + 1) % val_interval == 0:
            outer_model.eval()
            inner_model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_data in val_loader:
                    val_images, val_labels, file_name = val_data[0].to(device), val_data[1].to(device), val_data[2]
                    val_labels = F.resize(val_labels, (128,128)).float().unsqueeze(0)

                    us_sim_val = inner_model(val_images.squeeze()) 
                    us_sim_val_resized = F.resize(us_sim_val.unsqueeze(0).unsqueeze(0), (128,128)).float()

                    pred = outer_model(us_sim_val_resized)
                    
                    val_loss += loss_function(pred, val_labels)

                    val_images_plot = F.resize(val_images, (128,128)).float().unsqueeze(0)
                    dict = plotter.create_return_dict('val', val_loss, val_images_plot, file_name[0], val_labels, pred, us_sim_val_resized, epoch)
                    plotter.validation_batch_end(dict)


                val_loss /= len(val_loader)
                wandb.log({"val_loss_epoch": val_loss})
                plotter.validation_epoch_end()
                print(f"--------------- END VAL ------------")
                 

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    # writer.close()


if __name__ == "__main__":
    main()










    # create a temporary directory and 40 random image, mask pairs
    # print(f"generating synthetic data to {tempdir} (this may take a while)")
    # for i in range(40):
    #     im, seg = create_test_image_2d(128, 128, num_seg_classes=1)
    #     Image.fromarray((im * 255).astype("uint8")).save(os.path.join(tempdir, f"img{i:d}.png"))
    #     Image.fromarray((seg * 255).astype("uint8")).save(os.path.join(tempdir, f"seg{i:d}.png"))

    # images = sorted(glob(os.path.join(tempdir, "img*.png")))
    # segs = sorted(glob(os.path.join(tempdir, "seg*.png")))

    # define transforms for image and segmentation
    # train_imtrans = Compose(
    #     [
    #         LoadImage(image_only=True, ensure_channel_first=True),
    #         ScaleIntensity(),
    #         RandSpatialCrop((96, 96), random_size=False),
    #         RandRotate90(prob=0.5, spatial_axes=(0, 1)),
    #     ]
    # )
    # train_segtrans = Compose(
    #     [
    #         LoadImage(image_only=True, ensure_channel_first=True),
    #         ScaleIntensity(),
    #         RandSpatialCrop((96, 96), random_size=False),
    #         RandRotate90(prob=0.5, spatial_axes=(0, 1)),
    #     ]
    # )
    # val_imtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity()])
    # val_segtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity()])

    # # define array dataset, data loader
    # check_ds = ArrayDataset(images, train_imtrans, segs, train_segtrans)
    # check_loader = DataLoader(check_ds, batch_size=10, num_workers=2, pin_memory=torch.cuda.is_available())
    # im, seg = monai.utils.misc.first(check_loader)
    # print(im.shape, seg.shape)

    # # create a training data loader
    # train_ds = ArrayDataset(images[:20], train_imtrans, segs[:20], train_segtrans)
    # train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())
    # # create a validation data loader
    # val_ds = ArrayDataset(images[-20:], val_imtrans, segs[-20:], val_segtrans)
    # val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available())







#    roi_size = (96, 96)
#                     sw_batch_size = 4
#                     val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, outer_model)
#                     val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
#                     # compute metric for current iteration
#                     dice_metric(y_pred=val_outputs, y=val_labels)
#                 # aggregate the final mean dice result
#                 metric = dice_metric.aggregate().item()
#                 # reset the status for next validation round
#                 dice_metric.reset()
#                 metric_values.append(metric)
#                 if metric > best_metric:
#                     best_metric = metric
#                     best_metric_epoch = epoch + 1
#                     torch.save(model.state_dict(), "best_metric_model_segmentation2d_array.pth")
#                     print("saved new best metric model")
#                 print(
#                     "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
#                         epoch + 1, metric, best_metric, best_metric_epoch
#                     )
#                 )
#                 # writer.add_scalar("val_mean_dice", metric, epoch + 1)
#                 # plot the last model output as GIF image in TensorBoard with the corresponding image and label
#                 plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
#                 plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
#                 plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")