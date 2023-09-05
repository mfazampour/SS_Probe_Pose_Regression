import random
import sys
import time

import configargparse
import numpy as np
import torch
import torchvision
import wandb
from monai.metrics import HausdorffDistanceMetric
from scipy.stats import wasserstein_distance
from torch.nn.functional import kl_div
from torch.utils.data import random_split
from tqdm.auto import tqdm, trange

from cut.data import create_dataset as cut_create_dataset
from cut.data.base_dataset import get_transform as cut_get_transform
from cut.models import create_model as cut_create_model
from cut.options.cactussend2end_options import CACTUSSEnd2EndOptions
from cut.util.visualizer import Visualizer as CUTVisualizer
from full_run_torch import training_loop_pose_net
from models.us_rendering_model_torch import UltrasoundRendering
from utils.configargparse_arguments_torch import build_configargparser
from utils.early_stopping import EarlyStopping
from utils.plotter_torch import Plotter
from utils.utils import argparse_summary, get_class_by_path


class CUTTrainer():
    def __init__(self, opt_cut, cut_model, inner_model, dataset_real_us, real_us_train_loader):
        super(CUTTrainer, self).__init__()
        self.opt_cut = opt_cut
        self.t_data = 0
        self.total_iter = 0
        self.optimize_time = 0.1
        self.cut_model = cut_model
        self.inner_model = inner_model
        self.dataset_real_us = dataset_real_us
        self.random_state = None
        self.real_us_train_loader = real_us_train_loader
        self.init_cut = True
        self.cut_plot_figs = []
        self.data_cut_real_us = []

    def cut_optimize_parameters(self, module, dice_loss):
        # update D
        self.cut_model.set_requires_grad(self.cut_model.netD, True)
        self.cut_model.optimizer_D.zero_grad()
        self.cut_model.loss_D = self.cut_model.compute_D_loss()
        self.cut_model.loss_D.backward()
        self.cut_model.optimizer_D.step()

        # update G
        self.cut_model.set_requires_grad(self.cut_model.netD, False)
        self.cut_model.optimizer_G.zero_grad()
        if self.cut_model.opt.netF == 'mlp_sample':
            self.cut_model.optimizer_F.zero_grad()
        self.cut_model.loss_G = self.cut_model.compute_G_loss()

        # self.cut_model.set_requires_grad(module.USRenderingModel, False)
        self.cut_model.loss_G.backward()
        # self.cut_model.set_requires_grad(module.USRenderingModel, True)
        # loss = self.cut_model.loss_G #+ dice_loss
        # loss.backward()

    def cut_optimizer_step(self):
        self.cut_model.optimizer_G.step()
        if self.cut_model.opt.netF == 'mlp_sample':
            self.cut_model.optimizer_F.step()

    def cut_transform(self, us_sim):
        transform_img = cut_get_transform(self.opt_cut, grayscale=False, convert=True, us_sim_flip=False)
        cut_img_transformed = transform_img(us_sim)
        return cut_img_transformed

    def inference_transformatons(self):
        return cut_get_transform(self.opt_cut, grayscale=False, convert=False, us_sim_flip=False, eval=True)

    def forward_cut_A(self, us_real):
        # torch.set_rng_state(self.random_state)
        no_random_transform = self.inference_transformatons()  # Resize and Normalize -1,1
        data_cut_reconstructed_us = no_random_transform(us_real)
        self.data_cut_real_us['A'] = data_cut_reconstructed_us  # add cut_model.real_B
        self.cut_model.set_input(self.data_cut_real_us)  # unpack data from dataset and apply preprocessing
        self.cut_model.forward()

    def forward_cut_B(self, us_sim):

        # torch.set_rng_state(self.random_state)
        no_random_transform = self.inference_transformatons()
        data_cut_rendered_us = no_random_transform(us_sim)
        self.data_cut_real_us['B'] = data_cut_rendered_us  # add cut_model.real_B
        self.cut_model.set_input(self.data_cut_real_us)  # unpack data from dataset and apply preprocessing
        self.cut_model.forward()

    # def forward_cut_B(self, us_sim, label):

    #     torch.set_rng_state(self.random_state)
    #     forward_cut_random_transform = self.inference_transformatons()  #cut_get_transform(self.opt_cut, grayscale=False, convert=False, us_sim_flip=False, eval=False) 
    #     data_cut_rendered_us = forward_cut_random_transform(us_sim)
    #     self.data_cut_real_us['B'] = data_cut_rendered_us   # add cut_model.real_B
    #     self.cut_model.set_input(self.data_cut_real_us)  # unpack data from dataset and apply preprocessing
    #     self.cut_model.forward()

    #     torch.set_rng_state(self.random_state)
    #     label = forward_cut_random_transform(label)
    #     label = (label / 2 ) + 0.5 # from [-1,1] to [0,1]

    #     return label

    def train_cut(self, module, us_sim_resized, epoch, dataloader_real_us_iterator, iter_data_time, epoch_iter):
        # with torch.autograd.detect_anomaly():

        # ct_slice = batch_data_ct[0].to(hparams.device)
        # us_sim = self.inner_model(ct_slice.squeeze()) 
        # us_sim_resized = F.resize(us_sim.unsqueeze(0).unsqueeze(0), (256, 256)).float().detach()    # just getting the image, no gradients

        try:
            self.data_cut_real_us = next(dataloader_real_us_iterator)
        except StopIteration:
            dataloader_real_us_iterator = iter(self.real_us_train_loader)
            self.data_cut_real_us = next(dataloader_real_us_iterator)

        self.cut_transform = cut_get_transform(self.opt_cut, grayscale=False, convert=False, us_sim_flip=True)
        self.random_state = torch.get_rng_state()
        self.data_cut_real_us["A"] = self.cut_transform(self.data_cut_real_us["A"]).to(hparams.device)
        # self.data_cut_real_us["A"] = self.data_cut_real_us["A"].to(hparams.device)

        real_us_batch_size = self.data_cut_real_us["A"].size(0)
        self.total_iter += real_us_batch_size
        epoch_iter += real_us_batch_size

        # transform_B = self.cut_transform    #cut_get_transform(opt_cut, grayscale=False, convert=False, us_sim_flip=True)    #it is already grayscale and tensor
        torch.set_rng_state(self.random_state)
        data_cut_rendered_us = self.cut_transform(us_sim_resized)
        # data_cut_rendered_us = data_cut_rendered_us.unsqueeze(0)
        self.data_cut_real_us['B'] = data_cut_rendered_us  # add cut_model.real_B

        optimize_start_time = time.time()
        # if epoch == opt_cut.epoch_count and self.init_cut:    #initialize only on epoch 1 
        if self.init_cut:  # initialize cut
            print(f"--------------- INIT CUT --------------")
            # self.data_cut_real_us['B'] = data_cut_rendered_us.detach()
            self.init_cut = False
            self.cut_model.data_dependent_initialize(self.data_cut_real_us)
            self.cut_model.setup(self.opt_cut)  # regular setup: load and print networks; create schedulers
            self.cut_model.parallelize()

        self.cut_model.set_input(self.data_cut_real_us)  # unpack data from dataset and apply preprocessing
        # forward
        self.cut_model.forward()

        self.cut_optimize_parameters(module, None)
        self.cut_optimizer_step()

        # self.cut_model.optimize_parameters()   # calculate loss functions, get gradients, update network weights, backward()
        # calculate loss functions, get gradients, update network weights, backward()
        self.optimize_time = (
                                     time.time() - optimize_start_time) / real_us_batch_size * 0.005 + 0.995 * self.optimize_time

        # self.cut_model.compute_visuals()
        # cut_imgs = self.cut_model.get_current_visuals()
        # idt_B = self.cut_model.idt_B

        iter_start_time = time.time()
        if self.total_iter % self.opt_cut.print_freq == 0:
            self.t_data = iter_start_time - iter_data_time

        # Print CUT results
        if self.total_iter % self.opt_cut.display_freq == 0:  # display images and losses on wandb
            self.cut_model.compute_visuals()
            self.cut_plot_figs.append(
                plotter.plot_images(self.cut_model.get_current_visuals(), epoch, wandb, plot_single=False))
            losses = self.cut_model.get_current_losses()
            self.opt_cut.visualizer.print_current_losses(epoch, epoch_iter, losses, self.optimize_time, self.t_data,
                                                         wandb)


# LOAD MODULE
def load_module(module):
    # ------------------------
    # LOAD MODULE
    # ------------------------
    module_path = f"modules.{hparams.module}"
    ModuleClass = get_class_by_path(module_path)

    return ModuleClass


def load_model(model):
    # ------------------------
    # LOAD MODEL
    # ------------------------
    model_path = f"models.{model}"
    ModelClass = get_class_by_path(model_path)

    return ModelClass


# LOAD DATASET
def load_dataset(dataloader):
    # dataset_path = f"datasets.{hparams.dataset}"
    # DatasetClass = get_class_by_path(dataset_path)

    dataset_path = f"datasets.{dataloader}"
    DataLoader = get_class_by_path(dataset_path)
    # parser = DatasetClass.add_dataset_specific_args(parser)
    return DataLoader


# def calc_kl_div(img_nr, epoch):
#     # Normalize the images to have a probability distribution over pixel values
#     img1 = img1 / torch.sum(img1)
#     img2 = img2 / torch.sum(img2)
#
#     # Compute the KL divergence between the two images
#     kl_divergence = kl_div(img1.flatten(), img2.flatten())
#
#     print("KL divergence between the two images: ", kl_divergence.item())
#     if hparams.logging: wandb.log({"kl_divergence_epoch_" + str(epoch): kl_divergence.item()})
#
#     return kl_divergence.item()


def add_online_augmentations(hparams, input, label, module):
    if hparams.net_input_augmentations_noise_blur:
        input = module.rendered_img_random_transf(input)

    return input, label


def main(opt_cut, hparams, plotter, visualizer):
    USRendereDefParams, cut_model, cut_trainer, early_stopping, early_stopping_best_val, inner_model, module, real_us_gt_test_dataloader, real_us_stopp_crit_dataloader, real_us_train_loader, train_loader_ct_labelmaps, val_loader_ct_labelmaps = prepare_for_training(
        hparams, opt_cut)

    train_losses, valid_losses, testset_losses, stoppcrit_losses, avg_train_losses, avg_valid_losses, avg_pred_mean_diff = (
        [] for i in range(7))
    # G_train_losses, D_train_losses, D_real_train_losses, D_fake_train_losses = ([] for i in range(4))
    # G_val_losses, D_val_losses, D_real_val_losses, D_fake_val_losses = ([] for i in range(4))

    # --------------------
    # RUN TRAINING
    # ---------------------
    only_CUT = True
    only_SEGNET = True
    cut_trainer.cut_model.set_requires_grad(USRendereDefParams,
                                            False)  # todo: comment this out if you want to train the USRendereDefParams
    for epoch in trange(1, hparams.max_epochs + 1):

        epoch_iter = 0
        opt_cut.visualizer.reset()
        iter_data_time = time.time()  # timer for data loading per iteration

        dataloader_real_us_iterator = iter(real_us_train_loader)
        step = 0
        module.train()
        module.outer_model.train()
        inner_model.train()

        # ------------------------------------------------------------------------------------------------------------------------------
        #                                         Train SEG NET only
        # ------------------------------------------------------------------------------------------------------------------------------
        if epoch <= hparams.epochs_only_pose_net and hparams.epochs_only_pose_net > 0:
            # print(f"--------------- INIT SEG NET ------------", cut_trainer.total_iter)
            training_loop_pose_net(hparams, module, inner_model, hparams.batch_size_manual, train_loader_ct_labelmaps,
                                   train_losses, plotter)  # todo: change this to pose net

        # opt_cut.isTrain = True
        # ------------------------------------------------------------------------------------------------------------------------------
        #                                         Train CUT only
        # ------------------------------------------------------------------------------------------------------------------------------
        if epoch <= hparams.epochs_only_cut and hparams.epochs_only_cut > 0:
            for i, batch_data_ct in tqdm(enumerate(train_loader_ct_labelmaps), total=len(train_loader_ct_labelmaps),
                                         ncols=100, position=0, leave=True):
                # print(f"--------------- INIT CUT ------------", cut_trainer.total_iter)
                cut_trainer.train_cut(module, batch_data_ct, epoch, dataloader_real_us_iterator, iter_data_time,
                                      epoch_iter)

        # ------------------------------------------------------------------------------------------------------------------------------
        #                                         Train US Renderer + SEG NET + CUT
        # ------------------------------------------------------------------------------------------------------------------------------
        if epoch > hparams.epochs_only_pose_net:
            only_CUT, only_SEGNET = train_epoch_full_pipeline(cut_trainer, dataloader_real_us_iterator, epoch,
                                                              epoch_iter, hparams, inner_model, iter_data_time, module,
                                                              only_CUT, only_SEGNET, plotter, step,
                                                              train_loader_ct_labelmaps, train_losses)

        print(f"--------------- ONLY CUT: {only_CUT} -------------- epoch: ", epoch)
        print(f"--------------- ONLY SEGNET: {only_SEGNET} -------------- epoch: ", epoch)

        # ------------------------------------------------------------------------------------------------------------------------------
        #                             POSE NET VALIDATION on every hparams.validate_every_n_steps
        # ------------------------------------------------------------------------------------------------------------------------------
        stop_early = False
        if epoch % hparams.validate_every_n_steps == 0 and epoch > hparams.epochs_only_cut:
            stop_early = validation_and_stop_check(USRendereDefParams, avg_train_losses, avg_valid_losses, cut_model,
                                                   cut_trainer, dataloader_real_us_iterator, early_stopping,
                                                   early_stopping_best_val, epoch, hparams, inner_model, module,
                                                   plotter, real_us_gt_test_dataloader, real_us_stopp_crit_dataloader,
                                                   real_us_train_loader, stoppcrit_losses, testset_losses, train_losses,
                                                   val_loader_ct_labelmaps, valid_losses)

        if stop_early:
            break

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

    print(
        f'train completed, avg_train_losses: {np.mean(avg_train_losses)} avg_valid_losses: {np.mean(avg_valid_losses)}')
    print(f'best val_loss: {early_stopping.val_loss_min} at best_epoch: {early_stopping.best_epoch}')


def validation_and_stop_check(USRendereDefParams, avg_train_losses, avg_valid_losses, cut_model, cut_trainer,
                              dataloader_real_us_iterator, early_stopping, early_stopping_best_val, epoch,
                              hparams, inner_model, module, plotter, real_us_gt_test_dataloader,
                              real_us_stopp_crit_dataloader, real_us_train_loader, stoppcrit_losses,
                              testset_losses, train_losses, val_loader_ct_labelmaps, valid_losses):
    stop_early = False
    module.eval()
    module.outer_model.eval()
    inner_model.eval()
    cut_model.eval()
    USRendereDefParams.eval()
    val_step = 0
    print(f"--------------- VALIDATION SEG NET ------------")
    stopp_crit_plot_figs = []
    def_renderer_plot_figs = []
    wasserstein_distance_bw_rendered_reconstr = []
    wasserstein_distance_bw_rendered_idt_reconstr = []
    with torch.no_grad():
        for nr, val_batch_data_ct in tqdm(enumerate(val_loader_ct_labelmaps),
                                          total=len(val_loader_ct_labelmaps), ncols=100, desc="validation"):
            val_step += 1

            validation_one_step(USRendereDefParams, cut_trainer, dataloader_real_us_iterator, def_renderer_plot_figs,
                                epoch, hparams, module, nr, plotter, real_us_train_loader, stopp_crit_plot_figs,
                                val_batch_data_ct, valid_losses, wasserstein_distance_bw_rendered_idt_reconstr,
                                wasserstein_distance_bw_rendered_reconstr)

        if len(stopp_crit_plot_figs) > 0:
            plot_stop_criteria_figs(epoch, hparams, plotter, stopp_crit_plot_figs,
                                    wasserstein_distance_bw_rendered_idt_reconstr,
                                    wasserstein_distance_bw_rendered_reconstr)

        if len(def_renderer_plot_figs) > 0:
            if hparams.logging:
                plotter.log_image(torchvision.utils.make_grid(def_renderer_plot_figs),
                                  "default_renderer|labelmap|defaultUS|learnedUS")
    # calculate average loss over an epoch
    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    avg_train_losses.append(train_loss)
    avg_valid_losses.append(valid_loss)
    epoch_len = len(str(hparams.max_epochs))
    print(f'[{epoch:>{epoch_len}}/{hparams.max_epochs:>{epoch_len}}] ' +
          f'pose_train_loss_epoch: {train_loss:.5f} ' +
          f'pose_valid_loss_epoch: {valid_loss:.5f}')
    if hparams.logging: wandb.log({"pose_train_loss_epoch": train_loss, "epoch": epoch})
    if hparams.logging: wandb.log({"pose_val_loss_epoch": valid_loss, "epoch": epoch})
    if hparams.logging and not hparams.log_default_renderer: plotter.validation_epoch_end()
    if (epoch > 20):  # hparams.min_epochs):
        early_stopping_best_val(valid_loss, epoch, module, cut_model.netG)
    # ------------------------------------------------------------------------------------------------------------------------------
    #                                      END POSE NETWORK VALIDATION
    # ------------------------------------------------------------------------------------------------------------------------------
    print(f"--------------- END POSE NETWORK VALIDATION ------------")
    cut_model.eval()

    # run inference
    # ------------------------------------------------------------------------------------------------------------------------------
    #             (GT STOPPING CRITERION) INFER REAL US STOPPONG CRIT TEST SET IMGS THROUGH THE CUT+SEG NET
    # -------------------------------------------------------------------------------------------------------------------------------
    avg_stopp_crit_loss = 1
    if hparams.stopping_crit_gt_imgs and epoch > hparams.epochs_check_stopp_crit:
        print(f"--------------- STOPPING CRITERIA US IMGS CHECK ------------")
        sopp_crit_imgs_plot_figs = []
        check_early_stopping(avg_stopp_crit_loss, cut_model, early_stopping, epoch, hparams, module, plotter,
                             real_us_stopp_crit_dataloader, sopp_crit_imgs_plot_figs, stoppcrit_losses)

        if early_stopping.early_stop and hparams.min_epochs:
            print("Early stopping")
            stop_early = True
        stoppcrit_losses = []

        # ------------------------------------------------------------------------------------------------------------------------------
        #             INFER REAL US IMGS THROUGH THE CUT+SEG NET - INFER THE WHOLE TEST SET
        # ------------------------------------------------------------------------------------------------------------------------------
        avg_testset_loss = 1
        std_testset_loss = 0
        if not stop_early:
            print(f"--------------- INFERENCE: WHOLE TEST US GT IMGS  ------------")
            gt_test_imgs_plot_figs = []

            infer_whole_dataset(cut_model, epoch, gt_test_imgs_plot_figs, hparams, module, plotter,
                                real_us_gt_test_dataloader, testset_losses)

        testset_losses = []
        hausdorff_epoch = []
        # ------------------------------------------------------------------------------------------------------------------------------
    return stop_early


def prepare_for_training(hparams, opt_cut):
    """
    Prepare components for training including models, datasets and other utilities.

    Args:
    hparams (object): hyperparameters required for preparation.
    opt_cut (object): options required for CUT training.

    Returns:
    tuple: containing models, datasets, loaders, trainers, and early stopping handlers.
    """

    # ---------------------
    # LOAD MODEL
    # ---------------------

    # Create CUT model based on the given options
    cut_model = cut_create_model(opt_cut)

    # Load module and model classes
    ModuleClass = load_module(hparams.module)
    InnerModelClass = load_model(hparams.inner_model)

    # Initialize inner model with hyperparameters
    inner_model = InnerModelClass(params=hparams)

    # Initialize Ultrasound Rendering with default parameters and set to device
    USRendereDefParams = UltrasoundRendering(params=hparams, default_param=True).to(hparams.device)

    # Initialize module with hyperparameters and inner model
    module = ModuleClass(params=hparams, inner_model=inner_model)

    # Set up logging if required
    if hparams.logging:
        wandb.watch([inner_model, module.outer_model], log='all', log_graph=True, log_freq=100)

    # ---------------------
    # LOAD DATA
    # ---------------------

    # Load CT dataset
    CTDatasetLoader = load_dataset(hparams.dataloader_ct_labelmaps)
    dataloader = CTDatasetLoader(hparams)
    train_loader_ct_labelmaps, train_dataset_ct_labelmaps, val_dataset_ct_labelmaps = dataloader.train_dataloader()
    val_loader_ct_labelmaps = dataloader.val_dataloader()

    # Create real ultrasound dataset and get its actual dataset  # todo: there is no image size here
    real_us_dataset = cut_create_dataset(opt_cut)
    dataset_real_us = real_us_dataset.dataset

    # Create real ultrasound data loader for training
    real_us_train_loader = torch.utils.data.DataLoader(
        dataset_real_us,
        batch_size=opt_cut.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=hparams.num_workers
    )

    # Load real US GT test and stopping criterion datasets and dataloaders
    RealUSGTDatasetClass = load_dataset(hparams.dataloader_real_us_test)
    real_us_gt_testdataset = RealUSGTDatasetClass(root_dir=hparams.data_dir_real_us_test)
    real_us_gt_test_dataloader = torch.utils.data.DataLoader(real_us_gt_testdataset, shuffle=False)

    real_us_stopp_crit_dataset = RealUSGTDatasetClass(root_dir=hparams.data_dir_real_us_stopp_crit)
    real_us_stopp_crit_dataloader = torch.utils.data.DataLoader(real_us_stopp_crit_dataset, shuffle=False)

    # ---------------------
    # TRAINING UTILITIES
    # ---------------------

    # Initialize CUT trainer with model, inner model, and dataset
    cut_trainer = CUTTrainer(opt_cut, cut_model, inner_model, dataset_real_us, real_us_train_loader)

    # Initialize early stopping handlers
    early_stopping = EarlyStopping(
        patience=hparams.early_stopping_patience,
        ckpt_save_path_model1=f'{hparams.output_path}/best_checkpoint_pose_renderer_test_loss_{hparams.exp_name}',
        ckpt_save_path_model2=f'{hparams.output_path}/best_checkpoint_CUT_test_loss_{hparams.exp_name}',
        verbose=True
    )

    early_stopping_best_val = EarlyStopping(
        patience=hparams.early_stopping_patience,
        ckpt_save_path_model1=f'{hparams.output_path}/best_checkpoint_pose_renderer_valid_loss_{hparams.exp_name}',
        ckpt_save_path_model2=f'{hparams.output_path}/best_checkpoint_CUT_val_loss_{hparams.exp_name}',
        verbose=True
    )

    # Return all prepared components
    return (
        USRendereDefParams, cut_model, cut_trainer, early_stopping, early_stopping_best_val,
        inner_model, module, real_us_gt_test_dataloader, real_us_stopp_crit_dataloader,
        real_us_train_loader, train_loader_ct_labelmaps, val_loader_ct_labelmaps
    )


def infer_whole_dataset(cut_model, epoch, gt_test_imgs_plot_figs, hparams, module, plotter,
                        real_us_gt_test_dataloader, testset_losses):
    # Ensure no gradient computation for performance during inference
    with torch.no_grad():
        # Loop over the real US test data
        for nr, batch_data_real_us_test in tqdm(enumerate(real_us_gt_test_dataloader),
                                                total=len(real_us_gt_test_dataloader), ncols=100,
                                                position=0, leave=True):

            # Extract images and their labels from the batch data
            real_us_test_img, real_label = batch_data_real_us_test[0].to(hparams.device), \
                batch_data_real_us_test[1].to(hparams.device).float()

            # Use the CUT model to reconstruct the US images
            reconstructed_us_testset = cut_model.netG(real_us_test_img)

            # Normalize reconstructed US images from range [-1, 1] to [0, 1]
            reconstructed_us_testset = (reconstructed_us_testset / 2) + 0.5

            # Forward pass through pose regression model
            testset_loss, pose_pred = module.pose_forward(reconstructed_us_testset, real_label)
            testset_losses.append(testset_loss)

            # If logging is enabled
            if hparams.logging:
                # Log the test set loss
                wandb.log({"testset_loss": testset_loss.item()})

                # If we're still below the max number of images to plot
                if nr < NR_IMGS_TO_PLOT:
                    # Normalize real US test images for plotting
                    real_us_test_img = (real_us_test_img / 2) + 0.5  # from [-1,1] to [0,1]

                    # # Compute the Hausdorff distance between predictions and labels # todo: add pose loss

                    # Create a plot with the GT, real US image, reconstructed US, predicted pose, and GT pose
                    gt_label_list = ["{:.4f}".format(value) for value in real_label.cpu().numpy().tolist()[0]]
                    pred_list = ["{:.4f}".format(value) for value in pose_pred.cpu().numpy().tolist()[0]]
                    plot_fig = plotter.plot_stopp_crit(
                        caption="stestset_gt_|real_us|reconstructed_us",
                        imgs=[real_us_test_img, reconstructed_us_testset],
                        img_text=f'estimated pose: {",".join(pred_list)}, gt pose: {",".join(gt_label_list)}',
                        epoch=epoch,
                        plot_single=False
                    )
                    gt_test_imgs_plot_figs.append(plot_fig)

        # If we've created any plots, then log them
        if len(gt_test_imgs_plot_figs) > 0:
            plotter.log_image(torchvision.utils.make_grid(gt_test_imgs_plot_figs),
                              "testset_gt_|real_us|reconstructed_us")

            # Compute and log statistics about test set loss and Hausdorff distances
            avg_testset_loss = torch.mean(torch.stack(testset_losses))
            std_testset_loss = torch.std(torch.stack(testset_losses))
            # avg_hausdorff_dist = torch.mean(torch.stack(hausdorff_epoch))
            # std_hausdorff = torch.std(torch.stack(hausdorff_epoch))

            wandb.log({
                "testset_gt_loss_epoch": avg_testset_loss,
                "testset_gt_loss_std_epoch": std_testset_loss,
                # "tesset_hausdorff_dist_epoch": avg_hausdorff_dist,
                # "testset_hausdorff_std_epoch": std_hausdorff,
                "epoch": epoch
            })

            # Clear the list of plots
            gt_test_imgs_plot_figs = []


def check_early_stopping(avg_stopp_crit_loss, cut_model, early_stopping, epoch, hparams, module, plotter,
                         real_us_stopp_crit_dataloader, sopp_crit_imgs_plot_figs, stoppcrit_losses):
    with torch.no_grad():
        for nr, batch_data_real_us_stopp_crit in tqdm(enumerate(real_us_stopp_crit_dataloader),
                                                      total=len(real_us_stopp_crit_dataloader), ncols=100,
                                                      position=0, leave=True):

            real_us_img, real_us_label = batch_data_real_us_stopp_crit[0].to(
                hparams.device), batch_data_real_us_stopp_crit[1].to(hparams.device).float()
            reconstructed_us_stopp_crit = cut_model.netG(real_us_img)

            reconstructed_us_stopp_crit = (reconstructed_us_stopp_crit / 2) + 0.5  # from [-1,1] to [0,1]

            stop_crit_loss, pose_pred = module.pose_forward(reconstructed_us_stopp_crit, real_us_label)
            stoppcrit_losses.append(stop_crit_loss)

            if hparams.logging:
                real_us_img = (real_us_img / 2) + 0.5  # from [-1,1] to [0,1]

                wandb.log({"stop_crit_loss": stop_crit_loss.item()})
                plot_fig_gt = plotter.plot_stopp_crit(
                    caption="stopp_crit|real_us|reconstructed_us",
                    imgs=[real_us_img, reconstructed_us_stopp_crit],
                    img_text='loss=' + "{:.4f}".format(stop_crit_loss.item()), epoch=epoch,
                    plot_single=False)
                sopp_crit_imgs_plot_figs.append(plot_fig_gt)
    if len(sopp_crit_imgs_plot_figs) > 0:
        plotter.log_image(torchvision.utils.make_grid(sopp_crit_imgs_plot_figs),
                          "stopp_crit|real_us|reconstructed_us")
        avg_stopp_crit_loss = torch.mean(torch.stack(stoppcrit_losses))
        wandb.log({"stoppcrit_loss_epoch": avg_stopp_crit_loss, "epoch": epoch})
        stoppcrit_losses = []
        sopp_crit_imgs_plot_figs = []

        # stop_criterion_loss_avg_epoch = np.average(stopp_crit_losses)
    # print(f"stop_criterion_loss_avg_epoch: {stop_criterion_loss_avg_epoch}")
    # if hparams.logging: wandb.log({"stop_criterion_loss_avg_epoch": stop_criterion_loss_avg_epoch})
    # early_stopping needs the validation loss to check if it has decresed,
    # and if it has, it will make a checkpoint of the current model
    if epoch > hparams.min_epochs:  # hparams.min_epochs):
        early_stopping(avg_stopp_crit_loss, epoch, module, cut_model.netG)


def plot_stop_criteria_figs(epoch, hparams, plotter, stopp_crit_plot_figs,
                            wasserstein_distance_bw_rendered_idt_reconstr, wasserstein_distance_bw_rendered_reconstr):
    if hparams.logging:
        plotter.log_image(torchvision.utils.make_grid(stopp_crit_plot_figs),
                          "infer_CUT_during_val_|real_us|reconstructed_us|us_rendered")

        wandb.log({"mean_wasserstein_distance_bw_rendered&reconstr_epoch": np.mean(
            wasserstein_distance_bw_rendered_reconstr), "epoch": epoch})
        if hparams.use_idtB: wandb.log({
            "mean_wasserstein_distance_bw_rendered_idt&reconstr_epoch": np.mean(
                wasserstein_distance_bw_rendered_idt_reconstr),
            "epoch": epoch})


def validation_one_step(
        USRendereDefParams,
        cut_trainer,
        dataloader_real_us_iterator,
        def_renderer_plot_figs,
        epoch,
        hparams,
        module,
        nr,
        plotter,
        real_us_train_loader,
        figs_to_plot,
        val_batch_data_ct,
        valid_losses,
        wasserstein_distance_bw_rendered_idt_reconstr,
        wasserstein_distance_bw_rendered_reconstr
):
    # Extract validation data and label
    val_input, val_label, filename = module.get_data(val_batch_data_ct)
    val_input_copy = val_input.clone().detach()

    idt_B_val, us_sim = plot_validation_result(USRendereDefParams, cut_trainer, def_renderer_plot_figs,
                                                          epoch, filename, hparams, module, nr, plotter, val_input,
                                                          val_input_copy, val_label, valid_losses)

    # Check stopping criteria for epochs beyond specified threshold
    print(f"--------------- INFER UNLABELLED REAL US IMGS FROM CUT TRAIN SET THROUGH THE POSE NET ------------")
    try:
        data_cut_real_us = next(dataloader_real_us_iterator)
    except StopIteration:
        dataloader_real_us_iterator = iter(real_us_train_loader)
        data_cut_real_us = next(dataloader_real_us_iterator)

    # Forward pass through CUT model
    data_cut_real_us_domain_real = data_cut_real_us['A'].to(hparams.device)
    cut_trainer.forward_cut_A(data_cut_real_us_domain_real)
    reconstructed_us = cut_trainer.cut_model.fake_B
    reconstructed_us = (reconstructed_us / 2) + 0.5  # Convert tensor range from [-1,1] to [0,1]
    _, pose_pred = module.pose_forward(reconstructed_us, torch.zeros_like(val_label))

    wasserstein_distance_bw_rendered_reconstr.append(
        wasserstein_distance(us_sim.cpu().flatten(), reconstructed_us.cpu().flatten()))
    if hparams.use_idtB:
        wasserstein_distance_bw_rendered_idt_reconstr.append(
            wasserstein_distance(idt_B_val.cpu().flatten(), reconstructed_us.cpu().flatten()))

    if hparams.logging and nr < NR_IMGS_TO_PLOT:
        no_random_transform = cut_trainer.inference_transformatons()
        data_cut_real_us_domain_real = no_random_transform(data_cut_real_us_domain_real)
        data_cut_real_us_domain_real = (data_cut_real_us_domain_real / 2) + 0.5

        pred_list = ["{:.4f}".format(value) for value in pose_pred.cpu().numpy().tolist()[0]]
        plot_fig = plotter.plot_stopp_crit(
            caption="infer_CUT_during_val_|real_us|reconstructed_us|us_rendered",
            imgs=[data_cut_real_us_domain_real, reconstructed_us, us_sim],
            img_text=f'estimated pose: {",".join(pred_list)}',
            epoch=epoch,
            plot_single=False
        )
        figs_to_plot.append(plot_fig)


def plot_validation_result(USRendereDefParams, cut_trainer, def_renderer_plot_figs, epoch, filename, hparams, module,
                           nr, plotter, val_input, val_input_copy, val_label, valid_losses):
    # Determine if to use idtB (Identity Translation from Domain B)
    if hparams.use_idtB:
        us_sim = module.rendering_forward(val_input)
        cut_trainer.forward_cut_B(us_sim)
        idt_B_val = cut_trainer.cut_model.idt_B
        idt_B_val = (idt_B_val / 2) + 0.5  # Convert tensor range from [-1,1] to [0,1]
        val_loss_step, pose_pred = module.pose_forward(idt_B_val, val_label)
        if not hparams.log_default_renderer:
            dict_ = module.plot_val_results(val_input, val_loss_step, filename, val_label, pose_pred, idt_B_val, epoch)
    else:
        us_sim = module.rendering_forward(val_input)
        idt_B_val = us_sim

        # Apply online augmentations if required
        if hparams.debug and hparams.net_input_augmentations_noise_blur:
            us_sim, val_label = add_online_augmentations(hparams, us_sim, val_label, module)

        val_loss_step, pose_pred = module.pose_forward(us_sim, val_label)
        if not hparams.log_default_renderer:
            dict_ = module.plot_val_results(val_input, val_loss_step, filename, val_label, pose_pred, us_sim, epoch)
    # Append current validation loss
    valid_losses.append(val_loss_step.item())
    # If logging the default renderer and within plotting limits
    if hparams.log_default_renderer and nr < NR_IMGS_TO_PLOT:
        us_sim_def = USRendereDefParams(val_input_copy.squeeze())
        plot_fig = create_fig(epoch, idt_B_val, plotter, pose_pred, us_sim, us_sim_def, val_input, val_label)
        def_renderer_plot_figs.append(plot_fig)
    elif not hparams.log_default_renderer:
        plotter.validation_batch_end(dict_)
    return idt_B_val, us_sim


def create_fig(epoch, idt_B, plotter, pose_pred, us_sim, us_sim_def, img_input, gt_label):
    # Convert tensor data to formatted strings
    gt_label_list = ["{:.4f}".format(value) for value in gt_label.cpu().numpy().tolist()[0]]
    pred_list = ["{:.4f}".format(value) for value in pose_pred.cpu().numpy().tolist()[0]]
    plot_fig = plotter.plot_stopp_crit(
        caption="default_renderer|labelmap|defaultUS|learnedUS|idtB",
        imgs=[img_input, us_sim_def, us_sim, idt_B],
        img_text=f'estimated pose: {",".join(pred_list)}, gt pose: {",".join(gt_label_list)}',
        epoch=epoch,
        plot_single=False
    )
    return plot_fig


def train_epoch_full_pipeline(cut_trainer, dataloader_real_us_iterator, epoch, epoch_iter, hparams, inner_model,
                              iter_data_time, module, only_CUT, only_SEGNET, plotter, step, train_loader_ct_labelmaps,
                              train_losses):
    if only_CUT: only_CUT = False
    if only_SEGNET: only_SEGNET = False
    step += 1
    batch_loss_list = []
    if hparams.batch_size_manual == 1:  # if batch==1, for now this is always true
        for i, batch_data_ct in tqdm(enumerate(train_loader_ct_labelmaps), total=len(train_loader_ct_labelmaps),
                                     ncols=100, desc=f"training epoch {epoch}:"):
            train_one_step(batch_data_ct, cut_trainer, dataloader_real_us_iterator, epoch, epoch_iter, hparams,
                           i, iter_data_time, module, plotter, step, train_losses)

    else:  # if batch>1  # this is not called for now
        train_larger_batch_size(batch_loss_list, cut_trainer, dataloader_real_us_iterator, epoch, epoch_iter,
                                hparams, step, inner_model, iter_data_time, module, plotter, step,
                                train_loader_ct_labelmaps, train_losses)
    if hparams.scheduler:
        module.scheduler.step()
        wandb.log({"lr_": module.optimizer.param_groups[0]["lr"]})
        wandb.log({"lr_2": module.optimizer.param_groups[1]["lr"]})
    # ------------------------------------------------------------------------------------------------------------------------------
    #                                         Plot CUT results during training
    # ------------------------------------------------------------------------------------------------------------------------------
    if len(cut_trainer.cut_plot_figs) > 0:
        plotter.log_image(torchvision.utils.make_grid(cut_trainer.cut_plot_figs), "real_A|fake_B|real_B|idt_B")
        cut_trainer.cut_plot_figs = []

        # acoustic_impedance_dict_before_cut = inner_model.acoustic_impedance_dict
        # plotter.log_us_rendering_values(inner_model, step)
        # cut_trainer.train_cut(batch_data_ct, epoch, dataloader_real_us_iterator, i, iter_data_time, epoch_iter)
        # if not torch.equal(acoustic_impedance_dict_before_cut, inner_model.acoustic_impedance_dict):
        #     print(f"--------------- US PARAMS CHANGED --------------")
    return only_CUT, only_SEGNET


def train_larger_batch_size(batch_loss_list, cut_trainer, dataloader_real_us_iterator, epoch, epoch_iter, hparams, i,
                            inner_model, iter_data_time, module, plotter, step, train_loader_ct_labelmaps,
                            train_losses):
    dataloader_iterator = iter(train_loader_ct_labelmaps)
    while step < len(train_loader_ct_labelmaps):
        step += 1
        module.optimizer.zero_grad()
        # while step % hparams.batch_size_manual != 0 :
        data = None
        while step % hparams.batch_size_manual != 0:
            data = next(dataloader_iterator)

            input, label = module.get_data(data)
            loss, us_sim, prediction = module.step(input, label)
            batch_loss_list.append(loss)
            step += 1
            if hparams.logging: wandb.log({"train_loss_step": loss.item()}, step=step)
            train_losses.append(loss.item())

        loss = torch.mean(torch.stack(batch_loss_list))

        # check_gradients(module)
        loss.backward()
        module.optimizer.step()
        batch_loss_list = []

        print(
            f"{step}/{len(train_loader_ct_labelmaps.dataset) // train_loader_ct_labelmaps.batch_size}, train_loss: {loss.item():.4f}")
        if hparams.logging: plotter.log_us_rendering_values(inner_model, step)

        if data is not None:
            cut_trainer.train_cut(module, data, epoch, dataloader_real_us_iterator, iter_data_time, epoch_iter)


def train_one_step(batch_data_ct, cut_trainer, dataloader_real_us_iterator, epoch, epoch_iter, hparams, i,
                   iter_data_time, module, plotter, step, train_losses):
    step += 1
    module.optimizer.zero_grad()
    input, label, filename = module.get_data(batch_data_ct)
    if hparams.use_idtB:  # we know that use_idtB is always true for now
        us_sim = module.rendering_forward(input)
        us_sim_cut = us_sim.clone().detach()
        cut_trainer.train_cut(module, us_sim_cut, epoch, dataloader_real_us_iterator, iter_data_time, epoch_iter)

        # label = cut_trainer.forward_cut_B(us_sim, label)
        cut_trainer.forward_cut_B(us_sim)
        idt_B = cut_trainer.cut_model.idt_B
        # idt_B = idt_B.clone()
        idt_B = (idt_B / 2) + 0.5  # from [-1,1] to [0,1]

        if hparams.net_input_augmentations_noise_blur:
            idt_B, label = add_online_augmentations(hparams, idt_B, label, module)

        loss, prediction = module.pose_forward(idt_B, label)

        if hparams.inner_model_learning_rate == 0:
            cut_trainer.cut_model.set_requires_grad(module.USRenderingModel, False)
        loss.backward()
        # Perform gradient clipping
        if hparams.grad_clipping:
            torch.nn.utils.clip_grad_norm_(module.USRenderingModel.parameters(), max_norm=1)
            torch.nn.utils.clip_grad_norm_(module.outer_model.parameters(), max_norm=1)

        module.optimizer.step()
    else:
        us_sim = module.rendering_forward(input)
        if hparams.net_input_augmentations_noise_blur:
            us_sim, label = add_online_augmentations(hparams, us_sim, label, module)
        loss, prediction = module.pose_forward(us_sim, label)
        # check_gradients(module)
        # with torch.autograd.detect_anomaly():
        # log_model_gradients(inner_model, step)

        if hparams.inner_model_learning_rate == 0:
            cut_trainer.cut_model.set_requires_grad(module.USRenderingModel, False)
        loss.backward()
        module.optimizer.step()
        us_sim_cut = us_sim.clone().detach()

        # cut_trainer.train_cut(module, us_sim_cut, epoch, dataloader_real_us_iterator, i, iter_data_time, epoch_iter)
    # print(f"{step}/{len(train_loader_ct_labelmaps.dataset) // train_loader_ct_labelmaps.batch_size}, train_loss: {loss.item():.4f}")
    if hparams.logging:
        wandb.log({"train_loss_step": loss.item()})
    train_losses.append(loss.item())
    if hparams.logging:
        plotter.log_us_rendering_values(module.USRenderingModel, step)


MANUAL_SEED = False
NR_IMGS_TO_PLOT = 32

if __name__ == "__main__":

    if MANUAL_SEED:
        torch.manual_seed(2023)
    # torch.multiprocessing.set_start_method('spawn')

    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('-c', is_config_file=True, help='config file path')
    parser, hparams = build_configargparser(parser)
    hparams.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    opt_cut = CACTUSSEnd2EndOptions().parse()  # get training options
    opt_cut.dataroot = hparams.data_dir_real_us_cut_training
    opt_cut.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hparams.aorta_only:
        opt_cut.no_flip = True

    if hparams.debug:
        hparams.exp_name = 'DEBUG'
    else:
        hparams.exp_name += str(opt_cut.lr) + '_' + str(hparams.inner_model_learning_rate) + '_' + str(
            hparams.outer_model_learning_rate)

    # todo: define params to run on slurm cluster

    hparams.exp_name = str(random.randint(0, 1000)) + "_" + hparams.exp_name

    if hparams.logging: wandb.init(name=hparams.exp_name, project=hparams.project_name)
    plotter = Plotter()
    visualizer = CUTVisualizer(opt_cut)  # create a visualizer that display/save images and plots
    opt_cut.visualizer = visualizer

    print(f'************************************************************************************* \n'
          #   f'COMMENT: {COMMENT}' # + str(hparams.nr_train_folds)} \n'
          #   f'************************************************************************************* \n'
          f'PYTHON VERSION: {sys.version} \n '
          # f'WANDB VERSION: {wandb.__version__} \n '
          f'TORCH VERSION: {torch.__version__} \n '
          #   f'TORCHVISION VERSION: {torchvision.__version__} \n '
          f'This will run on polyaxon: {str(hparams.on_polyaxon)} \n'
          f'torch.cuda.is_available(): {torch.cuda.is_available()} '
          f'************************************************************************************* \n'
          )

    argparse_summary(hparams, parser)

    main(opt_cut=opt_cut, hparams=hparams, plotter=plotter, visualizer=visualizer)

    # load the last checkpoint with the best model

    # model.load_state_dict(torch.load('checkpoint.pt'))
