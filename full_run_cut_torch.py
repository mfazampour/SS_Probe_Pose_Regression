from collections import OrderedDict
import logging
import random
import sys
import numpy as np
import torch
import time
import wandb
from tqdm.auto import tqdm, trange
import torchvision
# from tensorboardX import SummaryWriter
# import monai
import configargparse
from utils.configargparse_arguments_torch import build_configargparser
import torchvision.transforms.functional as F
from utils.plotter_torch import Plotter
from utils.utils import argparse_summary, get_class_by_path
from utils.early_stopping import EarlyStopping

from cut.data import create_dataset as cut_create_dataset
from cut.models import create_model as cut_create_model
from cut.options.cactussend2end_options import CACTUSSEnd2EndOptions
from cut.util.visualizer import Visualizer as CUTVisualizer
from cut.data.base_dataset import get_transform as cut_get_transform
# from cut.options.train_options import TrainOptions as TrainOptionsCUT
import torchvision.transforms as transforms
from torch.utils.data import random_split
import torchvision.transforms as T
from torch.nn.functional import kl_div
from scipy.stats import wasserstein_distance
from full_run_torch import training_loop_seg_net
from monai.metrics import HausdorffDistanceMetric
from models.us_rendering_model_torch import UltrasoundRendering

MANUAL_SEED = False
NR_IMGS_TO_PLOT = 32

# torch.use_deterministic_algorithms(True, warn_only=True)
# tb_logger = SummaryWriter('log_dir/cactuss_end2end')

# Define the desired transforms
# transform_real_us_data = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

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
        #torch.set_rng_state(self.random_state)
        no_random_transform = self.inference_transformatons()   #Resize and Normalize -1,1
        data_cut_reconstructed_us = no_random_transform(us_real)
        self.data_cut_real_us['A'] = data_cut_reconstructed_us   # add cut_model.real_B
        self.cut_model.set_input(self.data_cut_real_us)  # unpack data from dataset and apply preprocessing
        self.cut_model.forward()


    def forward_cut_B(self, us_sim):
        
        #torch.set_rng_state(self.random_state)
        no_random_transform = self.inference_transformatons()
        data_cut_rendered_us = no_random_transform(us_sim)
        self.data_cut_real_us['B'] = data_cut_rendered_us   # add cut_model.real_B
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



    def train_cut(self, module, us_sim_resized, epoch, dataloader_real_us_iterator, i, iter_data_time, epoch_iter):
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
        self.data_cut_real_us['B'] = data_cut_rendered_us   # add cut_model.real_B
        
        optimize_start_time = time.time()
        # if epoch == opt_cut.epoch_count and self.init_cut:    #initialize only on epoch 1 
        if self.init_cut:    #initialize cut
            print(f"--------------- INIT CUT --------------")
            # self.data_cut_real_us['B'] = data_cut_rendered_us.detach()
            self.init_cut = False
            self.cut_model.data_dependent_initialize(self.data_cut_real_us)
            self.cut_model.setup(self.opt_cut)               # regular setup: load and print networks; create schedulers
            self.cut_model.parallelize()

        self.cut_model.set_input(self.data_cut_real_us)  # unpack data from dataset and apply preprocessing
        # forward
        self.cut_model.forward()

        self.cut_optimize_parameters(module, None)
        self.cut_optimizer_step()

        # self.cut_model.optimize_parameters()   # calculate loss functions, get gradients, update network weights, backward()
           # calculate loss functions, get gradients, update network weights, backward()
        self.optimize_time = (time.time() - optimize_start_time) / real_us_batch_size * 0.005 + 0.995 * self.optimize_time

        # self.cut_model.compute_visuals()
        # cut_imgs = self.cut_model.get_current_visuals()
        # idt_B = self.cut_model.idt_B


        iter_start_time = time.time() 
        if self.total_iter % self.opt_cut.print_freq == 0:
            self.t_data = iter_start_time - iter_data_time

        #Print CUT results
        if self.total_iter % self.opt_cut.display_freq == 0:   # display images and losses on wandb
            self.cut_model.compute_visuals()
            self.cut_plot_figs.append(plotter.plot_images(self.cut_model.get_current_visuals(), epoch, wandb, plot_single=False))
            losses = self.cut_model.get_current_losses()
            self.opt_cut.visualizer.print_current_losses(epoch, epoch_iter, losses, self.optimize_time, self.t_data, wandb)




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

def calc_kl_div(img_nr, epoch):
    # Normalize the images to have a probability distribution over pixel values
    img1 = img1 / torch.sum(img1)
    img2 = img2 / torch.sum(img2)

    # Compute the KL divergence between the two images
    kl_divergence = kl_div(img1.flatten(), img2.flatten())

    print("KL divergence between the two images: ", kl_divergence.item())
    if hparams.logging: wandb.log({"kl_divergence_epoch_" + str(epoch): kl_divergence.item()})

    return kl_divergence.item()


def add_online_augmentations(hparams, input, label):

    if hparams.seg_net_input_augmentations_noise_blur:
        input = module.rendered_img_random_transf(input)

    if hparams.seg_net_input_augmentations_rand_crop:
        state = torch.get_rng_state()
        input = module.rendered_img_masks_random_transf(input)
        torch.set_rng_state(state)
        label = module.rendered_img_masks_random_transf(label)

    return input, label

if __name__ == "__main__":

    # monai.config.print_config()
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    if MANUAL_SEED: torch.manual_seed(2023)
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

    opt_cut = CACTUSSEnd2EndOptions().parse()   # get training options
    opt_cut.dataroot = hparams.data_dir_real_us_cut_training
    opt_cut.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hparams.aorta_only: opt_cut.no_flip = True

    if hparams.debug:
        hparams.exp_name = 'DEBUG'
    else:
        hparams.exp_name += str(hparams.pred_label) + '_' + str(opt_cut.lr) + '_' + str(hparams.inner_model_learning_rate) + '_' + str(hparams.outer_model_learning_rate)
    
    if hparams.on_polyaxon:
        from polyaxon_client.tracking import Experiment, get_data_paths, get_outputs_path
        print('get_data_paths(): ', get_data_paths())

        hparams.base_folder_data_path = get_data_paths()['data1'] + hparams.polyaxon_imgs_folder
        hparams.base_folder_mask_path = get_data_paths()['data1'] + hparams.polyaxon_masks_folder
        print('Labelmaps DATASET Folder: ', hparams.base_folder_data_path)

        hparams.data_dir_real_us_cut_training = get_data_paths()['data1'] + hparams.polyaxon_folder_real_us
        opt_cut.dataroot = hparams.data_dir_real_us_cut_training
        print('Real US DATASET Folder: ', hparams.data_dir_real_us_cut_training)

        hparams.data_dir_real_us_test = get_data_paths()['data1'] + hparams.polyaxon_data_dir_real_us_test
        print('Real US GT DATASET Folder: ', hparams.data_dir_real_us_test)

        hparams.data_dir_real_us_stopp_crit = get_data_paths()['data1'] + hparams.polyaxon_data_dir_real_us_stopp_crit
        print('Real US GT DATASET FOR STOPP CRIT Folder: ', hparams.data_dir_real_us_stopp_crit)

        hparams.output_path = get_outputs_path()
        poly_experiment_info = Experiment.get_experiment_info()
        poly_experiment_nr = poly_experiment_info['experiment_name'].split(".")[-1]
        hparams.exp_name = poly_experiment_nr + "_" + hparams.exp_name
        print(f'get_outputs_path: {hparams.output_path} \n '
              f'experiment_info: {poly_experiment_info} \n experiment_nr: {poly_experiment_nr}')
    else:
        hparams.exp_name = str(random.randint(0, 1000)) + "_" + hparams.exp_name


    if hparams.logging: wandb.init(name=hparams.exp_name, project=hparams.project_name)
    plotter = Plotter()
    visualizer = CUTVisualizer(opt_cut)   # create a visualizer that display/save images and plots
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


    # ---------------------
    # LOAD MODEL
    # ---------------------
    cut_model = cut_create_model(opt_cut)      # create a model given opt.model and other options

    ModuleClass = load_module(hparams.module)
    InnerModelClass = load_model(hparams.inner_model)
    # OuterModelClass = hparams.outer_model
    inner_model = InnerModelClass(params=hparams)

    # InnerModelClass2 = load_model(hparams.inner_model)
    USRendereDefParams = UltrasoundRendering(params=hparams, default_param=True).to(hparams.device)



    if not hparams.outer_model_monai: 
        OuterModelClass = load_model(hparams.outer_model)
        outer_model = OuterModelClass(hparams=hparams)
        module = ModuleClass(params=hparams, inner_model=inner_model, outer_model=outer_model)
    else:
        module = ModuleClass(params=hparams, inner_model=inner_model)

    # wandb.watch(inner_model, log='all', log_graph=True, log_freq=10)
    if hparams.logging: wandb.watch([inner_model, module.outer_model], log='all', log_graph=True, log_freq=100)

    # ---------------------
    # LOAD DATA 
    # ---------------------
    CTDatasetLoader = load_dataset(hparams.dataloader_ct_labelmaps)
    dataloader = CTDatasetLoader(hparams)
    train_loader_ct_labelmaps, train_dataset_ct_labelmaps, val_dataset_ct_labelmaps  = dataloader.train_dataloader()
    val_loader_ct_labelmaps = dataloader.val_dataloader()

    real_us_dataset = cut_create_dataset(opt_cut)  # create a dataset given opt.dataset_mode and other options
    RealUSGTDatasetClass = load_dataset(hparams.dataloader_real_us_test)
    dataset_real_us = real_us_dataset.dataset   
    # dataset_size = len(dataset_real_us)    # get the number of images in the dataset.
    # train_size = int(0.8 * len(dataset_real_us))
    # val_size = len(dataset_real_us) - train_size
    # train_dataset_real_us, val_dataset_real_us = random_split(dataset_real_us, [train_size, val_size])#, generator=Generator().manual_seed(0))
    real_us_train_loader = torch.utils.data.DataLoader(dataset_real_us, batch_size=opt_cut.batch_size, shuffle=True, drop_last=True, num_workers=hparams.num_workers)
    # real_us_val_loader = torch.utils.data.DataLoader(val_dataset_real_us, batch_size=opt_cut.batch_size, shuffle=False, drop_last=False, num_workers=hparams.num_workers)

    real_us_gt_testdataset = RealUSGTDatasetClass(root_dir=hparams.data_dir_real_us_test)
    real_us_gt_test_dataloader = torch.utils.data.DataLoader(real_us_gt_testdataset, shuffle=False)

    real_us_stopp_crit_dataset = RealUSGTDatasetClass(root_dir=hparams.data_dir_real_us_stopp_crit)
    real_us_stopp_crit_dataloader = torch.utils.data.DataLoader(real_us_stopp_crit_dataset, shuffle=False)


    cut_trainer = CUTTrainer(opt_cut, cut_model, inner_model, dataset_real_us, real_us_train_loader)
    early_stopping = EarlyStopping(patience=hparams.early_stopping_patience, 
                                    ckpt_save_path_model1 = f'{hparams.output_path}/best_checkpoint_seg_renderer_test_loss_{hparams.exp_name}', 
                                    ckpt_save_path_model2 = f'{hparams.output_path}/best_checkpoint_CUT_test_loss_{hparams.exp_name}', 
                                    verbose=True)

    early_stopping_best_val = EarlyStopping(patience=hparams.early_stopping_patience, 
                                    ckpt_save_path_model1 = f'{hparams.output_path}/best_checkpoint_seg_renderer_valid_loss_{hparams.exp_name}', 
                                    ckpt_save_path_model2 = f'{hparams.output_path}/best_checkpoint_CUT_val_loss_{hparams.exp_name}', 
                                    verbose=True)
    

    train_losses, valid_losses, testset_losses, hausdorff_epoch, stoppcrit_losses, avg_train_losses, avg_valid_losses, avg_seg_pred_mean_diff = ([] for i in range(8))
    # G_train_losses, D_train_losses, D_real_train_losses, D_fake_train_losses = ([] for i in range(4))
    # G_val_losses, D_val_losses, D_real_val_losses, D_fake_val_losses = ([] for i in range(4))

    # --------------------
    # RUN TRAINING
    # ---------------------
    only_CUT = True
    only_SEGNET = True
    cut_trainer.cut_model.set_requires_grad(USRendereDefParams, False)
    for epoch in trange(1, hparams.max_epochs + 1):

        epoch_iter = 0 
        opt_cut.visualizer.reset() 
        iter_data_time = time.time()    # timer for data loading per iteration

        dataloader_real_us_iterator = iter(real_us_train_loader)
        step = 0
        module.train()
        module.outer_model.train()
        inner_model.train()

        # ------------------------------------------------------------------------------------------------------------------------------
        #                                         Train SEG NET only
        # ------------------------------------------------------------------------------------------------------------------------------
        if epoch <= hparams.epochs_only_seg_net and hparams.epochs_only_seg_net > 0:
            # print(f"--------------- INIT SEG NET ------------", cut_trainer.total_iter)
            training_loop_seg_net(hparams, module, inner_model, hparams.batch_size_manual, train_loader_ct_labelmaps, train_losses, plotter)

        # opt_cut.isTrain = True
        # ------------------------------------------------------------------------------------------------------------------------------
        #                                         Train CUT only
        # ------------------------------------------------------------------------------------------------------------------------------
        if epoch <= hparams.epochs_only_cut and hparams.epochs_only_cut > 0:
            for i, batch_data_ct in tqdm(enumerate(train_loader_ct_labelmaps), total=len(train_loader_ct_labelmaps), ncols= 100, position=0, leave=True):
                # print(f"--------------- INIT CUT ------------", cut_trainer.total_iter)
                cut_trainer.train_cut(batch_data_ct, epoch, dataloader_real_us_iterator, i, iter_data_time, epoch_iter)
    
        # ------------------------------------------------------------------------------------------------------------------------------
        #                                         Train US Renderer + SEG NET + CUT
        # ------------------------------------------------------------------------------------------------------------------------------
        if epoch > hparams.epochs_only_seg_net:
            if only_CUT: only_CUT = False
            if only_SEGNET: only_SEGNET = False
            step += 1

            batch_loss_list = []
            if hparams.batch_size_manual == 1:      # if batch==1
                for i, batch_data_ct in tqdm(enumerate(train_loader_ct_labelmaps), total=len(train_loader_ct_labelmaps), ncols= 100):   #, position=0, leave=True):
                    step += 1
                    module.optimizer.zero_grad()
                    input, label, filename = module.get_data(batch_data_ct)

                    if 1 in label:  #ONLY AORTA ONES
                        if hparams.use_idtB:
                            us_sim = module.rendering_forward(input)
                            us_sim_cut = us_sim.clone().detach()
                            cut_trainer.train_cut(module, us_sim_cut, epoch, dataloader_real_us_iterator, i, iter_data_time, epoch_iter)

                            # label = cut_trainer.forward_cut_B(us_sim, label)
                            cut_trainer.forward_cut_B(us_sim)
                            idt_B = cut_trainer.cut_model.idt_B
                            # idt_B = idt_B.clone()
                            idt_B = (idt_B / 2 ) + 0.5 # from [-1,1] to [0,1]
                            
                            if hparams.seg_net_input_augmentations_noise_blur or hparams.seg_net_input_augmentations_rand_crop:
                                idt_B, label = add_online_augmentations(hparams, idt_B, label)

                            loss, prediction = module.seg_forward(idt_B, label)

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
                            if hparams.seg_net_input_augmentations_noise_blur or hparams.seg_net_input_augmentations_rand_crop:
                                us_sim, label = add_online_augmentations(hparams, us_sim, label)
                            loss, prediction = module.seg_forward(us_sim, label)
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
                        if hparams.logging: wandb.log({"train_loss_step": loss.item()}, step=step)
                        train_losses.append(loss.item())
                        if hparams.logging: plotter.log_us_rendering_values(module.USRenderingModel, step)
                    

            else:       # if batch>1
                dataloader_iterator = iter(train_loader_ct_labelmaps)
                while step < len(train_loader_ct_labelmaps):
                    step += 1
                    module.optimizer.zero_grad()
                    # while step % hparams.batch_size_manual != 0 :
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
                    batch_loss_list =[]

                    print(f"{step}/{len(train_loader_ct_labelmaps.dataset) // train_loader_ct_labelmaps.batch_size}, train_loss: {loss.item():.4f}")
                    if hparams.logging: plotter.log_us_rendering_values(inner_model, step)

                    cut_trainer.train_cut(data, epoch, dataloader_real_us_iterator, i, iter_data_time, epoch_iter)


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

        
        print(f"--------------- ONLY CUT: {only_CUT} -------------- epoch: ", epoch)
        print(f"--------------- ONLY SEGNET: {only_SEGNET} -------------- epoch: ", epoch)
        
        # ------------------------------------------------------------------------------------------------------------------------------
        #                             SEG NET VALIDATION on every hparams.validate_every_n_steps
        # ------------------------------------------------------------------------------------------------------------------------------
        if epoch % hparams.validate_every_n_steps == 0 and epoch > hparams.epochs_only_cut:
            module.eval()
            module.outer_model.eval()
            inner_model.eval()
            cut_model.eval()
            USRendereDefParams.eval()
            val_step = 0
            print(f"--------------- VALIDATION SEG NET ------------")
            stopp_crit_plot_figs = []
            def_renderer_plot_figs = []
            reconstructed_seg_pred_nr_white_pxs = []
            rendered_seg_pred_nr_white_pxs = []
            wasserstein_distance_bw_rendered_reconstr = []
            wasserstein_distance_bw_rendered_idt_reconstr = []
            with torch.no_grad():
                for nr, val_batch_data_ct in tqdm(enumerate(val_loader_ct_labelmaps), total=len(val_loader_ct_labelmaps), ncols= 100):
                    
                    val_step += 1

                    val_input, val_label, filename = module.get_data(val_batch_data_ct)  

                    if 1 in val_label:
                        val_input_copy =  val_input.clone().detach()            
                        if hparams.use_idtB:
                            us_sim = module.rendering_forward(val_input)
                            
                            cut_trainer.forward_cut_B(us_sim)

                            idt_B_val = cut_trainer.cut_model.idt_B
                            # idt_B = idt_B.clone()
                            idt_B_val = (idt_B_val / 2 ) + 0.5 # from [-1,1] to [0,1]

                            # if hparams.debug and (hparams.seg_net_input_augmentations_noise_blur or hparams.seg_net_input_augmentations_rand_crop):
                            #     idt_B_val, val_label = add_online_augmentations(hparams, idt_B_val, val_label)

                            val_loss_step, rendered_seg_pred = module.seg_forward(idt_B_val, val_label)
                            if not hparams.log_default_renderer: dict = module.plot_val_results(val_input, val_loss_step, filename, val_label, rendered_seg_pred, idt_B_val, epoch)
                        
                        else:
                            us_sim = module.rendering_forward(val_input)
                            
                            if hparams.debug and (hparams.seg_net_input_augmentations_noise_blur or hparams.seg_net_input_augmentations_rand_crop):
                                us_sim, val_label = add_online_augmentations(hparams, us_sim, val_label)

                            val_loss_step, rendered_seg_pred = module.seg_forward(us_sim, val_label)
                            if not hparams.log_default_renderer: dict = module.plot_val_results(val_input, val_loss_step, filename, val_label, rendered_seg_pred, us_sim, epoch)

                        # rendered_seg_pred, us_sim, val_loss_step, dict = module.validation_step(val_batch_data_ct, epoch)
                        # print(f"{val_step}/{len(val_dataset_ct_labelmaps) // val_loader_ct_labelmaps.batch_size}, seg_val_loss: {val_loss.item():.4f}")
                        # if hparams.logging: wandb.log({"seg_val_loss_step": val_loss_step.item(), "step": val_step})

                        valid_losses.append(val_loss_step.item())

                        # USRendereDefParams.plot_fig(val_label, "val_label", True)


                        if hparams.log_default_renderer and nr < NR_IMGS_TO_PLOT:
                            us_sim_def = USRendereDefParams(val_input_copy.squeeze()) 
                            if not hparams.use_idtB: idt_B_val = us_sim
                            plot_fig = plotter.plot_stopp_crit(caption="default_renderer|labelmap|defaultUS|learnedUS|seg_input|seg_pred|gt",
                                                        imgs=[val_input, us_sim_def, us_sim, idt_B_val, rendered_seg_pred, val_label], 
                                                        img_text='', epoch=epoch, plot_single=False) #mean_diff=' + "{:.4f}".format(seg_pred_mean_diff.item()), )
                            def_renderer_plot_figs.append(plot_fig)
                        else:
                            plotter.validation_batch_end(dict)

                


        # ------------------------------------------------------------------------------------------------------------------------------
        #                             STOPPING CRITERIA CHECK - infer imgs from CUT train_set through the SEG NET
        # ------------------------------------------------------------------------------------------------------------------------------
                    if epoch > hparams.epochs_check_stopp_crit and hparams.plot_avg_seg_px_dff and hparams.pred_label in val_batch_data_ct[0]:
                        print(f"--------------- INFER UNLABELLED REAL US IMGS FROM CUT TRAIN SET THROUGH THE SEG NET ------------")
                        # cut_model.eval()
                        try:
                            data_cut_real_us = next(dataloader_real_us_iterator)
                        except StopIteration:
                            dataloader_real_us_iterator = iter(real_us_train_loader)
                            data_cut_real_us = next(dataloader_real_us_iterator)
                        
                        data_cut_real_us_domain_real = data_cut_real_us['A'].to(hparams.device)

                        cut_trainer.forward_cut_A(data_cut_real_us_domain_real)
                        reconstructed_us = cut_trainer.cut_model.fake_B 

                        # reconstructed_us = cut_model.netG(data_cut_real_us_domain_real)

                        reconstructed_us = (reconstructed_us / 2 ) + 0.5 # from [-1,1] to [0,1]
                        _, reconstructed_seg_pred  = module.seg_forward(reconstructed_us, reconstructed_us)

                        #calc the diff bw avg number of white pixels bw the seg predictions of rendered and recoconstructed 
                        # seg_pred_mean_diff = torch.abs(torch.mean(reconstructed_seg_pred) - torch.mean(rendered_seg_pred))
                        # avg_seg_pred_mean_diff.append(seg_pred_mean_diff.item())

                        reconstructed_seg_pred_nr_white_pxs.append(torch.sum(reconstructed_seg_pred))
                        rendered_seg_pred_nr_white_pxs.append(torch.sum(rendered_seg_pred))
                        wasserstein_distance_bw_rendered_reconstr.append(wasserstein_distance(us_sim.cpu().flatten(), reconstructed_us.cpu().flatten()))
                        if hparams.use_idtB: wasserstein_distance_bw_rendered_idt_reconstr.append(wasserstein_distance(idt_B_val.cpu().flatten(), reconstructed_us.cpu().flatten()))

                        if hparams.logging and nr < NR_IMGS_TO_PLOT:
                            # wandb.log({"seg_pred_mean_diff": seg_pred_mean_diff.item()})
                            no_random_transform = cut_trainer.inference_transformatons()
                            data_cut_real_us_domain_real = no_random_transform(data_cut_real_us_domain_real)
                            data_cut_real_us_domain_real = (data_cut_real_us_domain_real / 2 ) + 0.5 # from [-1,1] to [0,1]
                            plot_fig = plotter.plot_stopp_crit(caption="infer_CUT_during_val_|real_us|reconstructed_us|seg_pred_real|seg_pred_rendered|us_rendered",
                                                    imgs=[data_cut_real_us_domain_real, reconstructed_us, reconstructed_seg_pred, rendered_seg_pred, us_sim], 
                                                    img_text='', epoch=epoch, plot_single=False) #mean_diff=' + "{:.4f}".format(seg_pred_mean_diff.item()), )
                            stopp_crit_plot_figs.append(plot_fig)
                
                if len(stopp_crit_plot_figs) > 0: 
                    if hparams.logging:
                        plotter.log_image(torchvision.utils.make_grid(stopp_crit_plot_figs), "infer_CUT_during_val_|real_us|reconstructed_us|seg_pred_real|seg_pred_rendered|us_rendered")
                        # avg_seg_pred_mean_diff_mean = torch.mean(seg_pred_mean_diff)
                        # wandb.log({"seg_pred_mean_diff_epoch": avg_seg_pred_mean_diff_mean, "epoch": epoch})
                        # seg_pred_mean_diff = []
                        wandb.log({"reconstructed_seg_pred_nr_white_pxs_mean_epoch": torch.mean(torch.stack(reconstructed_seg_pred_nr_white_pxs)), "epoch": epoch})
                        wandb.log({"reconstructed_seg_pred_nr_white_pxs_std_epoch": torch.std(torch.stack(reconstructed_seg_pred_nr_white_pxs)), "epoch": epoch})

                        wandb.log({"rendered_seg_pred_nr_white_pxs_mean_epoch": torch.mean(torch.stack(rendered_seg_pred_nr_white_pxs)), "epoch": epoch})
                        wandb.log({"rendered_seg_pred_nr_white_pxs_std_epoch": torch.std(torch.stack(rendered_seg_pred_nr_white_pxs)), "epoch": epoch})

                        wandb.log({"kl_divergence_bw_nr_pxs_epoch": kl_div(torch.stack(reconstructed_seg_pred_nr_white_pxs), torch.stack(rendered_seg_pred_nr_white_pxs)), "epoch": epoch})
                        wandb.log({"wasserstein_distance_bw_nr_pxs_epoch": wasserstein_distance(torch.stack(reconstructed_seg_pred_nr_white_pxs).cpu(), torch.stack(rendered_seg_pred_nr_white_pxs).cpu()), "epoch": epoch})

                        wandb.log({"mean_wasserstein_distance_bw_rendered&reconstr_epoch": np.mean(wasserstein_distance_bw_rendered_reconstr), "epoch": epoch})
                        if hparams.use_idtB: wandb.log({"mean_wasserstein_distance_bw_rendered_idt&reconstr_epoch":  np.mean(wasserstein_distance_bw_rendered_idt_reconstr), "epoch": epoch})

                    stopp_crit_plot_figs = [] 
                    reconstructed_seg_pred_nr_white_pxs = []
                    rendered_seg_pred_nr_white_pxs = []
                    wasserstein_distance_bw_rendered_reconstr = []
                    wasserstein_distance_bw_rendered_idt_reconstr = []

                if len(def_renderer_plot_figs) > 0: 
                    if hparams.logging:
                        plotter.log_image(torchvision.utils.make_grid(def_renderer_plot_figs), "default_renderer|labelmap|defaultUS|learnedUS|seg_pred|gt")

            
            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            
            epoch_len = len(str(hparams.max_epochs))
            print(f'[{epoch:>{epoch_len}}/{hparams.max_epochs:>{epoch_len}}] ' +
                        f'seg_train_loss_epoch: {train_loss:.5f} ' +
                        f'seg_valid_loss_epoch: {valid_loss:.5f}')

            if hparams.logging: wandb.log({"seg_train_loss_epoch": train_loss, "epoch": epoch})
            if hparams.logging: wandb.log({"seg_val_loss_epoch": valid_loss, "epoch": epoch})
            if hparams.logging and not hparams.log_default_renderer: plotter.validation_epoch_end()

            if(epoch > 20): # hparams.min_epochs):
                early_stopping_best_val(valid_loss, epoch, module, cut_model.netG)
                
            # if early_stopping.early_stop and hparams.min_epochs:
            #     print("Early stopping")
            #     break

        # ------------------------------------------------------------------------------------------------------------------------------
        #                                      END SEG NETWORK VALIDATION
        # ------------------------------------------------------------------------------------------------------------------------------
            print(f"--------------- END SEG NETWORK VALIDATION ------------")
            cut_model.eval()
            # opt_cut.isTrain = False
            # opt_cut.phase = 'test'
            # opt_cut.gpu_ids = '-1'
            # run inference
        # ------------------------------------------------------------------------------------------------------------------------------
        #             (GT STOPPING CRITERION) INFER REAL US STOPPONG CRIT TEST SET IMGS THROUGH THE CUT+SEG NET 
        # -------------------------------------------------------------------------------------------------------------------------------
            avg_stopp_crit_loss = 1
            if hparams.stopping_crit_gt_imgs and epoch > hparams.epochs_check_stopp_crit:
                print(f"--------------- STOPPING CRITERIA US IMGS CHECK ------------")
                sopp_crit_imgs_plot_figs = []
                with torch.no_grad():
                    for nr, batch_data_real_us_stopp_crit in tqdm(enumerate(real_us_stopp_crit_dataloader), total=len(real_us_stopp_crit_dataloader), ncols= 100, position=0, leave=True):
                       
                        real_us_stopp_crit_img, real_us_stopp_crit_label = batch_data_real_us_stopp_crit[0].to(hparams.device), batch_data_real_us_stopp_crit[1].to(hparams.device).float()
                        reconstructed_us_stopp_crit = cut_model.netG(real_us_stopp_crit_img)

                        # reconstructed_us = transforms.functional.hflip(reconstructed_us_testset)
                        # real_us_test_img_label = transforms.functional.hflip(real_us_test_img_label)

                        reconstructed_us_stopp_crit = (reconstructed_us_stopp_crit / 2 ) + 0.5 # from [-1,1] to [0,1]

                        stoppcrit_loss, seg_pred_stopp_crit  = module.seg_forward(reconstructed_us_stopp_crit, real_us_stopp_crit_label)
                        # print(f"stop_criterion_loss: {testset_loss.item():.4f}")
                        # if hparams.logging: wandb.log({"stoppcrit_loss": stoppcrit_loss.item()})
                        stoppcrit_losses.append(stoppcrit_loss)

                        if hparams.logging:
                            real_us_stopp_crit_img = (real_us_stopp_crit_img / 2 ) + 0.5 # from [-1,1] to [0,1]

                            wandb.log({"stoppcrit_loss": stoppcrit_loss.item()})
                            plot_fig_gt = plotter.plot_stopp_crit(caption="stopp_crit|real_us|reconstructed_us|seg_pred|gt_label",
                                                    imgs=[real_us_stopp_crit_img, reconstructed_us_stopp_crit, seg_pred_stopp_crit, real_us_stopp_crit_label], 
                                                    img_text='loss=' + "{:.4f}".format(stoppcrit_loss.item()), epoch=epoch, plot_single=False)
                            sopp_crit_imgs_plot_figs.append(plot_fig_gt)
                
                if len(sopp_crit_imgs_plot_figs) > 0: 
                    plotter.log_image(torchvision.utils.make_grid(sopp_crit_imgs_plot_figs), "stopp_crit|real_us|reconstructed_us|seg_pred|gt_label")
                    avg_stopp_crit_loss = torch.mean(torch.stack(stoppcrit_losses))
                    wandb.log({"stoppcrit_loss_epoch": avg_stopp_crit_loss, "epoch": epoch})
                    stoppcrit_losses = []
                    sopp_crit_imgs_plot_figs = [] 
                

                # stop_criterion_loss_avg_epoch = np.average(stopp_crit_losses)
                # print(f"stop_criterion_loss_avg_epoch: {stop_criterion_loss_avg_epoch}")
                # if hparams.logging: wandb.log({"stop_criterion_loss_avg_epoch": stop_criterion_loss_avg_epoch})

                # early_stopping needs the validation loss to check if it has decresed, 
                # and if it has, it will make a checkpoint of the current model
                if(epoch > 20): # hparams.min_epochs):
                    early_stopping(avg_stopp_crit_loss, epoch, module, cut_model.netG)
                
                if early_stopping.early_stop and hparams.min_epochs:
                    print("Early stopping")
                    break
                stoppcrit_losses = []

        
        # ------------------------------------------------------------------------------------------------------------------------------
        #             INFER REAL US IMGS THROUGH THE CUT+SEG NET - INFER THE WHOLE TEST SET
        # ------------------------------------------------------------------------------------------------------------------------------
                avg_testset_loss = 1
                std_testset_loss = 0
                print(f"--------------- INFERENCE: WHOLE TEST US GT IMGS  ------------")
                gt_test_imgs_plot_figs = []

                with torch.no_grad():
                    for nr, batch_data_real_us_test in tqdm(enumerate(real_us_gt_test_dataloader), total=len(real_us_gt_test_dataloader), ncols= 100, position=0, leave=True):
                       
                        real_us_test_img, real_us_test_img_label = batch_data_real_us_test[0].to(hparams.device), batch_data_real_us_test[1].to(hparams.device).float()
                        reconstructed_us_testset = cut_model.netG(real_us_test_img)

                        # reconstructed_us = transforms.functional.hflip(reconstructed_us_testset)
                        # real_us_test_img_label = transforms.functional.hflip(real_us_test_img_label)

                        reconstructed_us_testset = (reconstructed_us_testset / 2 ) + 0.5 # from [-1,1] to [0,1]

                        # self.fake_B = reconstructed_us_testset[:self.real_A.size(0)] #???
                        # cut_model.forward()
                        # cut_model.compute_visuals()
                        # visuals = cut_model.get_current_visuals()  # get image results   for label, image in visuals.items():
                        # output_cut = visuals['fake_B']

                        testset_loss, seg_pred  = module.seg_forward(reconstructed_us_testset, real_us_test_img_label)
                        # print(f"stop_criterion_loss: {testset_loss.item():.4f}")
                        if hparams.logging: wandb.log({"testset_loss": testset_loss.item()})
                        testset_losses.append(testset_loss)

                        if hparams.logging and nr < NR_IMGS_TO_PLOT:
                            real_us_test_img = (real_us_test_img / 2 ) + 0.5 # from [-1,1] to [0,1]

                            wandb.log({"testset_loss": testset_loss.item()})

                            hausdorff_metric = HausdorffDistanceMetric()
                            pred_binary = torch.ge(seg_pred.data, 0.5).float()
                            hausdorff_dist = hausdorff_metric(y_pred=pred_binary, y=real_us_test_img_label)

                            wandb.log({"hausdorff_dist": hausdorff_dist.item()})
                            hausdorff_epoch.append(hausdorff_dist)

                            plot_fig_gt = plotter.plot_stopp_crit(caption="stestset_gt_|real_us|reconstructed_us|seg_pred|gt_label",
                                                    imgs=[real_us_test_img, reconstructed_us_testset, seg_pred, real_us_test_img_label], 
                                                    img_text='loss=' + "{:.4f}".format(testset_loss.item()), epoch=epoch, plot_single=False)
                            gt_test_imgs_plot_figs.append(plot_fig_gt)
                
                if len(gt_test_imgs_plot_figs) > 0: 
                    plotter.log_image(torchvision.utils.make_grid(gt_test_imgs_plot_figs), "testset_gt_|real_us|reconstructed_us|seg_pred|gt_label")
                    avg_testset_loss = torch.mean(torch.stack(testset_losses))
                    std_testset_loss = torch.std(torch.stack(testset_losses))
                    wandb.log({"testset_gt_loss_epoch": avg_testset_loss, "epoch": epoch})
                    wandb.log({"testset_gt_loss_std_epoch": std_testset_loss, "epoch": epoch})

                    avg_hausdorff_dist = torch.mean(torch.stack(hausdorff_epoch))
                    std_hausdorff = torch.std(torch.stack(hausdorff_epoch))
                    wandb.log({"tesset_hausdorff_dist_epoch": avg_hausdorff_dist, "epoch": epoch})
                    wandb.log({"testset_hausdorff_std_epoch": std_hausdorff, "epoch": epoch})
                    gt_test_imgs_plot_figs = [] 

                
                testset_losses = []
                hausdorff_epoch = []


            
            # clear lists to track next epoch
            train_losses = []
            valid_losses = []

    


        
    print(f'train completed, avg_train_losses: {avg_train_losses} avg_valid_losses: {avg_valid_losses}')
    print(f'best val_loss: {early_stopping.val_loss_min} at best_epoch: {early_stopping.best_epoch}')




# load the last checkpoint with the best model
# model.load_state_dict(torch.load('checkpoint.pt'))