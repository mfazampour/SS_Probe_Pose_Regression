from collections import OrderedDict
import logging
import sys
import numpy as np
import torch
import time
import wandb
from tqdm.auto import tqdm, trange
from time import sleep

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
from full_run_torch import training_loop_seg_net

MANUAL_SEED = False
EPOCHS_ONLY_CUT = 0
EPOCHS_ONLY_SEG_NET = 0

# torch.use_deterministic_algorithms(True, warn_only=True)
# tb_logger = SummaryWriter('log_dir/cactuss_end2end')

# Define the desired transforms
# transform_real_us_data = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

class CUTTrainer():
    def __init__(self, cut_model, inner_model, dataset_real_us, real_us_train_loader):
        super(CUTTrainer, self).__init__()
        self.t_data = 0 
        self.total_iter = 0 
        self.optimize_time = 0.1
        self.cut_model = cut_model
        self.inner_model = inner_model
        self.dataset_real_us = dataset_real_us
        self.real_us_train_loader = real_us_train_loader
        self.init_cut = True


    def train_cut(self, batch_data_ct, epoch, dataloader_real_us_iterator, i, iter_data_time, epoch_iter):
        # with torch.autograd.detect_anomaly():
        
        # us_rendered = module.us_rendering_forward(batch_data_ct=batch_data_ct)   # just getting the image, no gradients
        ct_slice = batch_data_ct[0].to(hparams.device)
        us_sim = self.inner_model(ct_slice.squeeze()) 
        us_sim_resized = F.resize(us_sim.unsqueeze(0).unsqueeze(0), (256, 256)).float().detach()
        # us_sim_resized = F.resize(us_sim.unsqueeze(0).unsqueeze(0), (256, 256)).float()   #dont detach if we want end to end???
        # # transform = T.ToPILImage()
        # us_rendered = transform(us_sim).convert('RGB')

        try:
            data_cut_real_us = next(dataloader_real_us_iterator)
        except StopIteration:
            dataloader_real_us_iterator = iter(self.real_us_train_loader)
            data_cut_real_us = next(dataloader_real_us_iterator)
        
        data_cut_real_us["A"] = data_cut_real_us["A"].to(hparams.device)
        real_us_batch_size = data_cut_real_us["A"].size(0)
        self.total_iter += real_us_batch_size
        epoch_iter += real_us_batch_size

        transform_B = cut_get_transform(opt_cut, grayscale=False, convert=False)    #it is already grayscale and tensor
        data_cut_rendered_us = transform_B(us_sim_resized)
        # data_cut_rendered_us = data_cut_rendered_us.unsqueeze(0)
        data_cut_real_us['B'] = data_cut_rendered_us   # add cut_model.real_B
        
        optimize_start_time = time.time()
        # if epoch == opt_cut.epoch_count and self.init_cut:    #initialize only on epoch 1 
        if self.init_cut:    #initialize cut
            print(f"--------------- INIT CUT --------------")
            self.init_cut = False
            self.cut_model.data_dependent_initialize(data_cut_real_us)
            self.cut_model.setup(opt_cut)               # regular setup: load and print networks; create schedulers
            self.cut_model.parallelize()

        self.cut_model.set_input(data_cut_real_us)  # unpack data from dataset and apply preprocessing
        self.cut_model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
        self.optimize_time = (time.time() - optimize_start_time) / real_us_batch_size * 0.005 + 0.995 * self.optimize_time

        iter_start_time = time.time() 
        if self.total_iter % opt_cut.print_freq == 0:
            self.t_data = iter_start_time - iter_data_time

        #Print CUT results
        if self.total_iter % opt_cut.display_freq == 0:   # display images and losses on wandb
            self.cut_model.compute_visuals()
            # visualizer.display_current_results(self.cut_model.get_current_visuals(), epoch, None, wandb)
            plotter.plot_images(self.cut_model.get_current_visuals(), epoch, wandb)
            losses = self.cut_model.get_current_losses()
            opt_cut.visualizer.print_current_losses(epoch, epoch_iter, losses, self.optimize_time, self.t_data, wandb)




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

    hparams.exp_name += str(hparams.pred_label) + '_' + str(opt_cut.lr) + '_' + str(hparams.inner_model_learning_rate) + '_' + str(hparams.outer_model_learning_rate)

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

        hparams.output_path = get_outputs_path()
        poly_experiment_info = Experiment.get_experiment_info()
        poly_experiment_nr = poly_experiment_info['experiment_name'].split(".")[-1]
        hparams.name = poly_experiment_nr + "_" + hparams.exp_name
        print(f'get_outputs_path: {hparams.output_path} \n '
              f'experiment_info: {poly_experiment_info} \n experiment_nr: {poly_experiment_nr}')


    # ---------------------
    # LOAD MODEL
    # ---------------------
    ModuleClass = load_module(hparams.module)
    InnerModelClass = load_model(hparams.inner_model)
    OuterModelClass = load_model(hparams.outer_model)
    # OuterModelClass = hparams.outer_model

    cut_model = cut_create_model(opt_cut)      # create a model given opt.model and other options

    inner_model = InnerModelClass()
    outer_model = OuterModelClass(hparams=hparams)

    module = ModuleClass(hparams, outer_model, inner_model)

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

    real_us_gt_dataset = RealUSGTDatasetClass(root_dir=hparams.data_dir_real_us_test)
    real_us_stopp_crit_test_dataloader = torch.utils.data.DataLoader(real_us_gt_dataset, shuffle=False)


    cut_trainer = CUTTrainer(cut_model, inner_model, dataset_real_us, real_us_train_loader)
    early_stopping = EarlyStopping(patience=hparams.early_stopping_patience, 
                                    ckpt_save_path = f'{hparams.output_path}/best_checkpoint_{hparams.exp_name}.pt', verbose=True)
    

    train_losses, valid_losses, stopp_crit_losses, avg_train_losses, avg_valid_losses = ([] for i in range(5))
    # G_train_losses, D_train_losses, D_real_train_losses, D_fake_train_losses = ([] for i in range(4))
    # G_val_losses, D_val_losses, D_real_val_losses, D_fake_val_losses = ([] for i in range(4))

    # --------------------
    # RUN TRAINING
    # ---------------------
    only_CUT = True
    only_SEGNET = True
    for epoch in trange(1, hparams.max_epochs + 1):

        epoch_iter = 0 
        opt_cut.visualizer.reset() 
        iter_data_time = time.time()    # timer for data loading per iteration

        dataloader_real_us_iterator = iter(real_us_train_loader)
        step = 0
        module.train()
        module.outer_model.train()
        inner_model.train()

        if epoch <= EPOCHS_ONLY_SEG_NET and EPOCHS_ONLY_SEG_NET > 0:
            # print(f"--------------- INIT SEG NET ------------", cut_trainer.total_iter)
            #train SEG NET only
            training_loop_seg_net(hparams, module, inner_model, hparams.batch_size_manual, train_loader_ct_labelmaps, train_losses, plotter)

        # opt_cut.isTrain = True
        if epoch <= EPOCHS_ONLY_CUT and EPOCHS_ONLY_CUT > 0:
            for i, batch_data_ct in tqdm(enumerate(train_loader_ct_labelmaps), total=len(train_loader_ct_labelmaps), ncols= 100, position=0, leave=True):
                # print(f"--------------- INIT CUT ------------", cut_trainer.total_iter)
                #train CUT only
                cut_trainer.train_cut(batch_data_ct, epoch, dataloader_real_us_iterator, i, iter_data_time, epoch_iter)
    
        if epoch > EPOCHS_ONLY_SEG_NET:
            # print(f"--------------- TRAIN JOINTLY SEG NET+CUT ------------", cut_trainer.total_iter)
            if only_CUT: only_CUT = False
            if only_SEGNET: only_SEGNET = False
            # print(f"--------------- Train US Renderer + SEG NET + CUT -------------- epoch: ", epoch)
            #Train US Renderer + SEG NET
            step += 1

            batch_loss_list =[]
            if hparams.batch_size_manual == 1:
                # for batch_data in train_loader:
                for i, batch_data_ct in tqdm(enumerate(train_loader_ct_labelmaps), total=len(train_loader_ct_labelmaps), ncols= 100):   #, position=0, leave=True):
                    step += 1
                    module.optimizer.zero_grad()
                    input, label = module.get_data(batch_data_ct)
                    loss, us_sim, prediction = module.step(input, label)
                    # check_gradients(module)
                    loss.backward()
                    # log_model_gradients(inner_model, step)
                    module.optimizer.step()

                    # print(f"{step}/{len(train_loader_ct_labelmaps.dataset) // train_loader_ct_labelmaps.batch_size}, train_loss: {loss.item():.4f}")
                    if hparams.logging: wandb.log({"train_loss_step": loss.item()}, step=step)
                    train_losses.append(loss.item())
                    if hparams.logging: plotter.log_us_rendering_values(inner_model, step)
                    
                    cut_trainer.train_cut(batch_data_ct, epoch, dataloader_real_us_iterator, i, iter_data_time, epoch_iter)

            else:
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




                # train_seg_loss, us_rendered = module.training_step(batch_data_ct)
                # train_seg_loss.backward()
                # # log_model_gradients(inner_model, step)
                # module.optimizer.step()

                # print(f"{step}/{len(train_dataset_ct_labelmaps) // train_loader_ct_labelmaps.batch_size}, seg_train_loss: {train_seg_loss.item():.4f}")
                # if hparams.logging: wandb.log({"seg_train_loss_step": train_seg_loss.item()}, step=step)
                # train_losses.append(train_seg_loss.item())

                # acoustic_impedance_dict_before_cut = inner_model.acoustic_impedance_dict
                # plotter.log_us_rendering_values(inner_model, step)

                # cut_trainer.train_cut(batch_data_ct, epoch, dataloader_real_us_iterator, i, iter_data_time, epoch_iter)
                
                # if not torch.equal(acoustic_impedance_dict_before_cut, inner_model.acoustic_impedance_dict):
                #     print(f"--------------- US PARAMS CHANGED --------------")

        
        print(f"--------------- ONLY CUT: {only_CUT} -------------- epoch: ", epoch)
        print(f"---------------ONLLY SEGNET: {only_SEGNET} -------------- epoch: ", epoch)
        
        # ---------------------
        # SEG NET VALIDATION
        # ---------------------
        if epoch % hparams.validate_every_n_steps == 0 and epoch > EPOCHS_ONLY_CUT:
            module.eval()
            module.outer_model.eval()
            inner_model.eval()
            val_step = 0
            # VALIDATION only SEG NET
            print(f"--------------- VALIDATION SEG NET ------------")
            with torch.no_grad():
                for val_batch_data_ct in tqdm(val_loader_ct_labelmaps, total=len(val_loader_ct_labelmaps), ncols= 100):
                    val_step += 1

                    rendered_seg_pred, us_sim, val_loss_step, dict = module.validation_step(val_batch_data_ct, epoch)
                    # print(f"{val_step}/{len(val_dataset_ct_labelmaps) // val_loader_ct_labelmaps.batch_size}, seg_val_loss: {val_loss.item():.4f}")
                    if hparams.logging: wandb.log({"seg_val_loss_step": val_loss_step.item()})

                    valid_losses.append(val_loss_step.item())
                    plotter.validation_batch_end(dict)

                    if epoch> 10 and hparams.plot_avg_seg_px_dff and hparams.pred_label in val_batch_data_ct[0]:
                        print(f"--------------- INFER US IMGS THROUGH SEG NET ------------")
                        cut_model.eval()
                        try:
                            data_cut_real_us = next(dataloader_real_us_iterator)
                        except StopIteration:
                            dataloader_real_us_iterator = iter(real_us_train_loader)
                            data_cut_real_us = next(dataloader_real_us_iterator)
                        
                        reconstructed_us = cut_model.netG(data_cut_real_us['A']) #.to('cpu')

                        reconstructed_us = (reconstructed_us / 2 ) + 0.5 # from [-1,1] to [0,1]
                        _, reconstructed_seg_pred  = module.seg_net_forward(reconstructed_us, reconstructed_us)
                        #calc the diff bw avg number of white pixels bw the seg predictions of rendered and recoconstructed 

                        seg_pred_mean_diff = torch.abs(torch.mean(reconstructed_seg_pred) - torch.mean(rendered_seg_pred))
                        print(f"seg_pred_mean_diff: {seg_pred_mean_diff.item():.4f}")

                        if hparams.logging: 
                            wandb.log({"seg_pred_mean_diff": seg_pred_mean_diff.item()})
                            plot_fig = plotter.plot_stopp_crit(caption="stopp_crit_|real_us|reconstructed_us|seg_pred_real|seg_pred_rendered|us_rendered",
                                                    imgs=[data_cut_real_us['A'], reconstructed_us, reconstructed_seg_pred, rendered_seg_pred, us_sim], 
                                                    img_text='mean_diff=' + "{:.4f}".format(seg_pred_mean_diff.item()), epoch=epoch)



            
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

            if hparams.logging: plotter.validation_epoch_end()






            
            print(f"--------------- END SEG NETWORK VALIDATION ------------")

            #STOPPING CRITERION for the FULL Training - infere us_real imgs through the seg net
            cut_model.eval()
            # opt_cut.isTrain = False
            # opt_cut.phase = 'test'
            # opt_cut.gpu_ids = '-1'
            # run inference

            if hparams.stopping_crit_gt_imgs:
                print(f"--------------- STOPPING CRITERIA CHECK ------------")
                with torch.no_grad():
                    for nr, batch_data_real_us_test in tqdm(enumerate(real_us_stopp_crit_test_dataloader), total=len(real_us_stopp_crit_test_dataloader), ncols= 100, position=0, leave=True):
                        real_us_test_img, real_us_test_img_label = batch_data_real_us_test[0].to(hparams.device), batch_data_real_us_test[1].to(hparams.device).float()
                        # real_us_test_img, real_us_test_img_label = batch_data_real_us_test[0].to('cpu'), batch_data_real_us_test[1].to('cpu')
                        reconstructed_us = cut_model.netG(real_us_test_img) #.to('cpu')
                        reconstructed_us = transforms.functional.hflip(reconstructed_us)
                        real_us_test_img_label = transforms.functional.hflip(real_us_test_img_label)

                        reconstructed_us = (reconstructed_us / 2 ) + 0.5 # from [-1,1] to [0,1]

                        # self.fake_B = reconstructed_us[:self.real_A.size(0)] #???
                        # cut_model.forward()
                        # cut_model.compute_visuals()
                        # visuals = cut_model.get_current_visuals()  # get image results   for label, image in visuals.items():
                        # output_cut = visuals['fake_B']

                        stop_criterion_loss, seg_pred  = module.seg_net_forward(reconstructed_us, real_us_test_img_label)
                        print(f"stop_criterion_loss: {stop_criterion_loss.item():.4f}")
                        if hparams.logging: wandb.log({"stop_criterion_loss": stop_criterion_loss.item()})
                        stopp_crit_losses.append(stop_criterion_loss.item())

                        # inference_plot = {f'real_us': (real_us_test_img.detach()),
                        #                 f'reconstructed_us': reconstructed_us.detach(),
                        #                 f'seg_pred': seg_pred.detach(),
                        #                 f'gt_label': real_us_test_img_label.detach(),
                        #                 }
                        
                        if nr <= 10:     # log only the first 10  
                            plot_fig = plotter.plot_stopp_crit(caption="stopp_crit_|real_us|reconstructed_us|seg_pred|gt_label",
                                                    imgs=[real_us_test_img, reconstructed_us, seg_pred, real_us_test_img_label], 
                                                    img_text='loss=' + "{:.4f}".format(stop_criterion_loss.item()), epoch=epoch)

                        # plotter.plot_images(inference_plot, epoch, wandb)
                    

                stop_criterion_loss_avg_epoch = np.average(stopp_crit_losses)
                print(f"stop_criterion_loss_avg_epoch: {stop_criterion_loss_avg_epoch}")
                if hparams.logging: wandb.log({"stop_criterion_loss_avg_epoch": stop_criterion_loss_avg_epoch})

                # early_stopping needs the validation loss to check if it has decresed, 
                # and if it has, it will make a checkpoint of the current model
                if(epoch > hparams.min_epochs):
                    early_stopping(stop_criterion_loss_avg_epoch, epoch, module)
                
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                stopp_crit_losses = []

            
            # clear lists to track next epoch
            train_losses = []
            valid_losses = []

    


        
    print(f'train completed, avg_train_losses: {avg_train_losses:.4f} avg_valid_losses: {avg_valid_losses}')
    print(f'best val_loss: {early_stopping.val_loss_min} at best_epoch: {early_stopping.best_epoch}')




# load the last checkpoint with the best model
# model.load_state_dict(torch.load('checkpoint.pt'))