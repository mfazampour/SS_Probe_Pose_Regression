from collections import OrderedDict
import logging
import sys
import numpy as np
import torch
import time
import wandb
# from tensorboardX import SummaryWriter
import monai
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImage,
    RandRotate90,
    RandSpatialCrop,
    ScaleIntensity,
)
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

MANUAL_SEED = False
EPOCHS_INIT_CUT = 5

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
        if epoch == opt_cut.epoch_count and self.init_cut:    #initialize only on epoch 1 
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


# def log_model_gradients(model, step):
#     for tag, value in model.named_parameters():
#         if value.grad is not None:
#             tb_logger.add_histogram(tag + "/grad", value.grad.cpu(), step)


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
    OuterModelClass = hparams.outer_model

    cut_model = cut_create_model(opt_cut)      # create a model given opt.model and other options

    inner_model = InnerModelClass()
    module = ModuleClass(hparams, OuterModelClass, inner_model)

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
    

    train_losses, valid_losses, avg_train_losses, avg_valid_losses = ([] for i in range(4))
    # G_train_losses, D_train_losses, D_real_train_losses, D_fake_train_losses = ([] for i in range(4))
    # G_val_losses, D_val_losses, D_real_val_losses, D_fake_val_losses = ([] for i in range(4))

    # ---------------------
    # RUN TRAINING
    # --------------------
    for epoch in range(1, hparams.max_epochs + 1):

        epoch_iter = 0 
        opt_cut.visualizer.reset() 
        iter_data_time = time.time()    # timer for data loading per iteration

        dataloader_real_us_iterator = iter(real_us_train_loader)
        step = 0
        for i, batch_data_ct in enumerate(train_loader_ct_labelmaps):
            # opt_cut.isTrain = True
            if epoch < EPOCHS_INIT_CUT:
                print(f"--------------- INIT CUT ------------", cut_trainer.total_iter)
                #init CUT
                cut_trainer.train_cut(batch_data_ct, epoch, dataloader_real_us_iterator, i, iter_data_time, epoch_iter)

            else:
                #Train US Renderer + SEG NET
                module.train()
                module.outer_model.train()
                inner_model.train()
                # step = 0
                # for batch_data_ct in train_loader:
                step += 1

                train_seg_loss, us_rendered = module.training_step(batch_data_ct)
                train_seg_loss.backward()
                # log_model_gradients(inner_model, step)
                module.optimizer.step()

                cut_trainer.train_cut(batch_data_ct, epoch, dataloader_real_us_iterator, i, iter_data_time, epoch_iter)

                print(f"{step}/{len(train_dataset_ct_labelmaps) // train_loader_ct_labelmaps.batch_size}, seg_train_loss: {train_seg_loss.item():.4f}")
                if hparams.logging: wandb.log({"seg_train_loss_step": train_seg_loss.item()}, step=step)

                train_losses.append(train_seg_loss.item())

                
        # ---------------------
        # SEG NET VALIDATION
        # --------------------v
        if epoch % hparams.validate_every_n_steps == 0 and epoch > EPOCHS_INIT_CUT:
            module.eval()
            module.outer_model.eval()
            inner_model.eval()
            val_step = 0
            with torch.no_grad():
                for val_batch_data_ct in val_loader_ct_labelmaps:
                    val_step += 1

                    val_loss, dict = module.validation_step(val_batch_data_ct, epoch)
                    print(f"{val_step}/{len(val_dataset_ct_labelmaps) // val_loader_ct_labelmaps.batch_size}, seg_val_loss: {val_loss.item():.4f}")
                    if hparams.logging: wandb.log({"seg_val_loss_step": val_loss.item()})

                    valid_losses.append(val_loss.item())
                    plotter.validation_batch_end(dict)
            
            #STOPPING CRITERION for the FULL Training - infere us_real imgs through the seg net
            cut_model.eval()
            # opt_cut.isTrain = False
            # opt_cut.phase = 'test'
            # opt_cut.gpu_ids = '-1'
            # run inference


            #get real_us_test_img and real_us_test_img_label
            with torch.no_grad():
                for batch_data_real_us_test in real_us_stopp_crit_test_dataloader:
                    real_us_test_img, real_us_test_img_label = batch_data_real_us_test[0].to(hparams.device), batch_data_real_us_test[1].to(hparams.device)
                    # real_us_test_img, real_us_test_img_label = batch_data_real_us_test[0].to('cpu'), batch_data_real_us_test[1].to('cpu')
                    reconstructed_us = cut_model.netG(real_us_test_img) #.to('cpu')
                    # self.fake_B = reconstructed_us[:self.real_A.size(0)] #???
                    # cut_model.forward()
                    # cut_model.compute_visuals()
                    # visuals = cut_model.get_current_visuals()  # get image results   for label, image in visuals.items():
                    # output_cut = visuals['fake_B']

                    stop_criterion_loss, seg_pred  = module.seg_net_forward(reconstructed_us, real_us_test_img_label)
                    print(f"stop_criterion_loss: {stop_criterion_loss.item():.4f}")
                    if hparams.logging: wandb.log({"stop_criterion_loss": stop_criterion_loss.item()})
                    inference_plot = {f'real_us': (real_us_test_img.detach()),
                                    f'reconstructed_us': reconstructed_us.detach(),
                                    f'seg_pred': seg_pred.detach(),
                                    f'gt_label': real_us_test_img_label.detach(),
                                    }
                    plotter.plot_images(inference_plot, epoch, wandb)



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
            
            print(f"--------------- END VAL ------------")
            
            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            if(epoch > hparams.min_epochs):
                early_stopping(stop_criterion_loss, epoch, module)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            # clear lists to track next epoch
            train_losses = []
            valid_losses = []


        
    print(f'train completed, avg_train_losses: {avg_train_losses:.4f} avg_valid_losses: {avg_valid_losses}' +
        f'best val_loss: {early_stopping.val_loss_min} at best_epoch: {early_stopping.best_epoch}')






















# load the last checkpoint with the best model
# model.load_state_dict(torch.load('checkpoint.pt'))








            # log_model_gradients(inner_model, step)

            # print(f'{step}/{len(train_dataset_ct_labelmaps) // train_loader_ct_labelmaps.batch_size}, train_loss: {total_loss.item():.4f}, loss_G: {loss_G.item():.4f}, loss_D: {loss_D.item():.4f}, loss_D_real: {loss_D_real.item():.4f}, loss_D_fake: {loss_D_fake.item():.4f}')

            # if hparams.logging: 
            #     wandb.log({"train_seg_loss_step": total_loss.item()}, step=step, commit=True)#)
            #     wandb.log({"train_loss_G_step": loss_G.item()}, commit=False)# step=step)
            #     wandb.log({"train_loss_D_total_step": loss_D.item()}, commit=False)# step=step)
            #     wandb.log({"train_loss_D_fake_step": loss_D_real.item()}, commit=False)# step=step)
            #     wandb.log({"train_loss_D_real_step": loss_D_fake.item()}, commit=False)# step=step)
            #     wandb.log({"step":step})

            #     us_rendering_values = [inner_model.acoustic_impedance_dict, inner_model.attenuation_dict, 
            #                             inner_model.mu_0_dict, inner_model.mu_1_dict, inner_model.sigma_0_dict]
            #     us_rendering_maps = ['acoustic_imp_', 'atten_', 'mu_0_', 'mu_1_', 'sigma_0_' ]
            #     for map, dict in zip(us_rendering_maps, us_rendering_values):
            #         for label, value in zip(inner_model.labels, dict):
            #             wandb.log({map+label: value}, commit=False)
            #         wandb.log({"step":step})


                    
                # for l in us_rendering_values:
                #     for label, value in zip(inner_model.labels, l):
                #         wandb.log({label: value}, commit=False)
                #     wandb.log({"step":step})



            # train_losses.append(total_loss.item())
            # G_train_losses.append(loss_G.item())
            # D_train_losses.append(loss_D.item())
            # D_fake_train_losses.append(loss_D_fake.item())
            # D_real_train_losses.append(loss_D_real.item())


        # if (epoch + 1) % hparams.validate_every_n_steps == 0:
        #     print("--------------------VALIDATION--------------------------")
        #     dataloader_real_us_val_iterator = iter(real_us_val_loader)

        #     module.outer_model.eval()
        #     inner_model.eval()
        #     val_step = 0
        #     with torch.no_grad():
        #         for val_batch in val_loader_ct_labelmaps:
        #             try:
        #                 data2_val = next(dataloader_real_us_val_iterator)
        #             except StopIteration:
        #                 dataloader_real_us_val_iterator = iter(real_us_val_loader)
        #                 data2_val = next(dataloader_real_us_val_iterator)

        #             val_step += 1

        #             # val_loss, dict = module.validation_step(val_batch, epoch)
        #             val_loss, val_loss_G, val_loss_D, val_loss_D_real, val_loss_D_fake, dict = module.validation_step(epoch=epoch, batch_data_ct=val_batch, batch_data_real_us=data2_val.to(hparams.device))
                    


        #             # print(f"{val_step}/{len(val_dataset) // val_loader.batch_size}, val_loss: {val_loss.item():.4f}")
        #             print(f'{val_step}/{len(val_dataset_ct_labelmaps) // val_loader_ct_labelmaps.batch_size}, val_loss: {val_loss.item():.4f}, loss_G: {val_loss_G.item():.4f}, loss_D: {val_loss_D.item():.4f}, loss_D_real: {val_loss_D_real.item():.4f}, loss_D_fake: {val_loss_D_fake.item():.4f}')

        #             if hparams.logging: 
        #                 wandb.log({"val_loss_step": val_loss.item()}, step=val_step)
        #                 wandb.log({"val_loss_G_step": val_loss_G.item()}, step=val_step)
        #                 wandb.log({"val_loss_D_total_step": val_loss_D.item()}, step=val_step)
        #                 wandb.log({"val_loss_D_fake_step": val_loss_D_real.item()}, step=val_step)
        #                 wandb.log({"val_loss_D_real_step": val_loss_D_fake.item()}, step=val_step)


        #             valid_losses.append(val_loss.item())
        #             plotter.validation_batch_end(dict)
                    
        #             G_val_losses.append(val_loss_G.item())
        #             D_val_losses.append(val_loss_D.item())
        #             D_fake_val_losses.append(val_loss_D_fake.item())
        #             D_real_val_losses.append(val_loss_D_real.item())


        # train_loss = np.average(train_losses)
        # valid_loss = np.average(valid_losses)
        # avg_train_losses.append(train_loss)
        # avg_valid_losses.append(valid_loss)

        # G_train_loss = np.average(G_train_losses)
        # D_train_loss = np.average(D_train_losses)
        # D_real_train_loss = np.average(D_real_train_losses)
        # D_fake_train_loss = np.average(D_fake_train_losses)

        # G_val_loss = np.average(G_val_losses)
        # D_val_loss = np.average(D_val_losses)
        # D_real_val_loss = np.average(D_real_val_losses)
        # D_fake_val_loss = np.average(D_fake_val_losses)
        
        # # module.seg_scheduler.step(valid_loss)
        # # module.discr_scheduler.step(D_train_loss)
        # seg_optimizer_lr = module.seg_optimizer.param_groups[0]['lr']
        # optimizer_D_lr = module.optimizer_D.param_groups[0]['lr']
        # print('learning rate seg_optimizer = %.7f' % seg_optimizer_lr)
        # print('learning rate optimizer_D = %.7f' % optimizer_D_lr)
        # # calculate average loss over an epoch
        # epoch_len = len(str(hparams.max_epochs))
        # print(f'[{epoch:>{epoch_len}}/{hparams.max_epochs:>{epoch_len}}] ' +
        #              f'train_loss_epoch: {train_loss:.5f} ' +
        #              f'G_train_loss_epoch: {G_train_loss:.5f} ' +
        #              f'D_train_loss_epoch: {D_train_loss:.5f} ' +
        #              f'D_real_train_loss_epoch: {D_real_train_loss:.5f} ' +
        #              f'D_fake_train_loss_epoch: {D_fake_train_loss:.5f} ' +
        #              f'valid_loss_epoch: {valid_loss:.5f}'  +
        #              f'G_val_loss_epoch: {G_val_loss:.5f}'  +
        #              f'D_val_loss_epoch: {D_val_loss:.5f}'  +
        #              f'D_real_val_loss_epoch: {D_real_val_loss:.5f}'  +
        #              f'D_fake_val_loss_epoch: {D_fake_val_loss:.5f}' 
        #              )

        # if hparams.logging: 
        #     wandb.log({"train_loss_epoch": train_loss, "epoch": epoch})
        #     wandb.log({"G_train_loss_epoch": G_train_loss, "epoch": epoch})
        #     wandb.log({"D_train_loss_epoch": D_train_loss, "epoch": epoch})
        #     wandb.log({"D_real_train_loss_epoch": D_real_train_loss, "epoch": epoch})
        #     wandb.log({"D_fake_train_loss_epoch": D_fake_train_loss, "epoch": epoch})
        #     wandb.log({"seg_optimizer_lr": seg_optimizer_lr, "epoch": epoch})
        #     wandb.log({"optimizer_D_lr": optimizer_D_lr, "epoch": epoch})

        # if hparams.logging: 
        #     wandb.log({"val_loss_epoch": valid_loss, "epoch": epoch})
        #     wandb.log({"G_val_loss_epoch": G_val_loss, "epoch": epoch})
        #     wandb.log({"D_val_loss_epoch": D_val_loss, "epoch": epoch})
        #     wandb.log({"D_real_val_loss_epoch": D_real_val_loss, "epoch": epoch})
        #     wandb.log({"D_fake_val_loss_epoch": D_fake_val_loss, "epoch": epoch})

        # if hparams.logging: plotter.validation_epoch_end()
        # print(f"--------------- END VAL ------------")
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        # if(epoch > hparams.min_epochs):
        #     early_stopping(valid_loss, epoch, module)
        
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break
        
        # # clear lists to track next epoch
        # train_losses = []
        # valid_losses = []
