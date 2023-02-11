import logging
import sys
import numpy as np
import torch
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
from cut.models.networks import define_D
 
from cut.options.train_options import TrainOptions as TrainOptionsCUT
import torchvision.transforms as transforms


MANUAL_SEED = False
torch.use_deterministic_algorithms(True, warn_only=True)
# COMMENT = f"################ UNET vessel segm ############################"
# tb_logger = SummaryWriter('log_dir/cactuss_end2end')

# Define the desired transforms
transform_real_us_data = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def train(hparams, opt_cut, ModuleClass, OuterModelClass, InnerModelClass, DatasetLoader, DatasetLoaderDiscr):

    if MANUAL_SEED: torch.manual_seed(2023)

    inner_model = InnerModelClass()
    discr_model = define_D(opt_cut.output_nc, opt_cut.ndf, opt_cut.netD, opt_cut.n_layers_D, 
                        opt_cut.normD, opt_cut.init_type, opt_cut.init_gain, opt_cut.no_antialias, 
                        opt_cut.gpu_ids, opt_cut)


    module = ModuleClass(hparams, opt_cut, OuterModelClass, inner_model, discr_model)

    # wandb.watch(inner_model, log='all', log_graph=True, log_freq=10)
    # wandb.watch([inner_model, outer_model], log='all', log_graph=True, log_freq=100)


    dataloader = DatasetLoader(hparams)
    train_loader_ct_labelmaps, train_dataset_ct_labelmaps, val_dataset_ct_labelmaps  = dataloader.train_dataloader()
    val_loader_ct_labelmaps = dataloader.val_dataloader()

    # dataset_discr = cut_create_dataset(opt_cut)  # create a dataset given opt.dataset_mode and other options
    discr_dataloader = DatasetLoaderDiscr(param=hparams, root_dir=hparams.data_dir_real_us, batch_size=opt_cut.batch_size, transform=transform_real_us_data)
    discr_train_loader, discr_val_loader = discr_dataloader.get_dataloaders()

    early_stopping = EarlyStopping(patience=hparams.early_stopping_patience, 
                                    ckpt_save_path = f'{hparams.output_path}/best_checkpoint_{hparams.exp_name}.pt', verbose=True)

    train_losses, valid_losses, avg_train_losses, avg_valid_losses = ([] for i in range(4))
    G_train_losses, D_train_losses, D_real_train_losses, D_fake_train_losses = ([] for i in range(4))
    G_val_losses, D_val_losses, D_real_val_losses, D_fake_val_losses = ([] for i in range(4))

    for epoch in range(1, hparams.max_epochs + 1):
        dataloader_discr_iterator = iter(discr_train_loader)

        module.outer_model.train()
        inner_model.train()
        # discr_model.train()
        step = 0
        for batch_data_ct in train_loader_ct_labelmaps:

            try:
                data2 = next(dataloader_discr_iterator)
            except StopIteration:
                dataloader_discr_iterator = iter(discr_train_loader)
                data2 = next(dataloader_discr_iterator)

            step += 1
            total_loss, loss_G, loss_D, loss_D_real, loss_D_fake = module.training_step(batch_data_ct=batch_data_ct, batch_data_real_us=data2.to(hparams.device))

            # log_model_gradients(inner_model, step)

            print(f'{step}/{len(train_dataset_ct_labelmaps) // train_loader_ct_labelmaps.batch_size}, train_loss: {total_loss.item():.4f}, loss_G: {loss_G.item():.4f}, loss_D: {loss_D.item():.4f}, loss_D_real: {loss_D_real.item():.4f}, loss_D_fake: {loss_D_fake.item():.4f}')

            if hparams.logging: 
                wandb.log({"train_seg_loss_step": total_loss.item()}, step=step, commit=True)#)
                wandb.log({"train_loss_G_step": loss_G.item()}, commit=False)# step=step)
                wandb.log({"train_loss_D_total_step": loss_D.item()}, commit=False)# step=step)
                wandb.log({"train_loss_D_fake_step": loss_D_real.item()}, commit=False)# step=step)
                wandb.log({"train_loss_D_real_step": loss_D_fake.item()}, commit=False)# step=step)
                wandb.log({"step":step})

                us_rendering_values = [inner_model.acoustic_impedance_dict, inner_model.attenuation_dict, 
                                        inner_model.mu_0_dict, inner_model.mu_1_dict, inner_model.sigma_0_dict]
                us_rendering_maps = ['acoustic_imp_', 'atten_', 'mu_0_', 'mu_1_', 'sigma_0_' ]
                for map, dict in zip(us_rendering_maps, us_rendering_values):
                    for label, value in zip(inner_model.labels, dict):
                        wandb.log({map+label: value}, commit=False)
                    wandb.log({"step":step})


                    
                # for l in us_rendering_values:
                #     for label, value in zip(inner_model.labels, l):
                #         wandb.log({label: value}, commit=False)
                #     wandb.log({"step":step})



            train_losses.append(total_loss.item())
            G_train_losses.append(loss_G.item())
            D_train_losses.append(loss_D.item())
            D_fake_train_losses.append(loss_D_fake.item())
            D_real_train_losses.append(loss_D_real.item())


        if (epoch + 1) % hparams.validate_every_n_steps == 0:
            dataloader_discr_val_iterator = iter(discr_val_loader)

            module.outer_model.eval()
            inner_model.eval()
            val_step = 0
            with torch.no_grad():
                for val_batch in val_loader_ct_labelmaps:
                    try:
                        data2_val = next(dataloader_discr_val_iterator)
                    except StopIteration:
                        dataloader_discr_val_iterator = iter(discr_val_loader)
                        data2_val = next(dataloader_discr_val_iterator)

                    val_step += 1

                    # val_loss, dict = module.validation_step(val_batch, epoch)
                    val_loss, val_loss_G, val_loss_D, val_loss_D_real, val_loss_D_fake, dict = module.validation_step(epoch=epoch, batch_data_ct=val_batch, batch_data_real_us=data2_val.to(hparams.device))
                    


                    # print(f"{val_step}/{len(val_dataset) // val_loader.batch_size}, val_loss: {val_loss.item():.4f}")
                    print(f'{val_step}/{len(val_dataset_ct_labelmaps) // val_loader_ct_labelmaps.batch_size}, val_loss: {val_loss.item():.4f}, loss_G: {val_loss_G.item():.4f}, loss_D: {val_loss_D.item():.4f}, loss_D_real: {val_loss_D_real.item():.4f}, loss_D_fake: {val_loss_D_fake.item():.4f}')

                    if hparams.logging: 
                        wandb.log({"val_loss_step": val_loss.item()}, step=val_step)
                        wandb.log({"val_loss_G_step": val_loss_G.item()}, step=val_step)
                        wandb.log({"val_loss_D_total_step": val_loss_D.item()}, step=val_step)
                        wandb.log({"val_loss_D_fake_step": val_loss_D_real.item()}, step=val_step)
                        wandb.log({"val_loss_D_real_step": val_loss_D_fake.item()}, step=val_step)


                    valid_losses.append(val_loss.item())
                    plotter.validation_batch_end(dict)
                    
                    G_val_losses.append(val_loss_G.item())
                    D_val_losses.append(val_loss_D.item())
                    D_fake_val_losses.append(val_loss_D_fake.item())
                    D_real_val_losses.append(val_loss_D_real.item())


        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        G_train_loss = np.average(G_train_losses)
        D_train_loss = np.average(D_train_losses)
        D_real_train_loss = np.average(D_real_train_losses)
        D_fake_train_loss = np.average(D_fake_train_losses)

        G_val_loss = np.average(G_val_losses)
        D_val_loss = np.average(D_val_losses)
        D_real_val_loss = np.average(D_real_val_losses)
        D_fake_val_loss = np.average(D_fake_val_losses)
        
        # module.seg_scheduler.step(valid_loss)
        # module.discr_scheduler.step(D_train_loss)
        seg_optimizer_lr = module.seg_optimizer.param_groups[0]['lr']
        optimizer_D_lr = module.optimizer_D.param_groups[0]['lr']
        print('learning rate seg_optimizer = %.7f' % seg_optimizer_lr)
        print('learning rate optimizer_D = %.7f' % optimizer_D_lr)
        # calculate average loss over an epoch
        epoch_len = len(str(hparams.max_epochs))
        print(f'[{epoch:>{epoch_len}}/{hparams.max_epochs:>{epoch_len}}] ' +
                     f'train_loss_epoch: {train_loss:.5f} ' +
                     f'G_train_loss_epoch: {G_train_loss:.5f} ' +
                     f'D_train_loss_epoch: {D_train_loss:.5f} ' +
                     f'D_real_train_loss_epoch: {D_real_train_loss:.5f} ' +
                     f'D_fake_train_loss_epoch: {D_fake_train_loss:.5f} ' +
                     f'valid_loss_epoch: {valid_loss:.5f}'  +
                     f'G_val_loss_epoch: {G_val_loss:.5f}'  +
                     f'D_val_loss_epoch: {D_val_loss:.5f}'  +
                     f'D_real_val_loss_epoch: {D_real_val_loss:.5f}'  +
                     f'D_fake_val_loss_epoch: {D_fake_val_loss:.5f}' 
                     )

        if hparams.logging: 
            wandb.log({"train_loss_epoch": train_loss, "epoch": epoch})
            wandb.log({"G_train_loss_epoch": G_train_loss, "epoch": epoch})
            wandb.log({"D_train_loss_epoch": D_train_loss, "epoch": epoch})
            wandb.log({"D_real_train_loss_epoch": D_real_train_loss, "epoch": epoch})
            wandb.log({"D_fake_train_loss_epoch": D_fake_train_loss, "epoch": epoch})
            wandb.log({"seg_optimizer_lr": seg_optimizer_lr, "epoch": epoch})
            wandb.log({"optimizer_D_lr": optimizer_D_lr, "epoch": epoch})

        if hparams.logging: 
            wandb.log({"val_loss_epoch": valid_loss, "epoch": epoch})
            wandb.log({"G_val_loss_epoch": G_val_loss, "epoch": epoch})
            wandb.log({"D_val_loss_epoch": D_val_loss, "epoch": epoch})
            wandb.log({"D_real_val_loss_epoch": D_real_val_loss, "epoch": epoch})
            wandb.log({"D_fake_val_loss_epoch": D_fake_val_loss, "epoch": epoch})

        if hparams.logging: plotter.validation_epoch_end()
        print(f"--------------- END VAL ------------")
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        if(epoch > hparams.min_epochs):
            early_stopping(valid_loss, epoch, module)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []


            
    print(f'train completed, avg_train_losses: {avg_train_losses:.4f} avg_valid_losses: {avg_valid_losses}' +
          f'best val_loss: {early_stopping.val_loss_min} at best_epoch: {early_stopping.best_epoch}')



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
    hparams.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hparams.exp_name += str(hparams.pred_label) + '_' + str(hparams.discr_model_learning_rate) + '_' + str(hparams.inner_model_learning_rate) + '_' + str(hparams.outer_model_learning_rate)

    opt_cut = TrainOptionsCUT().parse()   # get training options
    opt_cut.dataroot = hparams.data_dir_real_us


    if hparams.logging: wandb.init(name=hparams.exp_name, project=hparams.project_name)
    print('-------------TEST 1 --------------')
    plotter = Plotter()
    print('-------------TEST 2 --------------')


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
    

    if hparams.on_polyaxon:
        from polyaxon_client.tracking import Experiment, get_data_paths, get_outputs_path
        print('get_data_paths(): ', get_data_paths())

        hparams.base_folder_data_path = get_data_paths()['data1'] + hparams.polyaxon_folder
        print('Labelmaps DATASET Folder: ', hparams.base_folder_data_path)

        hparams.data_dir_real_us = get_data_paths()['data1'] + hparams.polyaxon_folder_real_us
        opt_cut.dataroot = hparams.data_dir_real_us
        print('Real US DATASET Folder: ', hparams.data_dir_real_us)

        hparams.output_path = get_outputs_path()
        poly_experiment_info = Experiment.get_experiment_info()
        poly_experiment_nr = poly_experiment_info['experiment_name'].split(".")[-1]
        hparams.name = poly_experiment_nr + "_" + hparams.exp_name
        print(f'get_outputs_path: {hparams.output_path} \n '
              f'experiment_info: {poly_experiment_info} \n experiment_nr: {poly_experiment_nr}')


    # ---------------------
    # LOAD MODEL and DATA 
    # ---------------------
    ModuleClass = load_module(hparams.module)
    InnerModelClass = load_model(hparams.inner_model)
    OuterModelClass = hparams.outer_model
    # parser = ModuleClass.add_module_specific_args(parser)

    DatasetLoader = load_dataset(hparams.dataloader)
    DatasetLoaderDiscr = load_dataset(hparams.dataloader_real_us)

    argparse_summary(hparams, parser)

    # ---------------------
    # RUN TRAINING
    # ---------------------
    train(hparams, opt_cut, ModuleClass, OuterModelClass, InnerModelClass, DatasetLoader, DatasetLoaderDiscr)

# load the last checkpoint with the best model
# model.load_state_dict(torch.load('checkpoint.pt'))