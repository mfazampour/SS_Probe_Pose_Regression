import logging
import sys
import numpy as np
import torch
import wandb
# from tensorboardX import SummaryWriter
# import monai
from tqdm import tqdm
import configargparse
from utils.configargparse_arguments_torch import build_configargparser
import torchvision.transforms.functional as F
from utils.plotter_torch import Plotter
from utils.utils import argparse_summary, get_class_by_path
from utils.early_stopping import EarlyStopping
from utils.wandb_sweeper import HyperParameterSearch
import sys

RANDOM_SEED = False
BATCH = 2
# torch.use_deterministic_algorithms(True, warn_only=True)
# COMMENT = f"################ UNET vessel segm ############################"
# tb_logger = SummaryWriter('log_dir/cactuss_end2end')
    



def check_gradients(net):
    for name, param in net.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f'NaN found in gradients for parameter {name}')
            sys.exit()


def training_loop_pose_net(hparams, module, batch_size, train_loader, train_losses, plotter, cut_trainer, simulated=False, epoch=0):
    step = 0
    for batch_data in tqdm(train_loader, total=len(train_loader), ncols=100):
        step += 1

        # loss, _ = module.training_step(batch_data)
        # input, label = module.get_data(batch_data)
        module.optimizer.zero_grad()

        input, pose_gt, file_name, volume, spacing, direction, origin, ct, ct_id = module.get_data(batch_data)

        if simulated:
            input = module.rendering_forward(input)

        cut_trainer.forward_cut_B(input)
        idt_B = cut_trainer.cut_model.idt_B
        # idt_B = idt_B.clone()
        idt_B = (idt_B / 2) + 0.5  # from [-1,1] to [0,1]

        losses, prediction = module.pose_forward(idt_B, pose_gt, ct_id=ct_id)
        loss = losses['sum_loss']

        check_gradients(module)
        loss.backward()
        # log_model_gradients(inner_model, step)
        module.optimizer.step()
        # batch_loss_list =[]

        # print(f"{step}/{len(train_loader.dataset) // train_loader.batch_size}, train_loss: {loss.item():.4f}")
        if hparams.logging:
            wandb.log({"phantom_slice_loss_step": losses, 'phantom_slice_epoch': epoch})
        train_losses.append(loss.item())

        if hparams.logging:
            plotter.log_us_rendering_values(module.USRenderingModel, step)  # todo: check if step is needed



def training_loop_seg_net(hparams, module, inner_model, batch_size, train_loader, train_losses,plotter):
    step = 0
    batch_loss_list =[]
    if batch_size == 1:
        for batch_data in tqdm(train_loader, total=len(train_loader), ncols= 100):
            step += 1

            # loss, _ = module.training_step(batch_data)
            # input, label = module.get_data(batch_data)
            module.optimizer.zero_grad()

            input, label = module.get_data(batch_data)
            loss, us_sim, prediction = module.step(input, label)

            check_gradients(module)
            loss.backward()
            # log_model_gradients(inner_model, step)
            module.optimizer.step()
            # batch_loss_list =[]

            print(f"{step}/{len(train_loader.dataset) // train_loader.batch_size}, train_loss: {loss.item():.4f}")
            if hparams.logging: wandb.log({"train_loss_step": loss.item()}, step=step)
            train_losses.append(loss.item())
            
            plotter.log_us_rendering_values(inner_model, step)

    else:
        dataloader_iterator = iter(train_loader)
        while step < len(train_loader):
            step += 1
            module.optimizer.zero_grad()
            # while step % hparams.batch_size_manual != 0 :
            while step % batch_size != 0:
                data = next(dataloader_iterator)

                input, label = module.get_data(data)
                loss, us_sim, prediction = module.step(input, label)
                batch_loss_list.append(loss)
                step += 1
                if hparams.logging: wandb.log({"train_loss_step": loss.item()}, step=step)
                train_losses.append(loss.item())
            
            loss = torch.mean(torch.stack(batch_loss_list))

            check_gradients(module)
            loss.backward()
            module.optimizer.step()
            batch_loss_list =[]

            print(f"{step}/{len(train_loader.dataset) // train_loader.batch_size}, train_loss: {loss.item():.4f}")
            plotter.log_us_rendering_values(inner_model, step)


def train(hparams, ModuleClass, OuterModelClass, InnerModelClass, DatasetLoader, wandb_conf):

    if not RANDOM_SEED:
        torch.manual_seed(2023)

    # with wandb.init(config=config):
    # wandb.config.update(config)
    # config = wandb.config
    # hparams.outer_model_learning_rate = config.lr_outer_model
    # hparams.inner_model_learning_rate = config.lr_inner_model
    plotter = Plotter()
    inner_model = InnerModelClass()
    outer_model = OuterModelClass(hparams=hparams)

    module = ModuleClass(hparams, outer_model, inner_model)

    if wandb_conf is not None:
        module.optimizer.param_groups[0]['lr'] = wandb_conf.lr_outer_model
        module.optimizer.param_groups[1]['lr'] = wandb_conf.lr_inner_model
        bs = wandb_conf.batch_size
    else:
        bs = hparams.batch_size_manual

    # if hparams.logging: wandb.watch(inner_model, log='all', log_graph=True, log_freq=100)
    if hparams.logging: wandb.watch([inner_model, module.outer_model], log='all', log_graph=True, log_freq=100)

    dataloader = DatasetLoader(hparams)
    train_loader, train_dataset, val_dataset  = dataloader.train_dataloader()
    val_loader = dataloader.val_dataloader()

    early_stopping = EarlyStopping(patience=hparams.early_stopping_patience, 
                                    ckpt_save_path = f'{hparams.output_path}/best_checkpoint_{hparams.exp_name}.pt', verbose=True)

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 


    for epoch in range(1, hparams.max_epochs + 1):
        module.outer_model.train()
        inner_model.train()

        training_loop_seg_net(hparams, module, inner_model, bs, train_loader, train_losses, plotter)


        if (epoch + 1) % hparams.validate_every_n_steps == 0:
            module.outer_model.eval()
            inner_model.eval()
            val_step = 0
            with torch.no_grad():
                for val_batch in tqdm(val_loader, total=len(val_loader), ncols= 100):
                    val_step += 1

                    _ , val_loss, dict = module.validation_step(val_batch, epoch)
                    print(f"{val_step}/{len(val_dataset) // val_loader.batch_size}, val_loss: {val_loss.item():.4f}")
                    if hparams.logging: wandb.log({"val_loss_step": val_loss.item()})

                    valid_losses.append(val_loss.item())
                    plotter.validation_batch_end(dict)

        # calculate average loss over an epoch
        train_loss_epoch = np.average(train_losses)
        val_loss_epoch = np.average(valid_losses)
        avg_train_losses.append(train_loss_epoch)
        avg_valid_losses.append(val_loss_epoch)
        
        epoch_len = len(str(hparams.max_epochs))
        print(f'[{epoch:>{epoch_len}}/{hparams.max_epochs:>{epoch_len}}] ' +
                    f'train_loss_epoch: {train_loss_epoch:.5f} ' +
                    f'val_loss_epoch: {val_loss_epoch:.5f}')

        if hparams.logging: wandb.log({"train_loss_epoch": train_loss_epoch, "epoch": epoch})
        if hparams.logging: wandb.log({"val_loss_epoch": val_loss_epoch, "epoch": epoch})

        if hparams.logging: plotter.validation_epoch_end()
        print(f"--------------- END VAL ------------")
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        if(epoch > hparams.min_epochs):
            early_stopping(val_loss_epoch, epoch, module)
        
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
    module_path = f"modules.{module}"
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
def load_dataset(hparams):
    dataset_path = f"datasets.{hparams.dataset}"
    DatasetClass = get_class_by_path(dataset_path)

    dataset_path = f"datasets.{hparams.datamodule}"
    DataLoader = get_class_by_path(dataset_path)
    # parser = DatasetClass.add_dataset_specific_args(parser)
    return DatasetClass, DataLoader



# if __name__ == "__main__":
def main():

    # monai.config.print_config()
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('-c', is_config_file=True, help='config file path')
    parser, hparams = build_configargparser(parser)
    hparams.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hparams.exp_name += str(hparams.pred_label) + '_' + str(hparams.inner_model_learning_rate) + '_' + str(hparams.outer_model_learning_rate)


    if hparams.wandb_conf:
        run = wandb.init()
        wandb_conf = wandb.config
    else:
        wandb_conf = None
        if hparams.logging: wandb.init(name=hparams.exp_name, project=hparams.project_name)


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

        hparams.base_folder_data_path = get_data_paths()['data1'] + hparams.polyaxon_imgs_folder
        hparams.base_folder_mask_path = get_data_paths()['data1'] + hparams.polyaxon_masks_folder
        print('Labelmaps DATASET Folder: ', hparams.base_folder_data_path)

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
    OuterModelClass = load_model(hparams.outer_model)
    # OuterModelClass = hparams.outer_model
    # parser = ModuleClass.add_module_specific_args(parser)

    DatasetClass, DatasetLoader = load_dataset(hparams)
    argparse_summary(hparams, parser)

    # wandb.init()

    # with wandb.init():
    #     config = wandb.config

    # ---------------------
    # RUN TRAINING
    # ---------------------
    train(hparams, ModuleClass, OuterModelClass, InnerModelClass, DatasetLoader, wandb_conf)


if __name__ == "__main__":
    
    parser = configargparse.ArgParser(
    config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('-c', is_config_file=True, help='config file path')
    parser, hparams = build_configargparser(parser)

    if hparams.wandb_conf:
        # wandb_sweeep = HyperParameterSearch(method='grid', metric='val_loss_epoch')
        # wandb_sweeep_config = wandb_sweeep.set_parameters_to_optimize_grid(batch_sizes=[1, 2, 4, 8, 32, 64, 128], 
        #                                                                     lrs_inner_model=[0.0001, 0.001, 0.01], 
        #                                                                     lrs_outer_model=[0.0001, 0.001, 0.01, 0.1])
        # # # wandb_sweeep_config = wandb_sweeep.set_parameters_to_optimize_random(batch_sizes_min_max=[1, 64], lrs_inner_model_min_max=[0, 0.1], lrs_outer_model_min_max=[0, 0.1])

        # sweep_id = wandb.sweep(wandb_sweeep_config, project="cactuss_end2end-sweeps")
        # # print("sweep_id", sweep_id)
        # wandb.config.update(wandb_sweeep_config)

        # api = wandb.Api()
        # run = api.run('danivelikova/cactuss_end2end-sweeps/69rfe3cg')
        # run.config["lrs_inner_model"] = [0.0001, 0.001, 0.01, 0.1]
        # run.config["lrs_outer_model"] = [0.0001, 0.001, 0.01, 0.1]
        # run.update()

        sweep_id = 'danivelikova/cactuss_end2end-sweeps/uvn3avwq'   #grid
        #sweep_id =  'danivelikova/cactuss_end2end-sweeps/aip1hggd'  #bayesian sweep
        wandb.agent(sweep_id, function=main, count=1)

    else:
        main()


# load the last checkpoint with the best model
# model.load_state_dict(torch.load('checkpoint.pt'))