from pathlib import Path
import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import configargparse
from pytorch_lightning.loggers import WandbLogger
# from ray.tune.integration.pytorch_lightning import TuneReportCallback
from utils.plotter import Plotter
from utils.utils import (
    argparse_summary,
    get_class_by_path,
)
from pytorch_lightning import seed_everything
from utils.configargparse_arguments import build_configargparser
from datetime import datetime
# import wandb
# import os
import sys
import numpy as np

# try:
#     from utils.plx_logger import PolyaxonLogger
# except ImportError:
#     assert 'Import Error'

RANDOM_SEED = False
torch.use_deterministic_algorithms(True, warn_only=True)

COMMENT = f"################ RANDOM SEED: {RANDOM_SEED} ############################"

# polyaxon_data_path = '/us_sim_sweeps/segmentation/'
# polyaxon_folder = polyaxon_data_path + 'aorta_segm_sim/8CTs_sim_ctlabelmap2_impr_3Splines_augm/data'


def train(hparams, ModuleClass, OuterModelClass, InnerModelClass, DatasetClass, DatasetModule, loggers, train_loader, val_loader):

    if not RANDOM_SEED:
        seed_everything(0)

    outer_model = OuterModelClass(hparams=hparams).to(hparams.device)
    inner_model = InnerModelClass(hparams=hparams).to(hparams.device)

    #################################################
    # inner_model = InnerModuleClass.load_from_checkpoint(ckpt_path, inner_model, criterion)
    # ###################################################################

    # load module
    module = ModuleClass(hparams, outer_model, inner_model)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{hparams.output_path}/checkpoints/",
        save_top_k=hparams.save_top_k,
        save_last=True,
        verbose=True,
        monitor=hparams.early_stopping_metric,
        mode='min',
        # prefix=hparams.name,
        filename=f'{{epoch}}-{{{hparams.early_stopping_metric}:.5f}}'
    )
    early_stop_callback = EarlyStopping(
        monitor=hparams.early_stopping_metric,
        min_delta=0.00,
        verbose=True,
        patience=3,
        mode='min')

    plotter = Plotter()
    # ------------------------
    # 3 INIT TRAINER
    # ------------------------

    # metrics = {"loss": "val_loss"}
    # tune_callbacks = [TuneReportCallback(metrics, on="validation_end")]

    trainer = pl.Trainer(
        gpus=1, #hparams.gpus,
        logger=loggers,
        accelerator=hparams.accelerator,
        # fast_dev_run=True,
        min_epochs=hparams.min_epochs,
        max_epochs=hparams.max_epochs,
        # checkpoint_callback=True,
        resume_from_checkpoint=hparams.resume_from_checkpoint,
        callbacks=[early_stop_callback, checkpoint_callback, plotter],#, tune_callbacks],
        enable_model_summary=True,  #weights_summary='full',
        # auto_lr_find=True,    #, results override hparams.learning_rate
        num_sanity_val_steps=hparams.num_sanity_val_steps,
        log_every_n_steps=hparams.log_every_n_steps,
        # check_val_every_n_epoch=1,  # how often the model will be validated
        limit_train_batches=hparams.limit_train,  # How much of training dataset to check (if 1 check all)
        limit_val_batches=hparams.limit_val,
        # limit_test_batches=LIMIT_TEST,
        # auto_scale_batch_size="binsearch",   # search for optimal batch size,  result overrides hparams.batch_size
        # automatic_optimization=True,
        profiler="simple",
        precision=16,
        # deterministic=True,
        # track_grad_norm=2, 
        detect_anomaly=True
        # val_check_interval=10.0
    )

    # print(f'---------------TRAIN ON GPU ???  {trainer.on_gpu}')

    print('batch_size: ', hparams.batch_size, 'Learning rate: ', hparams.learning_rate)
    # trainer.tuner.lr_find(model=module)
    # trainer.tune(model=module, datamodule=data_module)
    # trainer.tune(model=module, train_dataloader=train_dataloader, val_dataloaders=val_loaders)
    # print('AFTER TUNE batch_size: ', hparams.batch_size, 'Learning rate: ', hparams.learning_rate)
    print('MIN EPOCHS: ', trainer.min_epochs)
    print('MAX EPOCHS: ', trainer.max_epochs)

    wandb_logger.watch(module, log="all", log_freq=100)  # plots the gradients
    # print(f'---------------TRAIN.FIT--------------\n'
        #   f'VAL DISABLED?: {trainer.disable_validation} \n'
        #   f'VAL ENABLED?: {trainer.enable_validation}')


    if DatasetModule is not None:
        data_module = DatasetModule(hparams)

        trainer.fit(model=module, datamodule=data_module)
    else:
        trainer.fit(module, train_dataloader=train_loader, val_dataloaders=val_loader)

    # print('---------------TEST--------------')
    # trainer.test(model=module, datamodule=data_module)

    # return trainer.checkpoint_callback.best_model_score  # best_model_loss


# LOAD MODULE
def load_module(hparams):
    # ------------------------
    # LOAD MODULE
    # ------------------------
    module_path = f"modules.{hparams.module}"
    ModuleClass = get_class_by_path(module_path)

    # inner_module_path = f"modules.{hparams.inner_module}"
    # InnerModuleClass = get_class_by_path(inner_module_path)
    return ModuleClass


def load_model(hparams):
    # ------------------------
    # LOAD MODEL
    # ------------------------
    outer_model_path = f"models.{hparams.outer_model}"
    OuterModelClass = get_class_by_path(outer_model_path)
    
    inner_model_path = f"models.{hparams.inner_model}"
    InnerModelClass = get_class_by_path(inner_model_path)
    # parser = OuterModelClass.add_model_specific_args(parser)

    return OuterModelClass, InnerModelClass


# LOAD DATASET
def load_dataset(hparams):
    dataset_path = f"datasets.{hparams.dataset}"
    DatasetClass = get_class_by_path(dataset_path)

    dataset_path = f"datasets.{hparams.datamodule}"
    DatasetModule = get_class_by_path(dataset_path)
    # parser = DatasetClass.add_dataset_specific_args(parser)
    return DatasetClass, DatasetModule


if __name__ == "__main__":
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('-c', is_config_file=True, help='config file path')
    parser, hparams = build_configargparser(parser)

    print(f'************************************************************************************* \n'
          f'COMMENT: {COMMENT}' # + str(hparams.nr_train_folds)} \n'
          f'************************************************************************************* \n'
          f'PL VERSION: {pl.__version__} \n'
          f'PYTHON VERSION: {sys.version} \n '
          # f'WANDB VERSION: {wandb.__version__} \n '
          f'TORCH VERSION: {torch.__version__} \n '
          f'TORCHVISION VERSION: {torchvision.__version__}')

    ModuleClass = load_module(hparams)
    OuterModelClass, InnerModelClass = load_model(hparams)
    parser = ModuleClass.add_module_specific_args(parser)
    # parser = OuterModelClass.add_module_specific_args(parser)
    # parser = InnerModelClass.add_model_specific_args(parser)
    
    DatasetClass, DataModule = load_dataset(hparams)
    # parser = DatasetClass.add_dataset_specific_args(parser)

    hparams = parser.parse_args()
    # setup logging
    exp_name = (
            hparams.module.split(".")[-1]
            + "_"
            + hparams.dataset.split(".")[-1]
            # + "_"
            # + hparams.model.replace(".", "_")
    )
    print(f'This will run on polyaxon: {str(hparams.on_polyaxon)}')

    hparams.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('torch.cuda.is_available(): ', torch.cuda.is_available())
    print('torch.cuda.current_device(): ', torch.cuda.current_device())
    print('torch.cuda.device(0): ', torch.cuda.device(torch.cuda.current_device()))
    print('torch.cuda.device_count(): ', torch.cuda.device_count())
    print('torch.cuda.get_device_name(): ', torch.cuda.get_device_name(torch.cuda.current_device()))

    print('device: ', hparams.device)

    if hparams.on_polyaxon:
        from polyaxon_client.tracking import Experiment, get_data_paths, get_outputs_path

        hparams.data_root = get_data_paths()['data1'] + polyaxon_folder
        print('DATASET Folder: ', hparams.data_root)
        hparams.output_path = get_outputs_path()
        poly_experiment_info = Experiment.get_experiment_info()
        poly_experiment_nr = poly_experiment_info['experiment_name'].split(".")[-1]
        hparams.name = poly_experiment_nr + "_" + exp_name
        print(f'get_outputs_path: {get_outputs_path()} \n '
              f'experiment_info: {poly_experiment_info} \n experiment_name: {poly_experiment_nr}')

    else:
        date_str = datetime.now().strftime("%y%m%d-%H%M%S_")
        hparams.name = 'local_' + date_str + exp_name
        # hparams.output_path = Path(hparams.output_path).absolute() / hparams.name

    # if hparams.on_polyaxon:
    wandb_logger = WandbLogger(name=hparams.name, project=f"cactuss_end2end-{hparams.module.split('.')[-1]}")
    # wandb.init(project=f"aorta_segm_sim-{hparams.model.split('.')[-1]}")
    loggers = wandb_logger
    # else:
    #     loggers = None

    argparse_summary(hparams, parser)

    # ---------------------
    # LOAD DATA LOADERS
    # ---------------------


    # train_loader = torch.utils.data.DataLoader(
    #     DatasetClass(hparams, "train"), batch_size=hparams.batch_size, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(
    #     DatasetClass(hparams, "val"), batch_size=hparams.batch_size, shuffle=False)
    # test_loader = torch.utils.data.DataLoader(
    #     DatasetClass(hparams, "test/"), batch_size=hparams.batch_size, shuffle=False)

    # ---------------------
    # RUN TRAINING
    # ---------------------
    train(hparams, ModuleClass, OuterModelClass, InnerModelClass, DatasetClass, DataModule, loggers, None, None)
    # train(hparams, ModuleClass, ModelClass, DatasetClass, None, loggers, train_loader, val_loader)


