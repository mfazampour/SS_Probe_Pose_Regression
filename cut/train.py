import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import wandb
from datetime import datetime


polyaxon_data_path = '/us_sim_sweeps/segmentation/'
polyaxon_folder = polyaxon_data_path + 'cut/aorta_us_8CTs'
# polyaxon_folder = polyaxon_data_path + 'cut/aorta_us_small'

if __name__ == '__main__':

    print('torch.cuda.is_available() ? ', torch.cuda.is_available())
    print('USE CUT and USE DATASET FOLDER: ', polyaxon_folder)

    opt = TrainOptions().parse()   # get training options
    print('INIT Learning rate: ', opt.lr)

    if opt.on_polyaxon:
        from polyaxon_client.tracking import Experiment, get_data_paths, get_outputs_path
        opt.dataroot = get_data_paths()['data1'] + polyaxon_folder
        opt.checkpoints_dir = get_outputs_path() + '/checkpoints'
        poly_experiment_info = Experiment.get_experiment_info()
        poly_experiment_nr = poly_experiment_info['experiment_name'].split(".")[-1]
        wandb_name = poly_experiment_nr
    else:
        opt.dataroot = './datasets/aorta_us_8CTs'
        opt.gpu_ids = '-1'
        date_str = datetime.now().strftime("%y%m%d-%H%M%S")
        wandb_name = 'local_' + date_str

    wandb.init(project="CUT", name=wandb_name)

    print('len(opt.gpu_ids): ', len(opt.gpu_ids))

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.

    model = create_model(opt)      # create a model given opt.model and other options
    print('The number of training images = %d' % dataset_size)

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    opt.visualizer = visualizer
    total_iters = 0                # the total number of training iterations

    optimize_time = 0.1

    times = []
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        dataset.set_epoch(epoch)
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result, wandb)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data, wandb)
                # if opt.display_id is None or opt.display_id > 0:
                #     visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            # if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
            #     print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            #     print(opt.name)  # it's useful to occasionally show the experiment name on console
            #     save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            #     model.save_networks(save_suffix)

            iter_data_time = time.time()

        ############disable saving for now ########
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            if opt.on_polyaxon:
                model.save_dir = get_outputs_path()
            model.save_networks('latest')
            model.save_networks(epoch)

        lr = model.update_learning_rate()                     # update learning rates at the end of every epoch.
        wandb.log({"lr": lr})
        print('Learning rate: ', lr)
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))


