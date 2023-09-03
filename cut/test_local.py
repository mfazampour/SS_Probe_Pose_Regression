"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.mdf
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images_no_web
# from util import html
import util.util as util
from glob import glob
import cv2
import numpy as np
from PIL import Image
import torchvision

polyaxon_data_path = '/us_sim_sweeps/segmentation/'
polyaxon_folder = polyaxon_data_path + 'cut/aorta_us_small'
# LOCAL_FOLDER = './datasets/aorta_for_val'
LOCAL_FOLDER = './datasets/aorta_for_inference'


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    if opt.on_polyaxon:
        from polyaxon_client.tracking import Experiment, get_data_paths, get_outputs_path, get_base_outputs_path
        opt.dataroot = get_data_paths()['data1'] + polyaxon_folder
        opt.checkpoints_dir = get_outputs_path() + '/checkpoints'

        print("opt.dataroot: ", opt.dataroot)
        print("opt.checkpoints_dir: ", opt.checkpoints_dir)
        print("get_base_outputs_path(): ", get_base_outputs_path())

    else:
        opt.dataroot = LOCAL_FOLDER
        opt.gpu_ids = '-1'

    # hard-code some parameters for test
    opt.gpu_ids = '-1'
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.phase = 'test'

    # files = sorted(glob(opt.checkpoints_dir + '/aorta_CUT/*.pth'))
    files = sorted(glob(opt.checkpoints_dir + '/experiment_name/*.pth'))
    for file in files[::3]:
        print('FILE: ', file)
        opt.epoch = file.split("/")[-1].split("_")[0]
        print("EPOCH: ", opt.epoch)

        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        # train_dataset = create_dataset(util.copyconf(opt, phase="train"))
        model = create_model(opt)      # create a model given opt.model and other options
        # create a webpage for viewing the results
        web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
        print('creating web directory', web_dir)
        # webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

        save_dir = opt.results_dir + opt.name + '/' + 'test_' + opt.epoch + '/' + 'images/'
        for i, data in enumerate(dataset):
            if i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                # model.parallelize()
                if opt.eval:
                    model.eval()
            if i >= opt.num_test:  # only apply our model to opt.num_test images.
                break
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()     # get image paths
            print('img_path: ', img_path)

            output_cut = visuals['fake_B']
            # pil_fig = torchvision.transforms.ToPILImage()(output_cut[0].numpy().astype(np.uint8).transpose(1, 2, 0)).convert('L')
            # pil_fig.show()
            image_numpy = output_cut.data[0].clamp(-1.0, 1.0).cpu().float().numpy()
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
            image_numpy = image_numpy.astype(np.uint8)

            # cv2.imshow("CUT", image_numpy)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


            # if i % 5 == 0:  # save images to an HTML file
            # print('processing (%04d)-th image... %s' % (i, img_path))
            # if not os.path.exists(save_dir):
            save_images_no_web(save_dir, visuals, img_path, width=opt.display_winsize)
        # webpage.save()  # save the HTML
