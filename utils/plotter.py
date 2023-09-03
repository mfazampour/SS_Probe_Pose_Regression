import matplotlib.pyplot as plt
from matplotlib import gridspec
# import plotly.express as px
from pytorch_lightning.callbacks import Callback
from matplotlib.colors import NoNorm
import torch
import torchvision
import wandb
import io
from PIL import Image, ImageFont, ImageDraw
import matplotlib.font_manager as fm # to create font
# from pympler import muppy, summary
# import matplotlib
# matplotlib.use('TkAgg')
import numpy as np


EVERY_N = 100
THRESHOLD = 0.5
# top/last N items
N = 16


class Plotter(Callback):
    def __init__(self):
        self.figs = []
        self.plot_figs = []
        self.train_fold_dict = []
        self.val_fold_dict = []
        self.val_fold_dict_epoch = []
        self.test_fold_dict_epoch = []
        # self.current_train_fold = current_train_fold
        self.plt_test_img_cnt = 0

    def log_image(self, plt, caption, trainer, pl_module):
        print('CAPTION: ', caption)
        pl_module.logger.experiment.log({caption: [wandb.Image(plt, caption=caption)]}, commit=True) #commit=True pushesh immediately to wandb
        # wandb.log({caption: [wandb.Image(plt, caption="Output" + caption)]}, commit=True)
    # pl_module.logger.experiment[0].add_image(caption, plt, trainer.global_step, dataformats="HW")

    def reshape_input(self, in_tensor):
        in_tensor = in_tensor[0, :, :, :].cpu()
        return in_tensor.numpy()

    def plot(self, imgs):
        # imgs_reshaped = [self.reshape_input(i) for i in imgs]

        rows = 1
        cols = len(imgs)
        fig = plt.figure(figsize=(cols * 4, rows * 4))
        spec = gridspec.GridSpec(rows, cols, fig, wspace=0, hspace=0)
        spec.tight_layout(fig)
        for idx, img in enumerate(imgs):
            # img = np.rot90(img.cpu().detach(), 3)  #???
            if idx==2:  # prediction blended over the us_sim image
                plt.subplot(spec[0, idx]).imshow(img, cmap='gray', interpolation='none', norm=None)
                us_sim_img = np.rot90(imgs[1].cpu().detach(), 3)
                plt.subplot(spec[0, idx]).imshow(us_sim_img, cmap='gray', alpha=0.4, interpolation='none', norm=None)
            elif idx==0:    # original ct_slice
                plt.subplot(spec[0, idx]).imshow(img)
            elif idx==1:    #us_sim
                plt.subplot(spec[0, idx]).imshow(img, cmap='gray', interpolation='none', norm=None)
            else:   #gt_mask
                plt.subplot(spec[0, idx]).imshow(img, cmap='gray', interpolation='none', norm=None)
            plt.axis('off')
            plt.colorbar()

        plt.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0)
        # plt.savefig("/fig.png", bbox_inches='tight')
        # plt.show()
        # imshow(data, cmap=plt.get_cmap('hot'), interpolation='nearest', vmin=0, vmax=1)

        return fig

    def preprocess_plot(self, file_name, orig_input, gt_mask, pred, us_sim, epoch):
        file_name = "volume_" + file_name                 #file_name.rsplit("/", 1)[1]
        # z_hat_probs = (result > THRESHOLD) * result  # .float()
        # print('z_hat_probs: ', z_hat_probs)
        text = 'epoch=' + str(epoch) + '_' + file_name
        # print('IMG TEXT: ', text)
        fontsize = 28
        ###############################################################################################################
        # plot_fig = self.plot([orig_input.unsqueeze(0), us_sim, gt_mask.unsqueeze(0), pred])
        plot_fig = self.plot([orig_input, us_sim, pred, gt_mask])

        buf = io.BytesIO()
        plot_fig.savefig(buf, format='png')
        buf.seek(0)
        plot_fig_pil = Image.open(buf)
        plot_fig_pil_draw = ImageDraw.Draw(plot_fig_pil)
        font = ImageFont.truetype(fm.findfont(fm.FontProperties()), fontsize)
        plot_fig_pil_draw.text((10, 10), text, (255, 255, 255), font=font)
        plot_fig_pil_t = torchvision.transforms.ToTensor()(plot_fig_pil)
        self.plot_figs.append(plot_fig_pil_t)
        # plot_fig.show()
        plot_fig.clf()  #clear the figure
        plt.close(plot_fig)     #closes windows associated with the fig
        # return plot_fig_pil_t



    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.val_fold_dict_epoch.append(outputs)



    def on_validation_epoch_end(self, trainer, pl_module):
        # val_dict_full = torch.stack(self.val_fold_dict_epoch, 0)
    
        loss_stack = sorted(torch.stack([x['val_loss'] for x in self.val_fold_dict_epoch]))
        # print(loss_stack)
    
        get_top_last = [loss_stack[:N], loss_stack[-N:]]
        caption = [f"best_{N}", f"worst_{N}"]
        # nr_imgs=0
        for i in range(2):
            for x in self.val_fold_dict_epoch:
                if x['val_loss'] in get_top_last[i]:
                    self.preprocess_plot(x["file_name"][0],
                                        x["val_images_unet"].squeeze(),
                                        x["val_images_gt"].squeeze(),
                                        x["val_images_pred"].squeeze(),
                                        x["val_images_us_sim"].squeeze(),
                                        x["epoch"])
                    # nr_imgs += 1
    
            self.log_image(torchvision.utils.make_grid(self.plot_figs), caption[i] +
                           "val |orig_input|us_sim|pred|gt_mask", trainer, pl_module)
            # self.plot_figs.clear()
            del self.plot_figs[:]
    
        # print('test_folds_loss_stack: ', folds_loss_stack)
        # self.val_fold_dict_epoch.clear()
        del self.val_fold_dict_epoch[:]




    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.test_fold_dict_epoch.append(outputs)

    def on_test_end(self, trainer, pl_module):
        loss_stack = sorted(torch.stack([x['test_loss'] for x in self.test_fold_dict_epoch]))
        print(loss_stack)

        get_top_last = [loss_stack[:N], loss_stack[-N:]]
        caption = [f"best_{N}", f"worst_{N}"]
        nr_imgs=0
        for i in range(2):
            for x in self.test_fold_dict_epoch:
                if x['test_loss'] in get_top_last[i]:
                    for n in range(len(x["file_name"])):
                        self.preprocess_plot(x["file_name"][n],
                                             x["test_images_unet"][n],
                                             x["test_images_gt"][n],
                                             x["test_images_pred"][n],
                                             x["test_images_pred_thresh"][n],
                                             x["epoch"])
                        nr_imgs += 1

            self.log_image(torchvision.utils.make_grid(self.plot_figs), caption[i] +
                           "test |orig_input|gt_mask|pred|pred_thresh", trainer, pl_module)
            del self.plot_figs[:]

        del self.test_fold_dict_epoch[:]



            # def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
    #     #display an example from the first training batch
    #     if batch_idx == 0:
    #         print('INSIDE on_train_batch_end')
    #         output_dict = outputs[0][0]['extra']['dict']
    #         # if self.plt_test_img_cnt % EVERY_N == 0:
    #         for i in range(len(output_dict["file_name"])):
    #             # img = output_dict[i, :, :, :]
    #             self.preprocess_plot(output_dict["file_name"][i],
    #                                             output_dict["train_images_unet"][i],
    #                                             output_dict["train_images_gt"][i],
    #                                             output_dict["train_images_pred"][i],
    #                                             output_dict["train_images_pred_thresh"][i],
    #                                             output_dict["epoch"])
    #         self.log_image(torchvision.utils.make_grid(self.plot_figs),
    #                        "train |orig_input|gt_mask|pred|pred_thresh", trainer, pl_module)
    #     del self.plot_figs[:]



    # def create_plot_data_dict(self, x_axis, y_axis, label, metric):
    #     plot_data = {x_axis: [], y_axis: [], label: []}
    #     for fold_dict in self.test_fold_dict:
    #         if fold_dict['epoch'] == metric:
    #             plot_data[y_axis].append(fold_dict['val_loss'].item())
    #             plot_data[label].append(fold_dict['current_val_fold'])
    #             if x_axis is 'aorta_size':
    #                 plot_data[x_axis].append(torch.sum(fold_dict["val_images_pred"][0]))
    #             elif x_axis is 'aorta_X_coord_mean':
    #
    #     print('plot_data: ', plot_data)
    #     fig = px.scatter(plot_data, x='aorta_size', y='DICE_loss', color="patient")
    #
    #     return fig

    # def on_train_end(self, trainer, pl_module):
    #
    #     print('---------------ON TRAIN END--------------')
    #     train_val_dict = [self.train_fold_dict, self.val_fold_dict]
    #     for i in range(2):
    #         plot_data = {'aorta_size': [], 'aorta_X_coord_mean': [], 'aorta_Y_coord_mean': [], 'DICE_loss': [],
    #                      'patient': []}
    #         type = 'train' if i == 0 else 'val'
    #         # print(train_val_dict[i])
    #         for fold_dict in train_val_dict[i]:
    #             if fold_dict['epoch'] == best_model_epoch:
    #                 plot_data['aorta_size'].append(torch.sum(fold_dict[f'{type}_images_pred'][0]))
    #                 plot_data['DICE_loss'].append(fold_dict[f'{type}_loss'].item())
    #                 plot_data['patient'].append(fold_dict[f'current_{type}_fold'])
    #                 indices_x_y = torch.nonzero(fold_dict[f'{type}_images_pred'][0][0, 0, :, :] == 1.0)
    #                 plot_data['aorta_X_coord_mean'].append(
    #                     (indices_x_y[:, 0] * 1.0).mean())  # mean x coord of all aorta pixels for this image
    #                 plot_data['aorta_Y_coord_mean'].append((indices_x_y[:, 1] * 1.0).mean())
    #
    #         fig = px.scatter(plot_data, x='aorta_size', y='DICE_loss', color="patient")
    #         # fig.show()
    #         caption = 'During training' if i == 0 else 'Eval'
    #         pl_module.logger.experiment.log(
    #             {f"{caption} Aorta size/DICE loss (Trained on: {self.current_train_fold})": fig})
    #         #############################################################################################
    #         # plot_data['aorta_X_coord'] = torch.unbind(torch.unique(indices_x_y[:, 0]))   #unique removes duplicate values, unbind is the opposite of stack
    #         fig = px.scatter(plot_data, x='aorta_X_coord_mean', y='DICE_loss', color="patient")
    #         pl_module.logger.experiment.log(
    #             {f"{caption} Aorta X coord mean/DICE loss (Trained on: {self.current_train_fold})": fig})
    #
    #         fig = px.scatter(plot_data, x='aorta_Y_coord_mean', y='DICE_loss', color="patient")
    #         pl_module.logger.experiment.log(
    #             {f"{caption} Aorta Y coord mean/DICE loss (Trained on: {self.current_train_fold})": fig})
    #
    #     # self.train_fold_dict.clear()
    #     # self.val_fold_dict.clear()
    #     del self.train_fold_dict[:]
    #     del self.val_fold_dict[:]


    # def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
    #     self.val_fold_dict.append(outputs)
        # ################# PLOT Only every n-th image ##################
        # if self.plt_test_img_cnt % EVERY_N == 0:
        #     file_name = outputs["file_name"]
        #     file_name = file_name[0][0].rsplit("/", 1)[1]
        #
        #     x = outputs["test_images_unet"][0]  # orig_input
        #     z = outputs["test_images_unet"][1]  # unet output
        #     y = outputs["test_images_end"][0]  # gt_mask from real img
        #     z_hat = outputs["test_images_end"][1]  # result segmentation after tiramisu net
        #     print('z_hat min: ', torch.min(z_hat), 'z_hat max: ', torch.max(z_hat))
        #
        #     # z_hat_probs = torch.sigmoid(z_hat)
        #     z_hat_probs = torch.ge(z_hat, THRESHOLD).float()
        ##################################################################################################################
        ################ USING torchvision make_grid ##################
        ##################################################################################################################
        # tensor_list = torch.cat((x, z, y, z_hat_probs), 0)
        # print('tensor_list shape: ', tensor_list.shape)
        # fig = torchvision.utils.make_grid(tensor_list, padding=4)
        # print('fig shape make_grid: ', fig.shape, ' dtype: ', fig.dtype)
        #
        # pil_fig = torchvision.transforms.ToPILImage()(fig)
        # draw = ImageDraw.Draw(pil_fig)
        # font = ImageFont.truetype(fm.findfont(fm.FontProperties()), fontsize)
        # draw.text((0, 0), file_name, (255, 255, 255), font=font)
        #
        # fig = torchvision.transforms.ToTensor()(pil_fig)
        # print('fig shape end: ', fig.shape, ' dtype: ', fig.dtype)
        #
        # self.figs.append(fig)  # , str(dataloader_idx) + '_' + file_name + "/orig_input/unet_out/gt_mask/final_seg"))
        # self.log_function(fig, "/orig_input/unet_out/gt_mask/final_seg", trainer, pl_module)

    # def unblockshaped(self, arr, h, w):
    #     """
    #     Return an array of shape (h, w) where
    #     h * w = arr.size

    #     If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    #     then the returned array preserves the "physical" layout of the sublocks.
    #     """
    #     n, nrows, ncols = arr.shape
    #     return (arr.reshape(h // nrows, -1, nrows, ncols)
    #             .swapaxes(1, 2)
    #             .reshape(h, w))

    #############################################################################################################
    # def on_validation_epoch_end(self, trainer, pl_module):

    # element = self.figs[-1]     #plot the last img after all steps within one epoch

    # for element in self.figs:
    #     self.log_function(element[0], element[1], trainer, pl_module)
    # self.figs.clear()

    # def on_validation_end(self, trainer, pl_module):
    #     ############################################################
    #     for element in self.figs:
    #         self.log_function(element[0], element[1], trainer, pl_module)
    #
    #     self.figs.clear()
    ####################################################################


    # def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
    #     print('-------------ON TRAIN BATCH END PRINT WEIGHTS:--------------')
    #     for name, param in pl_module.UnetModel.named_parameters():
    #         # pl_module.logger.experiment.log({"Unet." + name: param.data})
    #         # pl_module.logger.experiment.summary[name] = param.data
    #         pl_module.logger.experiment.log({"Unet." + name: wandb.Histogram(param.data.numpy())})
    #
    #     for name, param in pl_module.tiramisuModel.named_parameters():
    #         # pl_module.logger.experiment.log({"Tiramisu." + name: param.data})
    #         # pl_module.logger.experiment.summary[name] = param.data
    #         pl_module.logger.experiment.log({"Tiramisu." + name: wandb.Histogram(param.data.numpy())})


    # def log_sds_plots(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, step_prefix="Test"):
    #     print("\nthis is being called once")
    #
    #     y = outputs["val_images"][0]
    #     z = outputs["val_images"][1]
    #     z = z[0, :, :, :].cpu()
    #     print (z.shape)
    #
    #     z_reshaped  = self.unblockshaped(z.numpy(),z.shape[1]*4  , z.shape[2]*5)
    #     y = y[0, :, :, :].cpu()
    #     y_reshaped  = self.unblockshaped(y.numpy(),y.shape[1]*4  , y.shape[2]*5)
    #     # matplotlib.image.imsave("/home/javi/temp/stupidz.jpeg", z_reshaped)
    #     # np.save(z_reshaped , )
    #
    #     self.zs.append({z_reshaped > -1 , str(batch_idx) + "_Z" })
    #     self.ys.append({y_reshaped , str(batch_idx) + "_Y" })
    #
    #     # self.log_function( z_reshaped > -1, str(batch_idx) + "_1", trainer, pl_module, False)
    #     # self.log_function( z_reshaped ,  str(batch_idx) + "_2", trainer, pl_module , False)
    #     # self.log_function( y_reshaped ,  str(batch_idx) + "_3", trainer, pl_module , False)


    # for element in self.input_img:
    #     self.log_function(element[0], element[1], trainer, pl_module, False)
    #
    # for element in self.unet_out:
    #     self.log_function(element[0], element[1], trainer, pl_module, False)
    #
    # for element in self.zs:
    #     self.log_function(element[0], element[1], trainer, pl_module, False)
    #     self.log_function(element[0] > 0.7, "Thresholded_" + element[1], trainer, pl_module, False)
    #
    # for element in self.ys:
    #     self.log_function(element[0], element[1], trainer, pl_module, False)
    #
    # self.input_img.clear()
    # self.unet_out.clear()
    # self.zs.clear()
    # self.ys.clear()

    # pl_module.logger.experiment[0].log({""},commit=True)