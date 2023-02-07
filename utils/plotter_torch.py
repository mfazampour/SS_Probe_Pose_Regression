import matplotlib.pyplot as plt
from matplotlib import gridspec
# import plotly.express as px
import torchvision
import wandb
import io
from PIL import Image, ImageFont, ImageDraw
import matplotlib.font_manager as fm # to create font
# from pympler import muppy, summary
# import matplotlib
# matplotlib.use('TkAgg')
import numpy as np
from operator import *

EVERY_N = 100
THRESHOLD = 0.5
# top/last N items
N = 10


class Plotter():
    def __init__(self):
        self.figs = []
        self.plot_figs = []
        self.train_fold_dict = []
        self.val_fold_dict = []
        self.val_fold_dict_epoch = []
        self.test_fold_dict_epoch = []
        self.plt_test_img_cnt = 0

    def log_image(self, plt, caption):
        print('CAPTION: ', caption)
        wandb.log({caption: [wandb.Image(plt, caption=caption)]}, commit=True)

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
            img = np.rot90(img.cpu().detach(), 3)
            if idx==2:  # prediction blended over the us_sim image
                plt.subplot(spec[0, idx]).imshow(img, cmap='gray')
                us_sim_img = np.rot90(imgs[1].cpu().detach(), 3)
                plt.subplot(spec[0, idx]).imshow(us_sim_img, cmap='gray', alpha=0.4)
            elif idx==0:    # original ct_slice
                plt.subplot(spec[0, idx]).imshow(img)
            elif idx==1:    #us_sim
                plt.subplot(spec[0, idx]).imshow(img, cmap='gray')
            else:   #gt_mask
                plt.subplot(spec[0, idx]).imshow(img, cmap='gray')
            plt.axis('off')

        plt.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0)
        # plt.savefig("/fig.png", bbox_inches='tight')
        # plt.show()

        return fig

    def preprocess_plot(self, file_name, orig_input, gt_mask, pred, us_sim, epoch):
        file_name = "volume_" + file_name                 #file_name.rsplit("/", 1)[1]
        # z_hat_probs = (result > THRESHOLD) * result  # .float()
        # print('z_hat_probs: ', z_hat_probs)
        text = 'epoch=' + str(epoch) + '_' + file_name
        # print('IMG TEXT: ', text)
        fontsize = 28
        ###############################################################################################################
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



    def validation_batch_end(self, outputs):
        self.val_fold_dict_epoch.append(outputs)


    def validation_epoch_end(self):
        # loss_stack = sorted(torch.stack([x['val_loss'] for x in self.val_fold_dict_epoch]))
        # sorted_val_dic = torch.stack([for s in sorted(self.val_fold_dict_epoch.iteritems(), key=lambda k_v: k_v[5]['val_loss'])])
        # sorted_val_dic = sorted(self.val_fold_dict_epoch.items(),key=lambda x:getitem(x[5],'val_loss'))
        sorted_best_loss_val_dic = sorted(self.val_fold_dict_epoch, key=lambda d: d['val_loss']) 

        get_top_last = [sorted_best_loss_val_dic[:N], sorted_best_loss_val_dic[-N:]]
        caption = [f"best_{N}", f"worst_{N}"]


        for i in range(2):
            for x in get_top_last[i]:
                self.preprocess_plot(x["file_name"][0],
                                    x["val_images_unet"].squeeze(),
                                    x["val_images_gt"].squeeze(),
                                    x["val_images_pred"].squeeze(),
                                    x["val_images_us_sim"].squeeze(),
                                    x["epoch"])
    
            self.log_image(torchvision.utils.make_grid(self.plot_figs), caption[i] +
                           "val |orig_input|us_sim|pred|gt_mask")

            # self.plot_figs.clear()
            del self.plot_figs[:]
    
        del self.val_fold_dict_epoch[:]

