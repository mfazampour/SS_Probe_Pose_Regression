import torch
# import torch.nn.functional as F
import torchvision.transforms.functional as F
import monai
from cut.models.networks import GANLoss
from torch.optim import lr_scheduler
# from cut.models.base_model import BaseModel
# torch.set_printoptions(profile="full")
THRESHOLD = 0.5


class SegmentationSim(torch.nn.Module):
    def __init__(self, params, opt_cut, outer_model, inner_model, discr_model):
        super(SegmentationSim, self).__init__()
        self.params = params
        self.opt_cut = opt_cut

        if outer_model=="UNet":
            self.outer_model = monai.networks.nets.UNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            ).to(params.device)
        else:
            self.outer_model=None

        self.USRenderingModel = inner_model.to(params.device)
        self.discr_model = discr_model.to(params.device)
        

        # define loss functions
        self.segm_loss_function = monai.losses.DiceLoss(sigmoid=True)
        self.criterionGAN = GANLoss(opt_cut.gan_mode).to(params.device)

        self.seg_optimizer, self.optimizer_D, self.seg_scheduler, self.discr_scheduler = self.configure_optimizers()

        print(f'OuterModel On cuda?: ', next(self.outer_model.parameters()).is_cuda)
        print('USRenderingModel On cuda?: ', next(self.USRenderingModel.parameters()).is_cuda)
        print('discr_model On cuda?: ', next(self.discr_model.parameters()).is_cuda)


    def normalize(self, img):
        return (img - torch.min(img)) / (torch.max(img) - torch.min(img))

    
    def create_return_dict(self, type, loss, input, file_name, gt_mask, z_hat_pred, us_sim, epoch):
        dict = {f"{type}_images_unet": (input.detach()),
                'file_name': file_name,
                f'{type}_images_gt': gt_mask.detach(),
                f'{type}_images_pred': z_hat_pred.detach(),
                f'{type}_images_us_sim': us_sim.detach(),
                f'{type}_loss': loss.detach(),
                'epoch': epoch}
                
        return dict

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


    # def step(self, input, label):
    #     # print('STEPP')
    #     # UltrasoundRendering().plot_fig(input.squeeze(), "input", False)

    #     us_sim = self.USRenderingModel(input.squeeze()) 
    #     us_sim_resized = F.resize(us_sim.unsqueeze(0).unsqueeze(0), (128,128)).float()

    #     # UltrasoundRendering().plot_fig(us_sim, "us_sim", True)

    #     output = self.outer_model(us_sim_resized)
    #     # UltrasoundRendering().plot_fig(z_hat.squeeze(), "z_hat", True)

    #     # z_norm = self.normalize(z)
    #     loss = self.segm_loss_function(output, label)

    #     return loss, us_sim_resized, output          


    def training_step(self, batch_data_ct, batch_data_real_us, batch_idx=None):

        input, label, file_name = batch_data_ct[0].to(self.params.device), batch_data_ct[1].to(self.params.device), batch_data_ct[2]
        # print('FILENAME: ' + file_name)
        label = F.resize(label, (128,128)).float().unsqueeze(0)

        # self.seg_optimizer.zero_grad()

        us_sim = self.USRenderingModel(input.squeeze()) 
        us_sim_resized = F.resize(us_sim.unsqueeze(0).unsqueeze(0), (128,128)).float()
        batch_data_real_us = F.resize(batch_data_real_us, (128,128)).float()

        # update D
        self.set_requires_grad(self.discr_model, True)
        self.optimizer_D.zero_grad()
        loss_D, loss_D_real, loss_D_fake = self.compute_D_loss(fake=us_sim_resized, real=batch_data_real_us)
        loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.discr_model, False)
        self.seg_optimizer.zero_grad()
        loss_G = self.compute_G_loss(fake=us_sim_resized)

        seg_output = self.outer_model(us_sim_resized)
        # segm_loss = self.segm_loss_function(seg_output, label)

        total_loss = loss_G #+ segm_loss

        total_loss.backward()
        self.seg_optimizer.step()

        # loss, us_sim, output = self.step(input, label)

        return total_loss, loss_G, loss_D, loss_D_real, loss_D_fake


    def compute_D_loss(self, fake, real):
        """Calculate GAN loss for the discriminator"""
        fake = fake.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.discr_model(fake)
        loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        pred_real = self.discr_model(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D, loss_D_real, loss_D_fake

    def compute_G_loss(self, fake):
        """Calculate GAN and NCE loss for the generator"""
        # fake = self.fake_B
        # First, G(A) should fake the discriminator
        pred_fake = self.discr_model(fake)
        loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.params.lambda_G_loss
        
        return loss_G_GAN
            
    

    def validation_step(self, epoch, batch_data_ct, batch_data_real_us, batch_idx=None):
        # print('IN VALIDATION... ')
        input, label, file_name = batch_data_ct[0].to(self.params.device), batch_data_ct[1].to(self.params.device), batch_data_ct[2]
        # print('FILENAME: ' + file_name)
        label = F.resize(label, (128,128)).float().unsqueeze(0)

        # self.seg_optimizer.zero_grad()

        us_sim = self.USRenderingModel(input.squeeze()) 
        us_sim_resized = F.resize(us_sim.unsqueeze(0).unsqueeze(0), (128,128)).float()
        batch_data_real_us = F.resize(batch_data_real_us, (128,128)).float()

        loss_D, loss_D_real, loss_D_fake = self.compute_D_loss(fake=us_sim_resized, real=batch_data_real_us)

        # update G
        loss_G = self.compute_G_loss(fake=us_sim_resized)

        seg_output = self.outer_model(us_sim_resized)
        # segm_loss = self.segm_loss_function(seg_output, label)

        total_loss = loss_G #+ segm_loss


        val_images_plot = F.resize(input, (128,128)).float().unsqueeze(0)
        dict = self.create_return_dict('val', total_loss, val_images_plot, file_name[0], label, seg_output, us_sim_resized, epoch)

        return total_loss, loss_G, loss_D, loss_D_real, loss_D_fake, dict


    def configure_optimizers(self):

        # if self.USRenderingModel is None:
        #     seg_optimizer = torch.optim.Adam(self.outer_model.parameters(), self.params.outer_model_learning_rate)
        # else:
        #     # params = list(outer_model.parameters()) + list(inner_model.parameters())
        #     # optimizer = torch.optim.Adam(params, self.params.learning_rate)

        #     seg_optimizer = torch.optim.Adam(
        #         [
        #             {"params": self.outer_model.parameters(), "lr": self.params.outer_model_learning_rate},
        #             {"params": self.USRenderingModel.parameters(), "lr": self.params.inner_model_learning_rate},
        #         ],
        #         lr=self.params.global_learning_rate,
        #     )


        seg_optimizer = torch.optim.Adam(self.USRenderingModel.parameters(), lr=self.params.inner_model_learning_rate, betas=(self.opt_cut.beta1, self.opt_cut.beta2))
        # self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
        optimizer_D = torch.optim.Adam(self.discr_model.parameters(), lr=self.params.discr_model_learning_rate, betas=(self.opt_cut.beta1, self.opt_cut.beta2))
        # optimizer_D = torch.optim.Adam(self.discr_model.parameters(), lr=self.opt_cut.lr, betas=(self.opt_cut.beta1, self.opt_cut.beta2))

        seg_scheduler = lr_scheduler.ReduceLROnPlateau(seg_optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        discr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', factor=0.2, threshold=0.01, patience=5)


        return seg_optimizer, optimizer_D, seg_scheduler, discr_scheduler


