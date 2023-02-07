import torch
# import torch.nn.functional as F
import torchvision.transforms.functional as F
import monai
# torch.set_printoptions(profile="full")
THRESHOLD = 0.5


class SegmentationSim(torch.nn.Module):
    def __init__(self, params, outer_model, inner_model):
        super(SegmentationSim, self).__init__()
        self.params = params

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
        self.loss_function = monai.losses.DiceLoss(sigmoid=True)
        self.optimizer = self.configure_optimizers(self.USRenderingModel, self.outer_model)

        print(f'OuterModel On cuda?: ', next(self.outer_model.parameters()).is_cuda)
        print('USRenderingModel On cuda?: ', next(self.USRenderingModel.parameters()).is_cuda)


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


    def step(self, input, label):
        # print('STEPP')
        # UltrasoundRendering().plot_fig(input.squeeze(), "input", False)

        us_sim = self.USRenderingModel(input.squeeze()) 
        us_sim_resized = F.resize(us_sim.unsqueeze(0).unsqueeze(0), (128,128)).float()

        # UltrasoundRendering().plot_fig(us_sim, "us_sim", True)

        output = self.outer_model(us_sim_resized)
        # UltrasoundRendering().plot_fig(z_hat.squeeze(), "z_hat", True)

        # z_norm = self.normalize(z)
        loss = self.loss_function(output, label)

        return loss, us_sim_resized, output          


    def training_step(self, batch_data, batch_idx=None):

        input, label, file_name = batch_data[0].to(self.params.device), batch_data[1].to(self.params.device), batch_data[2]
        # print('FILENAME: ' + file_name)
        label = F.resize(label, (128,128)).float().unsqueeze(0)

        self.optimizer.zero_grad()
        loss, us_sim, output = self.step(input, label)

        return loss
    

    def validation_step(self, batch_data, epoch, batch_idx=None):
        # print('IN VALIDATION... ')
        input, label, file_name = batch_data[0].to(self.params.device), batch_data[1].to(self.params.device), batch_data[2]
        label = F.resize(label, (128,128)).float().unsqueeze(0)

        loss, us_sim_resized, pred = self.step(input, label)

        val_images_plot = F.resize(input, (128,128)).float().unsqueeze(0)
        dict = self.create_return_dict('val', loss, val_images_plot, file_name[0], label, pred, us_sim_resized, epoch)

        return loss, dict


    def configure_optimizers(self, inner_model, outer_model):

        if inner_model is None:
            optimizer = torch.optim.Adam(outer_model.parameters(), self.params.outer_model_learning_rate)
        else:
            # params = list(outer_model.parameters()) + list(inner_model.parameters())
            # optimizer = torch.optim.Adam(params, self.params.learning_rate)

            optimizer = torch.optim.Adam(
                [
                    {"params": outer_model.parameters(), "lr": self.params.outer_model_learning_rate},
                    {"params": inner_model.parameters(), "lr": self.params.inner_model_learning_rate},
                ],
                lr=self.params.global_learning_rate,
            )

        return optimizer    


