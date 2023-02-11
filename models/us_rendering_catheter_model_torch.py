import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

# 2 - lung; 3 - fat; 4 - vessel; 6 - kidney; 8 - muscle; 9 - background; 10 - catheter; 11 - liver; 12 - soft tissue; 13 - bone; 
# Default Parameters from: https://github.com/Blito/burgercpp/blob/master/examples/ircad11/liver.scene , labels 9 and 12 from the USHybridSim_auto_generate_imgs_masks_param2_3Splines.iws file
# indexes = 2,3,4,6,8,9,10,11,12,13
acoustic_imped_def_dict =torch.tensor([0.0004, 1.38, 1.49, 1.62, 1.62, 0.3, 0.01, 1.65, 1.63, 7.8], requires_grad=True).to(device='cuda')    # Z in MRayl
attenuation_def_dict = torch.tensor([1.64, 0.63, 0.18, 1.0, 1.09, 0.54, 0.54, 0.7, 0.54, 5.0], requires_grad=True).to(device='cuda')    # alpha in dB cm^-1 at 1 MHz

mu_0_def_dict = torch.tensor([0.78, 0.5, 0.0, 0.4, 0.4, 0.3, 1.0, 0.19, 0.64, 0.78], requires_grad=True).to(device='cuda') # mu_0 - scattering_density
mu_1_def_dict = torch.tensor([0.56, 0.5, 0.0, 0.6, 0.6, 0.2, 1.0, 1.0, 0.64, 0.56], requires_grad=True).to(device='cuda') # mu_1 - scattering_mu
sigma_0_def_dict = torch.tensor([0.1, 0.0, 0.0, 0.3, 0.3, 0.0, 1.0, 0.24, 0.1, 0.1], requires_grad=True).to(device='cuda') # sigma_0 - scattering_sigma

alpha_coeff_boundary_map = 0.1
beta_coeff_scattering = 10


def gaussian_kernel(size: int, mean: float, std: float):
    delta_t = 1
    x_cos = np.array(list(range(-size, size+1)), dtype=np.float32)
    x_cos *= delta_t

    d1 = torch.distributions.Normal(mean, std*3)
    d2 = torch.distributions.Normal(mean, std)
    vals_x = d1.log_prob(torch.arange(-size, size+1, dtype=torch.float32)*delta_t).exp()
    vals_y = d2.log_prob(torch.arange(-size, size+1, dtype=torch.float32)*delta_t).exp()

    gauss_kernel = torch.einsum('i,j->ij', vals_x, vals_y)
    
    return gauss_kernel / torch.sum(gauss_kernel).reshape(1, 1)

g_kernel = gaussian_kernel(3, 0., 1.)
g_kernel = torch.tensor(g_kernel[None, None, :, :], dtype=torch.float32).to(device='cuda')


class UltrasoundRendering(torch.nn.Module):
    def __init__(self):
        super(UltrasoundRendering, self).__init__()

        # self.save_hyperparameters()
        self.acoustic_impedance_dict = torch.nn.Parameter(acoustic_imped_def_dict)#.to(device='cuda')
        self.attenuation_dict = torch.nn.Parameter(attenuation_def_dict)#.to(device='cuda')
        self.mu_0_dict = torch.nn.Parameter(mu_0_def_dict)
        self.mu_1_dict = torch.nn.Parameter(mu_1_def_dict)
        self.sigma_0_dict = torch.nn.Parameter(sigma_0_def_dict)

        # self.acoustic_impedance_dict = acoustic_imped_def_dict
        # self.attenuation_dict = attenuation_def_dict
        # self.mu_0_dict = mu_0_def_dict
        # self.mu_1_dict = mu_1_def_dict
        # self.sigma_0_dict = sigma_0_def_dict



    def map_dict_to_array(self, dictionary, arr):
        mapping_keys = torch.tensor([2, 3, 4, 6, 8, 9, 10, 11, 12, 13], dtype=torch.long).to(device='cuda')
        keys = torch.unique(arr)
        # print('keys: ', keys.requires_grad)

        index = torch.where(mapping_keys[None, :] == keys[:, None])[1]
        values = torch.gather(dictionary, dim=0, index=index)
        values = values.to(device='cuda')
        # values.register_hook(lambda grad: print(grad))
        # print('values: ', values.requires_grad)

        mapping = torch.zeros(keys.max().item() + 1).to(device='cuda')
        mapping[keys] = values
        # print('mapping: ', mapping.requires_grad)
        return mapping[arr]


    # def map_dict_to_array(self, dictionary, arr):
        
    #     mapping_dict = {2: dictionary[0], 3: dictionary[1], 4: dictionary[2], 6: dictionary[3], 
    #                     8: dictionary[4], 9: dictionary[5], 11: dictionary[6], 12: dictionary[7], 13: dictionary[8]}

    #     keys = torch.unique(arr)
    #     print('keys: ', keys.requires_grad)

    #     values = torch.tensor([mapping_dict[key.item()] for key in keys], requires_grad=True).to(device='cuda')
    #     values.register_hook(lambda grad: print(grad))

    #     print('values: ', values.requires_grad)

    #     mapping = torch.zeros(keys.max().item() + 1).to(device='cuda')#, dtype=torch.int64)
    #     mapping[keys] = values
    #     print('mapping: ', mapping.requires_grad)

    #     return mapping[arr]


    def plot_fig(self, fig, fig_name, grayscale):
        save_dir='results_test/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        plt.clf()

        if torch.is_tensor(fig):
            # fig = fig.cpu().data.numpy()
            fig = fig.cpu().detach().numpy()
        fig = np.rot90(fig, 3)

        if grayscale:
            plt.imshow(fig, cmap='gray', interpolation='nearest')
        else:
            plt.imshow(fig, interpolation='none')
        plt.colorbar()
        plt.savefig(save_dir + fig_name + '.png')



    def rendering(self, H, W, z_vals=None, attenuation_medium_map=None, refl_map=None,
                    boundary_map=None, mu_0_map=None, mu_1_map=None, sigma_0_map=None):

        torch.manual_seed(2023)
        dists = torch.abs(z_vals[..., :-1, None] - z_vals[..., 1:, None])     # dists.shape=(W, H-1, 1)
        dists = dists.squeeze(-1)                                             # dists.shape=(W, H-1)
        dists = torch.cat([dists, dists[:, -1, None]], dim=-1)                 # dists.shape=(W, H)

        attenuation = torch.exp(-attenuation_medium_map * dists)
        attenuation_total = torch.cumprod(attenuation, dim=1, dtype=torch.float32, out=None)

        attenuation_total = (attenuation_total - torch.min(attenuation_total)) / (torch.max(attenuation_total) - torch.min(attenuation_total))

        reflection_total = torch.cumprod(1. - refl_map * boundary_map, dim=1, dtype=torch.float32, out=None)
        reflection_total = reflection_total.squeeze(-1)
        reflection_total_plot = torch.log(reflection_total + torch.finfo(torch.float32).eps)

        texture_noise = torch.randn(H, W, dtype=torch.float32).to(device='cuda')
        scattering_probability = torch.randn(H, W, dtype=torch.float32).to(device='cuda')

        scattering_zero = torch.zeros(H, W, dtype=torch.float32).to(device='cuda')

        z = mu_0_map - scattering_probability
        sigmoid_map = torch.sigmoid(beta_coeff_scattering * z)
        scatterers_map =  (1 - sigmoid_map) * (texture_noise * sigma_0_map + mu_1_map) + sigmoid_map * scattering_zero

        # scatterers_map = torch.where(scattering_probability <= mu_0_map, 
        #                     texture_noise * sigma_0_map + mu_1_map, 
        #                     scattering_zero)

        psf_scatter_conv = torch.nn.functional.conv2d(input=scatterers_map[None, None, :, :], weight=g_kernel, stride=1, padding="same")

        # psf_scatter_conv = torch.nn.functional.conv2d(input=scatterers_map[None, :, :, None], weight=g_kernel, stride=1, padding=1)

        psf_scatter_conv = psf_scatter_conv.squeeze()

        b = attenuation_total * psf_scatter_conv

        border_convolution = torch.nn.functional.conv2d(input=boundary_map[None, None, :, :], weight=g_kernel, stride=1, padding="same")
        border_convolution = border_convolution.squeeze()

        r = attenuation_total * reflection_total * refl_map * border_convolution
        intensity_map = b + r
        intensity_map = intensity_map.squeeze()
        intensity_map = torch.clamp(intensity_map, 0, 1)

        return intensity_map, attenuation_total, reflection_total_plot, scatterers_map, scattering_probability, border_convolution, texture_noise, b, r


    def get_rays_us_linear(self, W, sw, c2w):
        t = torch.Tensor(c2w[:3, -1])
        R = torch.Tensor(c2w[:3, :3])
        i = torch.arange(W, dtype=torch.float32)
        rays_o_x = t[0] + sw * i
        rays_o_y = torch.full_like(rays_o_x, t[1])
        rays = torch.stack([rays_o_x, rays_o_y, torch.ones_like(rays_o_x) * t[2]], -1) #x,y,z
        shift = torch.matmul(R, torch.tensor([25., 27.5, 0.], dtype=torch.float32)) * 0.001 #What are these constants????
        rays_o = rays - shift
        dirs = torch.stack([torch.zeros_like(rays_o_x), torch.ones_like(rays_o_x), torch.zeros_like(rays_o_x)], -1)
        rays_d = torch.sum(dirs[..., None, :] * R, -1)

        return rays_o.to(device='cuda'), rays_d.to(device='cuda')


    def render_rays(self, W, H, rays=None, near=0., far=100. * 0.001):
        """Render rays

        Args:
        H: int. Height of image in pixels.
        W: int. Width of image in pixels.
        rays: array of shape [2, batch_size, 3]. Ray origin and direction for
            each example in batch.
        c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
        near: float or array of shape [batch_size]. Nearest distance for a ray.
        far: float or array of shape [batch_size]. Farthest distance for a ray.
        """
        sw = 40 * 0.001 / float(W)
        c2w = np.eye(4)[:3,:4].astype(np.float32) # identity pose matrix

        if c2w is not None:
            # special case to render full image
            rays_o, rays_d = self.get_rays_us_linear(W, sw, c2w)
        else:
            # use provided ray batch
            rays_o, rays_d = rays


        # Create ray batch
        rays_o = torch.tensor(rays_o).view(-1, 3).float()
        rays_d = torch.tensor(rays_d).view(-1, 3).float()
        near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])

        rays = torch.cat([rays_o, rays_d, near, far], dim=-1)

        N_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]

        t_vals = torch.linspace(0., 1., H).to(device='cuda')

        z_vals = t_vals.unsqueeze(0).expand(N_rays, -1) * 2


        return z_vals 


    def forward(self, ct_slice):
        # self.plot_fig(ct_slice, "ct_slice", False)
        
        # self.acoustic_impedance_dict.register_hook(lambda grad: print(grad))
        # self.attenuation_dict.register_hook(lambda grad: print(grad))
        # self.mu_0_dict.register_hook(lambda grad: print(grad))
        # self.mu_1_dict.register_hook(lambda grad: print(grad))
        # self.sigma_0_dict.register_hook(lambda grad: print(grad))

        #init tissue maps
        #generate 2D acousttic_imped map
        acoustic_imped_map = self.map_dict_to_array(self.acoustic_impedance_dict, ct_slice)#.astype('int64'))
        # print('acoustic_imped_map: ', acoustic_imped_map.requires_grad)
        # self.plot_fig(acoustic_imped_map, "acoustic_imped_map", False)

        #generate 2D attenuation map
        attenuation_medium_map = self.map_dict_to_array(self.attenuation_dict, ct_slice)
        # print('attenuation_medium_map: ', attenuation_medium_map.requires_grad)
        # self.plot_fig(attenuation_medium_map, "attenuation_medium_map", False)

        mu_0_map = self.map_dict_to_array(self.mu_0_dict, ct_slice)
        # print('mu_0_map: ', mu_0_map.requires_grad)

        mu_1_map = self.map_dict_to_array(self.mu_1_dict, ct_slice)
        # print('mu_1_map: ', mu_1_map.requires_grad)

        sigma_0_map = self.map_dict_to_array(self.sigma_0_dict, ct_slice)
        # print('sigma_0_map: ', sigma_0_map.requires_grad)


        acoustic_imped_map = torch.rot90(acoustic_imped_map, 1, [0, 1])
        diff_arr = torch.diff(acoustic_imped_map, dim=0)
        # print('diff_arr: ', diff_arr.requires_grad)

        diff_arr = torch.cat((torch.zeros(diff_arr.shape[1], dtype=torch.float32).unsqueeze(0).to(device='cuda'), diff_arr))
        # print('diff_arr2: ', diff_arr.requires_grad)

        # boundary_map = torch.where(diff_arr != 0., torch.tensor(1., dtype=torch.float32).to(device='cuda'), torch.tensor(0., dtype=torch.float32).to(device='cuda'))

        boundary_map =  -torch.exp(-(diff_arr**2)/alpha_coeff_boundary_map) + 1
        # print('boundary_map: ', boundary_map.requires_grad)

        boundary_map = torch.rot90(boundary_map, 3, [0, 1])

        # self.plot_fig(diff_arr, "diff_arr", False)
        # self.plot_fig(boundary_map, "boundary_map", True)

        shifted_arr = torch.roll(acoustic_imped_map, -1, dims=0)
        shifted_arr[-1:] = 0

        sum_arr = acoustic_imped_map + shifted_arr
        sum_arr[sum_arr == 0] = 1
        div = diff_arr / sum_arr

        refl_map = div ** 2
        refl_map = torch.sigmoid(refl_map)      # 1 / (1 + (-refl_map).exp())
        refl_map = torch.rot90(refl_map, 3, [0, 1])

        # self.plot_fig(refl_map, "refl_map", True)

        z_vals = self.render_rays(ct_slice.shape[0], ct_slice.shape[1])


        # attenuation_medium_map.register_hook(lambda grad: print(sum(grad)))

        ret_list = self.rendering(ct_slice.shape[0], ct_slice.shape[1], z_vals=z_vals,
            attenuation_medium_map=attenuation_medium_map, refl_map=refl_map, boundary_map=boundary_map, 
            mu_0_map=mu_0_map, mu_1_map=mu_1_map, sigma_0_map=sigma_0_map)

        self.intensity_map  = ret_list[0]

        result_list = ["intensity_map", "attenuation_total", "reflection_total", 
                        "scatters_map", "scattering_probability", "border_convolution", 
                        "texture_noise", "b", "r"]

        # for k in range(len(ret_list)):
        #     result_np = ret_list[k]
        #     if torch.is_tensor(result_np):
        #         result_np = result_np.cpu().numpy()
                     
        #     if k==2:
        #         self.plot_fig(result_np, result_list[k], False)
        #     else:
        #         self.plot_fig(result_np, result_list[k], True)
        #     print(result_list[k], ", ", result_np.shape)

        us_mask = Image.open('us_convex_mask.png')
        us_mask = us_mask.resize((ret_list[0].shape[0], ret_list[0].shape[1]))
        us_mask = transforms.ToTensor()(us_mask).squeeze().to(device='cuda')
        self.intensity_map_masked = self.intensity_map  * torch.transpose(us_mask, 0, 1) #np.transpose(us_mask)
        # self.plot_fig(intensity_map_masked, "intensity_map_masked", True)

        return self.intensity_map_masked

