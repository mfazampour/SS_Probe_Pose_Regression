import os
import numpy as np
import matplotlib.pyplot as plt
import torch

np.set_printoptions(threshold=10_0000)


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
# g_kernel = torch.tensor(g_kernel[:, :, None, None], dtype=torch.float32)
g_kernel = torch.tensor(g_kernel[None, None, :, :], dtype=torch.float32).to(device='cuda')
print(g_kernel)



def rendering(H, W, z_vals=None, attenuation_medium_map=None, refl_map=None,
boundary_map=None, mu_0_map=None, mu_1_map=None, sigma_0_map=None):

    dists = torch.abs(z_vals[..., :-1, None] - z_vals[..., 1:, None])     # dists.shape=(W, H-1, 1)
    dists = dists.squeeze(-1)                                             # dists.shape=(W, H-1)
    dists = torch.cat([dists, dists[:, -1, None]], dim=-1)                 # dists.shape=(W, H)

    attenuation = torch.exp(-attenuation_medium_map * dists)
    attenuation_total = torch.cumprod(attenuation, dim=1, dtype=torch.float32, out=None)

    attenuation_total = (attenuation_total - torch.min(attenuation_total)) / (torch.max(attenuation_total) - torch.min(attenuation_total))

    reflection_total = torch.cumprod(1. - refl_map * boundary_map, dim=1, dtype=torch.float32, out=None)
    reflection_total = reflection_total.squeeze(-1)
    reflection_total_plot = torch.log(reflection_total + torch.finfo(torch.float32).eps)

    texture_noise = torch.randn(W, H, dtype=torch.float32).to(device='cuda')
    scattering_probability = torch.randn(W, H, dtype=torch.float32).to(device='cuda')

    scattering_zero = torch.zeros(W, H, dtype=torch.float32).to(device='cuda')
    scatterers_map = torch.where(scattering_probability <= mu_0_map, 
                        texture_noise * sigma_0_map + mu_1_map, 
                        scattering_zero)

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



def get_rays_us_linear(W, sw, c2w):
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


def render_us(W, H, rays=None, near=0., far=100. * 0.001, 
              attenuation_medium_map=None, refl_map=None, boundary_map=None,
              mu_0_map=None, mu_1_map=None, sigma_0_map=None):
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
        rays_o, rays_d = get_rays_us_linear(W, sw, c2w)
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

    ret_list = rendering(H, W, z_vals, attenuation_medium_map, refl_map, boundary_map, mu_0_map, mu_1_map, sigma_0_map)    

    return ret_list 

