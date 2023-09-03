def rendering(self, H, W, z_vals=None, refl_map=None, boundary_map=None):
        
        dists = torch.abs(z_vals[..., :-1, None] - z_vals[..., 1:, None])     # dists.shape=(W, H-1, 1)
        dists = dists.squeeze(-1)                                             # dists.shape=(W, H-1)
        dists = torch.cat([dists, dists[:, -1, None]], dim=-1)                # dists.shape=(W, H)

        attenuation = torch.exp(-self.attenuation_medium_map * dists)
        attenuation_total = torch.cumprod(attenuation, dim=1, dtype=torch.float32, out=None)

        gain_coeffs = np.linspace(1, TGC, attenuation_total.shape[1])
        gain_coeffs = np.tile(gain_coeffs, (attenuation_total.shape[0], 1))
        gain_coeffs = torch.tensor(gain_coeffs).to(device='cuda') 
        attenuation_total = attenuation_total * gain_coeffs     #apply TGC

        reflection_total = torch.cumprod(1. - refl_map * boundary_map, dim=1, dtype=torch.float32, out=None) 
        reflection_total = reflection_total.squeeze(-1) 
        reflection_total_plot = torch.log(reflection_total + torch.finfo(torch.float32).eps)

        texture_noise = torch.randn(H, W, dtype=torch.float32).to(device='cuda')
        scattering_probability = torch.randn(H, W, dtype=torch.float32).to(device='cuda') 

        scattering_zero = torch.zeros(H, W, dtype=torch.float32).to(device='cuda')


        z = self.mu_1_map - scattering_probability
        sigmoid_map = torch.sigmoid(beta_coeff_scattering * z)
        scatterers_map =  (sigmoid_map) * (texture_noise * self.sigma_0_map + self.mu_0_map) #+ (1 -sigmoid_map) * scattering_zero

        # scatterers_map = torch.where(scattering_probability <= mu_1_map, 
        #                     texture_noise * sigma_0_map + mu_0_map, 
        #                     scattering_zero)

        psf_scatter_conv = torch.nn.functional.conv2d(input=scatterers_map[None, None, :, :], weight=g_kernel, stride=1, padding="same")

        # psf_scatter_conv = torch.nn.functional.conv2d(input=scatterers_map[None, :, :, None], weight=g_kernel, stride=1, padding=1)

        psf_scatter_conv = psf_scatter_conv.squeeze()

        b = attenuation_total * psf_scatter_conv #* reflection_total

        border_convolution = torch.nn.functional.conv2d(input=boundary_map[None, None, :, :], weight=g_kernel, stride=1, padding="same")
        border_convolution = border_convolution.squeeze()

        r = attenuation_total * reflection_total * refl_map * border_convolution 
        if self.params.cactuss_mode:
            intensity_map = r   #b + r 
        else:
            intensity_map = b + r 
        intensity_map = intensity_map.squeeze() 
        intensity_map = torch.clamp(intensity_map, 0, 1)


        # b2 = attenuation_total_TGC * psf_scatter_conv
        # r2 = attenuation_total_TGC * reflection_total * refl_map * border_convolution
        # intensity_map2 = b2 + r2
        # intensity_map2 = intensity_map2.squeeze()
        # intensity_map = torch.clamp(intensity_map, 0, 1)


        return intensity_map, attenuation_total, reflection_total_plot, scatterers_map, scattering_probability, border_convolution, texture_noise, b, r
