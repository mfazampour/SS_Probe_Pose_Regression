import os
import configargparse
from configargparse_arguments import build_configargparser
import nibabel as nib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import ultrasound_rendering
import torch
from PIL import Image
from torchvision import transforms


# ------------------------
# CONFIG ARGUMENTS
# ------------------------
parser = configargparse.ArgParser(
    config_file_parser_class=configargparse.YAMLConfigFileParser)

parser.add('-c', is_config_file=True, help='config file path')
parser, hparams = build_configargparser(parser)

# SAVE_DIR_US_SIM = hparams.save_dir_us_sim

BASE_FOLDER_DATA = hparams.base_folder_data_path
sub_folder_CT = [sub_f for sub_f in sorted(os.listdir(BASE_FOLDER_DATA))]
full_labelmap_path = [BASE_FOLDER_DATA + s + hparams.labelmap_path for s in sub_folder_CT]
# labelmap = [l.endswith('.nii.gz') for l in full_labelmap_path]



# Each value n the labelmap corresponds to a tissue as follows:
# 2 - lung; 3 - fat; 4 - vessel; 6 - kidney; 8 - muscle; 11 - liver; 12 - soft tissue; 13 - bone; 9 - background
# acoustic_impedance_dict = {2: 0.0004, 3: 1.40, 4: 1.68, 6: 1.65, 8: 1.62, 9: 0.6, 11: 1.69, 12: 1.69, 13: 6.2}    # Z in MRayl
# attenuation_dict = {2: 1.38, 3: 0.48, 4: 0.2, 6: 1.0, 8: 1.09, 9: 0.54, 11: 0.5, 12: 0.75, 13: 7}    # alpha in dB cm^-1 at 1 MHz

# Parameters from: https://github.com/Blito/burgercpp/blob/master/examples/ircad11/liver.scene , labels 9 and 12 from the USHybridSim_auto_generate_imgs_masks_param2_3Splines.iws file
acoustic_impedance_dict =   {2: 0.0004, 3: 1.38, 4: 1.61, 6: 1.62, 8: 1.62, 9: 0.3, 11: 1.65, 12: 1.63, 13: 7.8}    # Z in MRayl
attenuation_dict =          {2: 1.64,   3: 0.63, 4: 0.18, 6: 1.0,  8: 1.09, 9: 0.54, 11: 0.7, 12: 0.54, 13: 5.0}    # alpha in dB cm^-1 at 1 MHz

mu_0_dict =                 {2: 0.78, 3: 0.5, 4: 0.001, 6: 0.4, 8: 0.4, 9: 0.3, 11: 0.19, 12: 0.64, 13: 0.78} # mu_0 - scattering_density
mu_1_dict =                 {2: 0.56, 3: 0.5, 4: 0.0,  6: 0.6,  8: 0.6, 9: 0.2, 11: 1.0,  12: 0.64, 13: 0.56} # mu_1 - scattering_mu
sigma_0_dict =              {2: 0.1,  3: 0.0, 4: 0.01, 6: 0.3,  8: 0.3, 9: 0.0, 11: 0.24, 12: 0.1,  13: 0.1} # sigma_0 - scattering_sigma
 



def map_dict_to_array(dictionary, arr):
    return torch.tensor(np.vectorize(dictionary.get)(arr).astype('float32')).to(device='cuda')

    # k = torch.tensor(list(dictionary.keys()))
    # v = torch.tensor(list(dictionary.values()))
    # mapping_ar = torch.zeros(k.max() + 1, dtype=torch.float32)
    # mapping_ar[k] = v

    # mapping_arr = [dictionary[k] for k in arr]

    # return mapping_ar[arr].to(device='cuda')


def plot_fig(fig, fig_name, grayscale, save_dir):
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

    

# H = 150 #267 #599 
# W = 250 #363 #549 

# near = 0.
# far = 100 * 0.001

# sw = 40 * 0.001 / float(W)  # 40 mm
# sh = 100 * 0.001 /float(H)  # 100 mm

if __name__ == '__main__':

    save_dir = None

    for folder in full_labelmap_path:
        labelmap = [lm for lm in sorted(os.listdir(folder)) if lm.endswith('.nii.gz') and "_" not in lm][0]

        img = nib.load(folder + labelmap)
        data = img.get_fdata()
        # # data = np.swapaxes(data, 0, 1)
        # data = np.flip(data, 0)


        # for slice in range(data.shape[0]):
        # slice = 130
        for slice in range(data.shape[2]):
            save_dir = 'results/' + labelmap.split('.')[0] + '/' + str(slice) + '/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # ct_slice = (data[50:300, 50:200, slice]).astype(np.int64)
            ct_slice = (data[:, :, slice]).astype(np.int64)

            plot_fig(ct_slice, "ct_slice", False, save_dir)

            #generate 2D acousttic_imped map
            acoustic_imped_map = map_dict_to_array(acoustic_impedance_dict, ct_slice)
            #acoustic_imped_map *= acoustic_imped_map * 100000.00
            plot_fig(acoustic_imped_map, "acoustic_imped_map", False, save_dir)

            #generate 2D attenuation map
            attenuation_medium_map = map_dict_to_array(attenuation_dict, ct_slice)
            plot_fig(attenuation_medium_map, "attenuation_medium_map", False, save_dir)

            acoustic_imped_map = torch.rot90(acoustic_imped_map, 1, [0, 1])
            diff_arr = torch.diff(acoustic_imped_map, dim=0)
            diff_arr = torch.cat((torch.zeros(diff_arr.shape[1], dtype=torch.float32).unsqueeze(0).to(device='cuda'), diff_arr))
            boundary_map = torch.where(diff_arr != 0., torch.tensor(1., dtype=torch.float32).to(device='cuda'), torch.tensor(0., dtype=torch.float32).to(device='cuda'))
            boundary_map = torch.rot90(boundary_map, 3, [0, 1])

            plot_fig(diff_arr, "diff_arr", False, save_dir)
            plot_fig(boundary_map, "boundary_map", True, save_dir)


            shifted_arr = torch.roll(acoustic_imped_map, -1, dims=0)
            shifted_arr[-1:] = 0

            sum_arr = acoustic_imped_map + shifted_arr
            sum_arr[sum_arr == 0] = 1
            div = diff_arr / sum_arr

            refl_map = div ** 2
            refl_map = torch.sigmoid(refl_map)      # 1 / (1 + (-refl_map).exp())
            refl_map = torch.rot90(refl_map, 3, [0, 1])

            plot_fig(refl_map, "refl_map", True, save_dir)

            mu_0_map = map_dict_to_array(mu_0_dict, ct_slice)
            mu_1_map = map_dict_to_array(mu_1_dict, ct_slice)
            sigma_0_map = map_dict_to_array(sigma_0_dict, ct_slice)


            test = ultrasound_rendering.render_us(ct_slice.shape[0], ct_slice.shape[1],
                        attenuation_medium_map=attenuation_medium_map, refl_map=refl_map, boundary_map=boundary_map, 
                        mu_0_map=mu_0_map, mu_1_map=mu_1_map, sigma_0_map=sigma_0_map)


            result_list = ["intensity_map", "attenuation_total", "reflection_total", 
                            "scatters_map", "scattering_probability", "border_convolution", 
                            "texture_noise", "b", "r"]

            for k in range(len(test)):
                result_np = test[k]
                if torch.is_tensor(result_np):
                    result_np = result_np.cpu().detach().numpy()
                
                if k==2:
                    plot_fig(result_np, result_list[k], False, save_dir)
                else:
                    plot_fig(result_np, result_list[k], True, save_dir)
                print(result_list[k], ", ", result_np.shape)

            us_mask = Image.open('us_convex_mask.png')
            us_mask = us_mask.resize((test[0].shape[0], test[0].shape[1]))
            us_mask = transforms.ToTensor()(us_mask).squeeze().to(device='cuda')
            intensity_map_masked = test[0] * torch.transpose(us_mask, 0, 1) #np.transpose(us_mask)
            plot_fig(intensity_map_masked, "intensity_map_masked", True, save_dir)


