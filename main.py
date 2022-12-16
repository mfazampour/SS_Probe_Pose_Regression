import os
import configargparse
from configargparse_arguments import build_configargparser
import nibabel as nib
import cv2
import numpy as np

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

# folder_catheter_data = [FOLDER_CATH_DATA + s for s in sub_folders_cath]


if __name__ == '__main__':
    for folder in full_labelmap_path:
        # sub_folder_CT = [lm.endswith('.nii.gz')[0] for lm in sorted(os.listdir(folder))]

        labelmap = [lm for lm in sorted(os.listdir(folder)) if lm.endswith('.nii.gz') and "_" not in lm][0]
        # lm = sorted(os.listdir(folder))  # .endswith('.nii.gz')

        img = nib.load(folder + labelmap)
        data = img.get_fdata()
        data = np.swapaxes(data, 0, 2)

        print(labelmap)
        cv2.imshow("image", (data[130, :, :] * 155.0).astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
