import gzip
from morphomnist import io
import numpy as np
import os
import torch
import torch.nn.functional as F
import shutil

def copy_and_rename(src_dir, dest_dir):
    """
    Copies the contents of src_dir to dest_dir and renames the directory.
    
    Parameters:
    - src_dir (str): Source directory path.
    - dest_dir (str): Destination directory path.
    """
    if not os.path.exists(dest_dir):
        shutil.copytree(src_dir, dest_dir)
    else:
        print(f"Directory {dest_dir} already exists!")


def scale(file_path, train, size):
    prefix = "train" if train else "t10k"
    images_filename = prefix + "-images-idx3-ubyte.gz"
    dataset = torch.from_numpy(io.load_idx(os.path.join(file_path, images_filename)))
    scaled_data = []
    for image in dataset:
        # Convert the image to float data type
        image = image.unsqueeze(0).unsqueeze(1).float()
        
        # Apply bilinear interpolation to upsample the image
        scaled_image = F.interpolate(image, size=size, mode='bilinear', align_corners=False)
        
        # Convert the upsampled image tensor back to 'Byte'
        scaled_image = scaled_image.squeeze().to(torch.uint8)
        
        # Add the upsampled image to the list
        scaled_data.append(scaled_image.numpy())
    
    scaled_data = np.stack(scaled_data, axis=0)

    # with gzip.open(os.path.join(file_path, 'test.zip'), 'wb') as f:
    #     np.save(upsampled_data, f)
    io.save_idx(scaled_data, os.path.join(file_path, images_filename))




def simulate_multiresol_dataset(origin_data_path, new_data_path):

    copy_and_rename(origin_data_path, new_data_path)

    raw_data_dirs = os.path.join(new_data_path, 'raw')
    frac_data_dirs = os.path.join(new_data_path, 'frac')
    thin_data_dirs = os.path.join(new_data_path, 'thin')
    thic_data_dirs = os.path.join(new_data_path, 'thic')
    swel_data_dirs = os.path.join(new_data_path, 'swel')
    
    scale(thin_data_dirs, True, 56)
    scale(thin_data_dirs, False, 56)

    scale(thic_data_dirs, True, 56)
    scale(thic_data_dirs, False, 56)

    scale(swel_data_dirs, True, 42)
    scale(swel_data_dirs, False, 42)

    scale(frac_data_dirs, True, 42)
    scale(frac_data_dirs, False, 42)









