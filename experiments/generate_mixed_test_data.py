import sys
sys.path.append("/home/jia/Desktop/MSc_Project/imagetemplate/FedLIT") 
from data import data_util
import numpy as np
import os
import torch
import torch.nn.functional as F
from exp_config import CLASSIFY_CONFIG
import random






def load_train_data(dataset, size, save_path):

    new_dataset = []
    for image, label in dataset:
        if label in CLASSIFY_CONFIG['numbers']:
            if size != 28:
                # Convert the image to float data type
                image = image.unsqueeze(0).unsqueeze(1).float()
                
                # Apply bilinear interpolation to upsample the image
                scaled_image = F.interpolate(image, size=size, mode='bilinear', align_corners=False)
                
                # Convert the upsampled image tensor back to 'Byte'
                scaled_image = scaled_image.squeeze().to(torch.uint8)
            else:
                scaled_image = image

            # Add the upsampled image to the list
            new_dataset.append((scaled_image.numpy(), label.item()))
    np.random.shuffle(new_dataset)
    selected_data = random.sample(new_dataset, CLASSIFY_CONFIG['train_data_size'] )
    np.save(save_path+'/train_data', selected_data)
    


def load_mixed_test_data(datasets, size, save_path):

    new_dataset = []   
    for dataset in datasets:
        for image, label in dataset:
            if label in CLASSIFY_CONFIG['numbers']:
                if size != 28:
                    # Convert the image to float data type
                    image = image.unsqueeze(0).unsqueeze(1).float()
                    
                    # Apply bilinear interpolation to upsample the image
                    scaled_image = F.interpolate(image, size=size, mode='bilinear', align_corners=False)
                    
                    # Convert the upsampled image tensor back to 'Byte'
                    scaled_image = scaled_image.squeeze().to(torch.uint8)
                else:
                    scaled_image = image

                # Add the upsampled image to the list
                new_dataset.append((scaled_image.numpy(), label.item()))
    np.random.shuffle(new_dataset)
    selected_data = random.sample(new_dataset, CLASSIFY_CONFIG['mixed_test_data_size'] )
    np.save(save_path+'/mixed_test_data', selected_data)




    
if __name__ == '__main__':
    MODEL_WEIGHT_BACKUP_PATH = "/home/jia/Desktop/MSc_Project/imagetemplate/version8/output"
    data_path = "/home/jia/Desktop/MSc_Project/imagetemplate/version8/origin_data"
    save_path = "/home/jia/Desktop/MSc_Project/imagetemplate/version8/classifier_data"

    client1_path = save_path + '/' + 'client1'
    client2_path = save_path + '/' + 'client2'
    client3_path = save_path + '/' + 'client3'
    client4_path = save_path + '/' + 'client4'
    client5_path = save_path + '/' + 'client5'

    for path in [client1_path, client2_path, client3_path, client4_path, client5_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    raw_data_dirs = [os.path.join(data_path, 'raw')]
    frac_data_dirs = [os.path.join(data_path, 'frac')]
    thin_data_dirs = [os.path.join(data_path, 'thin')]
    thic_data_dirs = [os.path.join(data_path, 'thic')]
    swel_data_dirs = [os.path.join(data_path, 'swel')]

    ### local train data

    thin_train_set = data_util.get_dataset(thin_data_dirs, train=True)
    thic_train_set = data_util.get_dataset(thic_data_dirs, train=True)
    raw_train_set = data_util.get_dataset(raw_data_dirs, train=True)
    swel_train_set = data_util.get_dataset(swel_data_dirs, train=True)
    frac_train_set = data_util.get_dataset(frac_data_dirs, train=True)

    load_train_data(thin_train_set, 56, client1_path )
    load_train_data(thic_train_set, 56, client2_path )
    load_train_data(raw_train_set, 28, client3_path )
    load_train_data(swel_train_set, 42, client4_path )
    load_train_data(frac_train_set, 42, client5_path )

    ### mixed test data
    thin_test_set = data_util.get_dataset(thin_data_dirs, train=False)
    thic_test_set = data_util.get_dataset(thic_data_dirs, train=False)
    raw_test_set = data_util.get_dataset(raw_data_dirs, train=False)
    swel_test_set = data_util.get_dataset(swel_data_dirs, train=False)
    frac_test_set = data_util.get_dataset(frac_data_dirs, train=False)

    datasets = [thin_test_set, thic_test_set, raw_test_set, swel_test_set, frac_test_set]
    load_mixed_test_data(datasets, 56, client1_path)
    load_mixed_test_data(datasets, 56, client2_path)
    load_mixed_test_data(datasets, 28, client3_path)
    load_mixed_test_data(datasets, 42, client4_path)
    load_mixed_test_data(datasets, 42, client5_path)





# data = np.load('/home/jia/Desktop/MSc_Project/imagetemplate/version4/classifier_data/client5/mixed_test_data.npy', allow_pickle=True)
# x, y = zip(*data)
# print(len(x)==len(y))
# print(set(y))
# print(type(y[0]))
# print(np.shape(x[0]))



