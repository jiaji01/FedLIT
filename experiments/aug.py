import sys
sys.path.append("/home/jia/Desktop/MSc_Project/imagetemplate/FedLIT") 
from vae import VAE1, VAE2, VAE3
from helper import cast_data, get_loader
from config import CONFIG, AUG
import torch
import numpy as np
from helper import get_loader
from scipy.stats import entropy
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.special import kl_div
from exp_config import CLASSIFY_CONFIG

np.random.seed(35813)
torch.manual_seed(35813)
### Setting up working environment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Direct to working environment
MODEL_WEIGHT_BACKUP_PATH = "/home/jia/Desktop/MSc_Project/imagetemplate/FedLIT/output"
data_path = "/home/jia/Desktop/MSc_Project/imagetemplate/FedLIT/data"

### import data
import os
from data import data_util

thin_data_dirs = [os.path.join(data_path, 'thin')]
thic_data_dirs = [os.path.join(data_path, 'thic')]
raw_data_dirs = [os.path.join(data_path, 'raw')]
swel_data_dirs = [os.path.join(data_path, 'swel')]
frac_data_dirs = [os.path.join(data_path, 'frac')]


# Thin MNIST
thin_train_set = data_util.get_dataset(thin_data_dirs, train=True)
thin_test_set = data_util.get_dataset(thin_data_dirs, train=False)
# Thic MNIST
thic_train_set = data_util.get_dataset(thic_data_dirs, train=True)
thic_test_set = data_util.get_dataset(thic_data_dirs, train=False)
# Raw MNIST
raw_train_set = data_util.get_dataset(raw_data_dirs, train=True)
raw_test_set = data_util.get_dataset(raw_data_dirs, train=False)
# Swel MNIST
swel_train_set = data_util.get_dataset(swel_data_dirs, train=True)
swel_test_set = data_util.get_dataset(swel_data_dirs, train=False)
# Frac MNIST
frac_train_set = data_util.get_dataset(frac_data_dirs, train=True)
frac_test_set = data_util.get_dataset(frac_data_dirs, train=False)


raw_train = cast_data(raw_train_set, CONFIG['number'])[0:3000]
raw_test = cast_data(raw_test_set, CONFIG['number'])[0:1000]

frac_train = cast_data(frac_train_set, CONFIG['number'])[0:3000]
frac_test = cast_data(frac_test_set, CONFIG['number'])[0:1000]

thin_train = cast_data(thin_train_set, CONFIG['number'])[0:3000]
thin_test = cast_data(thin_test_set, CONFIG['number'])[0:1000]

thic_train = cast_data(thic_train_set, CONFIG['number'])[0:3000]
thic_test = cast_data(thic_test_set, CONFIG['number'])[0:1000]

swel_train = cast_data(swel_train_set, CONFIG['number'])[0:3000]
swel_test = cast_data(swel_test_set, CONFIG['number'])[0:1000]


### import model
thin_model = VAE1(CONFIG['latent_dim']).to(device)
thic_model = VAE1(CONFIG['latent_dim']).to(device)
raw_model = VAE2(CONFIG['latent_dim']).to(device)
swel_model = VAE3(CONFIG['latent_dim']).to(device)
frac_model = VAE3(CONFIG['latent_dim']).to(device)


def generate_random_noise(generated_LIT, num, m, n):
    mean_embedding = np.mean(generated_LIT)
    std_embedding = np.std(generated_LIT)
    random_noise = np.concatenate([np.random.normal(loc= m*mean_embedding, scale= n*std_embedding, size=(1, np.shape(generated_LIT)[1] )) for i in range(num)], axis = 0)
    return random_noise


def generate_augmented_embeddings(LIT, num, m, n):
    '''
    generate a list of augmented embeddings, where each element is a torch tensor of size (1,latent_dim)
    '''
    expanded_LIT = np.repeat(LIT, num, axis = 0)
    random_noise = generate_random_noise(LIT, num, m, n)
    augmented_embeddings = 0.5*expanded_LIT+0.5*random_noise
    augmented_embeddings = [augmented_embeddings[i].reshape(1, -1) for i in range(augmented_embeddings.shape[0])]
    return augmented_embeddings


def generate_groundtruth_embeddings(model, test_set):
    '''
    generate a list of  embeddings, where each element is a torch tensor of size (1,latent_dim)
    '''
    test_loader = get_loader(test_set, CONFIG['batch_size'])
    model.eval()
    embeddings = []
    for batch_idx, data in enumerate(test_loader):
        data = data.to(device)
        z = model.enc(data)
        z = torch.split(z, 1)
        embeddings.extend(z)
    
    embeddings = [emb.detach().cpu().numpy() for emb in embeddings]
    return embeddings

def reconstruct_images(model, embeddings):

    embeddings = [torch.tensor(emb, dtype=torch.float32).squeeze(0) for emb in embeddings]
    data_loader = get_loader(embeddings, CONFIG['batch_size'])

    images = []

    for batch_idx, data in enumerate(data_loader):
        data = data.to(device)
        model.eval()

        recon = model.dec(data)
        images.extend((torch.split(recon, 1, dim=0)))

    images = [np.squeeze(image.detach().cpu().numpy()) for image in images]

    return images


import os
def clear_dir(dir_name):
    for file in os.listdir(dir_name):
        os.remove(os.path.join(dir_name, file))

def generate_aug():
    # thin_test_loader = get_loader(thin_test, CONFIG['batch_size'])
    # thic_test_loader = get_loader(thic_test, CONFIG['batch_size'])
    # raw_test_loader = get_loader(raw_test, CONFIG['batch_size'])
    # swel_test_loader = get_loader(swel_test, CONFIG['batch_size'])
    # frac_test_loader = get_loader(frac_test, CONFIG['batch_size'])

    num = CLASSIFY_CONFIG['aug_data_size']

    for number in range(10):
        work_path = MODEL_WEIGHT_BACKUP_PATH + "/" + 'number' + str(number) 


        #################### ground truth embeddings##############################
        gt_embs_path = work_path + '/' + 'ablated_nofed' + '/' + 'gt_embs' + '/'
        if not os.path.exists(gt_embs_path):
                os.makedirs(gt_embs_path)


        model_path = MODEL_WEIGHT_BACKUP_PATH + f'/number{number}/ablated_nofed/model/'
        thin_model.load_state_dict(torch.load(model_path + "client1" + ".model"))
        thic_model.load_state_dict(torch.load(model_path + "client2" + ".model"))
        raw_model.load_state_dict(torch.load(model_path + "client3" + ".model"))
        swel_model.load_state_dict(torch.load(model_path + "client4" + ".model"))
        frac_model.load_state_dict(torch.load(model_path + "client5" + ".model"))

        train_embs1 = generate_groundtruth_embeddings(thin_model, thin_train)
        train_embs2 = generate_groundtruth_embeddings(thic_model, thic_train)
        train_embs3 = generate_groundtruth_embeddings(raw_model, raw_train)
        train_embs4 = generate_groundtruth_embeddings(swel_model, swel_train)
        train_embs5 = generate_groundtruth_embeddings(frac_model, frac_train)

        np.save(gt_embs_path+  "client1_train_embs", train_embs1)
        np.save(gt_embs_path+  "client2_train_embs", train_embs2)
        np.save(gt_embs_path+  "client3_train_embs", train_embs3)
        np.save(gt_embs_path+  "client4_train_embs", train_embs4)
        np.save(gt_embs_path+  "client5_train_embs", train_embs5)


        test_embs1 = generate_groundtruth_embeddings(thin_model, thin_test)
        test_embs2 = generate_groundtruth_embeddings(thic_model, thic_test)
        test_embs3 = generate_groundtruth_embeddings(raw_model, raw_test)
        test_embs4 = generate_groundtruth_embeddings(swel_model, swel_test)
        test_embs5 = generate_groundtruth_embeddings(frac_model, frac_test)


        np.save(gt_embs_path+  "client1_test_embs", test_embs1)
        np.save(gt_embs_path+  "client2_test_embs", test_embs2)
        np.save(gt_embs_path+  "client3_test_embs", test_embs3)
        np.save(gt_embs_path+  "client4_test_embs", test_embs4)
        np.save(gt_embs_path+  "client5_test_embs", test_embs5)
        
        ###############################################################################

        for fed_type in ['ablated_nofed', 'ablated_fed']:
            save_path = work_path + "/" + fed_type

            aug_embs_path = save_path + "/" + "aug_embs" + "/"
            if not os.path.exists(aug_embs_path):
                os.makedirs(aug_embs_path)
            aug_imgs_path = save_path + "/" + "aug_imgs" + "/"
            if not os.path.exists(aug_imgs_path):
                os.makedirs(aug_imgs_path)

            
            LIT_path = save_path + '/embedding_templates/'
            LIT1 = np.load(LIT_path + 'client1' + '_embedding_template' + '.npy')
            LIT2 = np.load(LIT_path + 'client2' + '_embedding_template' + '.npy')
            LIT3 = np.load(LIT_path + 'client3' + '_embedding_template' + '.npy')
            LIT4 = np.load(LIT_path + 'client4' + '_embedding_template' + '.npy')
            LIT5 = np.load(LIT_path + 'client5' + '_embedding_template' + '.npy')

            #################### augmented embeddings##############################
            aug_embs1 = generate_augmented_embeddings(LIT1, num, AUG['thin']['m'], AUG['thin']['n'])
            aug_embs2 = generate_augmented_embeddings(LIT2, num, AUG['thic']['m'], AUG['thic']['n'])
            aug_embs3 = generate_augmented_embeddings(LIT3, num, AUG['raw']['m'], AUG['raw']['n'])
            aug_embs4 = generate_augmented_embeddings(LIT4, num, AUG['swel']['m'], AUG['swel']['n'])
            aug_embs5 = generate_augmented_embeddings(LIT5, num, AUG['frac']['m'], AUG['frac']['n'])

            clear_dir(aug_embs_path)
            np.save(aug_embs_path+  "client1_aug_embs", aug_embs1)
            np.save(aug_embs_path+  "client2_aug_embs", aug_embs2)
            np.save(aug_embs_path+  "client3_aug_embs", aug_embs3)
            np.save(aug_embs_path+  "client4_aug_embs", aug_embs4)
            np.save(aug_embs_path+  "client5_aug_embs", aug_embs5)

            #################### augmented images##############################
            model_path = MODEL_WEIGHT_BACKUP_PATH + f'/number{number}/' + fed_type + '/model/'
            thin_model.load_state_dict(torch.load(model_path + "client1" + ".model"))
            thic_model.load_state_dict(torch.load(model_path + "client2" + ".model"))
            raw_model.load_state_dict(torch.load(model_path + "client3" + ".model"))
            swel_model.load_state_dict(torch.load(model_path + "client4" + ".model"))
            frac_model.load_state_dict(torch.load(model_path + "client5" + ".model"))

            aug_imgs1 = reconstruct_images(thin_model, aug_embs1)
            aug_imgs2 = reconstruct_images(thic_model, aug_embs2)
            aug_imgs3 = reconstruct_images(raw_model, aug_embs3)
            aug_imgs4 = reconstruct_images(swel_model, aug_embs4)
            aug_imgs5 = reconstruct_images(frac_model, aug_embs5)

            np.save(aug_imgs_path+  "client1_aug_imgs", aug_imgs1)
            np.save(aug_imgs_path+  "client2_aug_imgs", aug_imgs2)
            np.save(aug_imgs_path+  "client3_aug_imgs", aug_imgs3)
            np.save(aug_imgs_path+  "client4_aug_imgs", aug_imgs4)
            np.save(aug_imgs_path+  "client5_aug_imgs", aug_imgs5)







