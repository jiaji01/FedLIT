
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.special import kl_div
import os
from util import mix_dataset



np.random.seed(35813)
torch.manual_seed(35813)

### Direct to working environment
WORK_PATH = "/home/jia/Desktop/MSc_Project/imagetemplate/FedLIT"
MODEL_OUTPUT_PATH = "/home/jia/Desktop/MSc_Project/imagetemplate/FedLIT/output"


# def plot_latent_distribution(local_ground_truth, mixed_ground_truth, predicted, save_path):
    
#     # Flatten the datasets to 1D arrays
#     flattened_dataset1 = np.array(local_ground_truth).reshape(-1,1)
#     flattened_dataset2 = np.array(mixed_ground_truth).reshape(-1,1)
#     flattened_dataset3 = np.array([img.flatten() for img in predicted]).reshape(-1,1)
#     # Calculate the KDE for each dataset
#     kde_dataset1 = gaussian_kde(flattened_dataset1)
#     kde_dataset2 = gaussian_kde(flattened_dataset2)
#     kde_dataset3 = gaussian_kde(flattened_dataset3)

#     # Define the x-axis range for the plots
#     x_range = np.linspace(min(flattened_dataset1.min(), flattened_dataset2.min(), flattened_dataset3.min()), 
#                           max(flattened_dataset1.max(), flattened_dataset2.max(), flattened_dataset3.max()), 50)

#     # Plot the KDE curves as line graphs
#     plt.plot(x_range, kde_dataset1(x_range), label='local test set')
#     plt.plot(x_range, kde_dataset2(x_range), label='mixed test set')
#     plt.plot(x_range, kde_dataset3(x_range), label='augmented')

#     # plt.hist(flattened_dataset1, bins=50, alpha=0.7, label='ground_truth')
#     # plt.hist(flattened_dataset2, bins=50, alpha=0.7, label='predicted')

#     plt.xlabel('Value')
#     plt.ylabel('Density')
#     plt.title('Distributions of Latent Embeddings')
#     plt.legend()
#     plt.savefig(save_path)
#     plt.show()

# def linearize_image_data(dataset):
#     new_dataset = []
#     for image, label in dataset:
#         linear_image = image.flatten()
#         new_dataset.append(linear_image)
#     return new_dataset

def plot_latent_distribution(local_ground_truth, mixed_ground_truth, predicted, save_path):
    
    # Flatten the datasets to 1D arrays
    flattened_dataset1 = np.concatenate(local_ground_truth).flatten()
    flattened_dataset2 = np.concatenate(mixed_ground_truth).flatten()
    flattened_dataset3 = np.concatenate(predicted).flatten()

    # Calculate the KDE for each dataset
    kde_dataset1 = gaussian_kde(flattened_dataset1)
    kde_dataset2 = gaussian_kde(flattened_dataset2)
    kde_dataset3 = gaussian_kde(flattened_dataset3)

    # Define the x-axis range for the plots
    x_range = np.linspace(min(flattened_dataset1.min(), flattened_dataset2.min(), flattened_dataset3.min()), 
                          max(flattened_dataset1.max(), flattened_dataset2.max(), flattened_dataset3.max()), 50)

    # Plot the KDE curves as line graphs
    plt.plot(x_range, kde_dataset1(x_range), label='local test set')
    plt.plot(x_range, kde_dataset2(x_range), label='mixed test set')
    plt.plot(x_range, kde_dataset3(x_range), label='augmented')

    # plt.hist(flattened_dataset1, bins=50, alpha=0.7, label='ground_truth')
    # plt.hist(flattened_dataset2, bins=50, alpha=0.7, label='predicted')

    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Distributions of Latent Embeddings')
    plt.legend()
    plt.savefig(save_path)
    plt.show()


def mix_dataset(datset_list):
    mixed_dataset = []

    for datset in datset_list:
        mixed_dataset.extend(datset)
    
    np.random.shuffle(mixed_dataset)

    return mixed_dataset


# if __name__ == '__main__':

#     for number in range(5):
#         output_path = MODEL_WEIGHT_BACKUP_PATH + "/" + 'number' + str(number)
#         data_path = WORK_PATH + "/" + 'classifier_data/'
#         gt_imgs1 = np.load(data_path + 'client1/mixed_test_data.npy', allow_pickle=True)
#         gt_imgs1 = linearize_image_data(gt_imgs1)
#         gt_imgs2 = np.load(data_path + 'client2/mixed_test_data.npy', allow_pickle=True)
#         gt_imgs2 = linearize_image_data(gt_imgs2)
#         gt_imgs3 = np.load(data_path + 'client3/mixed_test_data.npy', allow_pickle=True)
#         gt_imgs3 = linearize_image_data(gt_imgs3)
#         gt_imgs4 = np.load(data_path + 'client4/mixed_test_data.npy', allow_pickle=True)
#         gt_imgs4 = linearize_image_data(gt_imgs4)
#         gt_imgs5 = np.load(data_path + 'client5/mixed_test_data.npy', allow_pickle=True)
#         gt_imgs5 = linearize_image_data(gt_imgs5)

#         mixed_gt_imgs = mix_dataset([gt_imgs1, gt_imgs2, gt_imgs3, gt_imgs4, gt_imgs5])

#         for type in ['ablated_nofed', 'ablated_fed']:
#             save_path = output_path + "/" + type
#             dist_path = save_path + "/" + "dist_imgs" + "/"
#             if not os.path.exists(dist_path):
#                 os.makedirs(dist_path)
      
#             aug_img_path = save_path + '/' + 'aug_imgs/'
#             aug_imgs1 = np.load(aug_img_path + 'client1_aug_imgs.npy')
#             aug_imgs2 = np.load(aug_img_path + 'client2_aug_imgs.npy')
#             aug_imgs3 = np.load(aug_img_path + 'client3_aug_imgs.npy')
#             aug_imgs4 = np.load(aug_img_path + 'client4_aug_imgs.npy')
#             aug_imgs5 = np.load(aug_img_path + 'client5_aug_imgs.npy')

#             plot_latent_distribution(gt_imgs1, mixed_gt_imgs, aug_imgs1, dist_path+'client1_latent_dist')
#             plot_latent_distribution(gt_imgs2, mixed_gt_imgs, aug_imgs2, dist_path+'client2_latent_dist')
#             plot_latent_distribution(gt_imgs3, mixed_gt_imgs, aug_imgs3, dist_path+'client3_latent_dist')
#             plot_latent_distribution(gt_imgs4, mixed_gt_imgs, aug_imgs4, dist_path+'client4_latent_dist')
#             plot_latent_distribution(gt_imgs5, mixed_gt_imgs, aug_imgs5, dist_path+'client5_latent_dist')


if __name__ == '__main__':

    for number in range(5):
        work_path = MODEL_OUTPUT_PATH + "/" + 'number' + str(number)

        gt_emb_path = MODEL_OUTPUT_PATH + f'/number{number}/ablated_nofed/gt_embs/'
        gt_embs1 = np.load(gt_emb_path + 'client1_test_embs.npy')
        gt_embs2 = np.load(gt_emb_path + 'client2_test_embs.npy')
        gt_embs3 = np.load(gt_emb_path + 'client3_test_embs.npy')
        gt_embs4 = np.load(gt_emb_path + 'client4_test_embs.npy')
        gt_embs5 = np.load(gt_emb_path + 'client5_test_embs.npy')

        mixed_gt_embs = mix_dataset([gt_embs1, gt_embs2, gt_embs3, gt_embs4, gt_embs5])

        for type in ['ablated_nofed', 'ablated_fed']:
            save_path = work_path + "/" + type
            dist_path = save_path + "/" + "dist_embs" + "/"
            if not os.path.exists(dist_path):
                os.makedirs(dist_path)
      
            aug_emb_path = save_path + '/' + 'aug_embs/'
            aug_embs1 = np.load(aug_emb_path + 'client1_aug_embs.npy')
            aug_embs2 = np.load(aug_emb_path + 'client2_aug_embs.npy')
            aug_embs3 = np.load(aug_emb_path + 'client3_aug_embs.npy')
            aug_embs4 = np.load(aug_emb_path + 'client4_aug_embs.npy')
            aug_embs5 = np.load(aug_emb_path + 'client5_aug_embs.npy')

            plot_latent_distribution(gt_embs1, mixed_gt_embs, aug_embs1, dist_path+'client1_latent_dist')
            plot_latent_distribution(gt_embs2, mixed_gt_embs, aug_embs2, dist_path+'client2_latent_dist')
            plot_latent_distribution(gt_embs3, mixed_gt_embs, aug_embs3, dist_path+'client3_latent_dist')
            plot_latent_distribution(gt_embs4, mixed_gt_embs, aug_embs4, dist_path+'client4_latent_dist')
            plot_latent_distribution(gt_embs5, mixed_gt_embs, aug_embs5, dist_path+'client5_latent_dist')






