import torch
import numpy as np
from scipy.stats import entropy
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.special import kl_div
import pickle
from sklearn import svm, metrics
from util import k_fold_split_indices
from exp_config import CLASSIFY_CONFIG
import random
from aug import generate_aug

np.random.seed(35813)
torch.manual_seed(35813)

### Direct to working environment
MODEL_OUTPUT_PATH = "/home/jia/Desktop/MSc_Project/imagetemplate/FedLIT/output"
data_path = "/home/jia/Desktop/MSc_Project/imagetemplate/FedLIT/classifier_data"




def cast_data_and_label(data0, data1, data2, data3, data4, data5, data6, data7, data8, data9):

    combined_imgs = np.concatenate([data0, data1, data2, data3, data4, data5, data6, data7, data8, data9], axis=0)
    combined_imgs = np.split(combined_imgs, combined_imgs.shape[0])

    # Create a corresponding list of labels for each embedding list
    labels_list0 = [0] * len(data0)
    labels_list1 = [1] * len(data1)
    labels_list2 = [2] * len(data2)
    labels_list3 = [3] * len(data3)
    labels_list4 = [4] * len(data4)
    labels_list5 = [5] * len(data5)
    labels_list6 = [6] * len(data6)
    labels_list7 = [7] * len(data7)
    labels_list8 = [8] * len(data8)
    labels_list9 = [9] * len(data9)

    # Combine the labels lists into a single list
    combined_labels = labels_list0 + labels_list1 + labels_list2 + labels_list3 + labels_list4 + labels_list5 + labels_list6 + labels_list7 + labels_list8 + labels_list9

    assert len(combined_imgs) == len(combined_labels)

    # Shuffle the combined dataset (embeddings and labels) using the same random order
    combined_dataset = list(zip(combined_imgs, combined_labels))
    np.random.shuffle(combined_dataset)

    return combined_dataset

# def cast_data_and_label(data1, data2, data3, data4, data5):

#     combined_imgs = np.concatenate([data1, data2, data3, data4, data5], axis=0)
#     combined_imgs = np.split(combined_imgs, combined_imgs.shape[0])

#     # Create a corresponding list of labels for each embedding list
#     labels_list1 = [5] * len(data1)
#     labels_list2 = [6] * len(data2)
#     labels_list3 = [7] * len(data3)
#     labels_list4 = [8] * len(data4)
#     labels_list5 = [4] * len(data5)

#     # Combine the labels lists into a single list
#     combined_labels = labels_list1 + labels_list2 + labels_list3 + labels_list4 + labels_list5

#     # Shuffle the combined dataset (embeddings and labels) using the same random order
#     combined_dataset = list(zip(combined_imgs, combined_labels))
#     np.random.shuffle(combined_dataset)

#     return combined_dataset

def compute_entropy(predicted_probs):
    '''
    Higher entropy values indicate higher uncertainty, and lower entropy values indicate higher confidence in the predictions
    '''
    entropies = -np.sum(predicted_probs * np.log2(predicted_probs), axis=1)

    return np.mean(entropies)



def load_client_aug_data(client_num, federated):
    
    aug_data = []  # [number0, number1, number2, number3, number4]
    for number in CLASSIFY_CONFIG['numbers']:
        if federated:
            aug_img_path = MODEL_OUTPUT_PATH + f'/number{number}/ablated_fed/aug_imgs/'
        else:
            aug_img_path = MODEL_OUTPUT_PATH + f'/number{number}/ablated_nofed/aug_imgs/'
        
        aug_imgs = np.load(aug_img_path + f'client{client_num}_aug_imgs.npy', allow_pickle=True)
        aug_data.append(aug_imgs)

    combined_dataset = cast_data_and_label(aug_data[0], aug_data[1], aug_data[2], aug_data[3], aug_data[4], aug_data[5], aug_data[6], aug_data[7], aug_data[8], aug_data[9])
    selected_data = random.sample(combined_dataset, CLASSIFY_CONFIG['aug_data_size'])
    return selected_data


# def load_client_aug_data(client_num, federated):
    
#     aug_data = []  # [number0, number1, number2, number3, number4]
#     for number in CLASSIFY_CONFIG['numbers']:
#         if federated:
#             aug_img_path = MODEL_WEIGHT_BACKUP_PATH + f'/number{number}/ablated_fed/aug_imgs/'
#         else:
#             aug_img_path = MODEL_WEIGHT_BACKUP_PATH + f'/number{number}/ablated_nofed/aug_imgs/'
        
#         aug_imgs = np.load(aug_img_path + f'client{client_num}_aug_imgs.npy', allow_pickle=True)
#         aug_data.append(aug_imgs)

#     combined_dataset = cast_data_and_label(aug_data[0], aug_data[1], aug_data[2], aug_data[3], aug_data[4])
#     selected_data = random.sample(combined_dataset, CLASSIFY_CONFIG['aug_data_size'])
#     return selected_data


def linearize_image_data(dataset):
    new_dataset = []
    for image, label in dataset:
        linear_image = image.flatten()
        new_dataset.append((linear_image, label))
    return new_dataset


def compute_aug_mean(acc_list):
    aug_mean = []
    for client in acc_list:
        assert len(client) == 5
        aug_mean.append(sum(client)/len(client))
    return aug_mean

def compute_aug_std(acc_list):
    aug_mean = []
    for client in acc_list:
        assert len(client) == 5
        aug_mean.append(np.std(client))
    return aug_mean



if __name__ == '__main__':
    no_aug = [[], [], [], [], []]
    nonfed_aug = [[], [], [], [], []]
    fed_aug = [[], [], [], [], []]
    for i in range(5):
        generate_aug()

        print('------------------------ Case 1 (no aug) ------------------------------')

        for i in range(1,6):
            print('-----------')
            print(f'Client {i}')
            print('-----------')
            train_data = np.load(data_path + '/' + f'client{i}' + '/train_data.npy', allow_pickle=True)
            train_data = linearize_image_data(train_data)
            x_train, y_train = zip(*train_data)
            test_data = np.load(data_path + '/' + f'client{i}' + '/mixed_test_data.npy', allow_pickle=True)
            test_data = linearize_image_data(test_data)
            x_test, y_test = zip(*test_data)

            # instantiate a model 
            svm_rbf = svm.SVC(kernel='linear')
            # fit
            svm_rbf.fit(x_train, y_train)
            # predict
            predictions = svm_rbf.predict(x_test)

            # measure accuracy
            total_acc = metrics.accuracy_score(y_true=y_test, y_pred=predictions)
            print(f'total acc: {total_acc}')
            # # class-wise accuracy
            # class_wise = metrics.classification_report(y_true=y_test, y_pred=predictions, digits=4)
            # print(class_wise)
            no_aug[i-1].append(total_acc)
        
        print('------------------------ Case 2 (nofed aug) ------------------------------')

        for i in range(1,6):
            print('-----------')
            print(f'Client {i}')
            print('-----------')
            train_data = np.load(data_path + '/' + f'client{i}' + '/train_data.npy', allow_pickle=True)
            train_data = linearize_image_data(train_data)
            x_local_train, y_local_train = zip(*train_data)

            test_data = np.load(data_path + '/' + f'client{i}' + '/mixed_test_data.npy', allow_pickle=True)
            test_data = linearize_image_data(test_data)
            x_test, y_test = zip(*test_data)

            aug_data = load_client_aug_data(i, False)
            aug_data = linearize_image_data(aug_data)
            x_aug, y_aug = zip(*aug_data)

            x_train = x_local_train + x_aug
            y_train = y_local_train + y_aug

            # instantiate a model 
            svm_rbf = svm.SVC(kernel='linear')
            # fit
            svm_rbf.fit(x_train, y_train)
            # predict
            predictions = svm_rbf.predict(x_test)
            
            # measure accuracy
            total_acc = metrics.accuracy_score(y_true=y_test, y_pred=predictions)
            print(f'total acc: {total_acc}')

            nonfed_aug[i-1].append(total_acc)

            # # class-wise accuracy
            # class_wise = metrics.classification_report(y_true=y_test, y_pred=predictions, digits=4)
            # print(class_wise)

        print('------------------------ Case 3 (fed aug) ------------------------------')

        for i in range(1, 6):
            print('-----------')
            print(f'Client {i}')
            print('-----------')
            train_data = np.load(data_path + '/' + f'client{i}' + '/train_data.npy', allow_pickle=True)
            train_data = linearize_image_data(train_data)
            x_local_train, y_local_train = zip(*train_data)

            test_data = np.load(data_path + '/' + f'client{i}' + '/mixed_test_data.npy', allow_pickle=True)
            test_data = linearize_image_data(test_data)
            x_test, y_test = zip(*test_data)

            aug_data = load_client_aug_data(i, True)
            aug_data = linearize_image_data(aug_data)
            x_aug, y_aug = zip(*aug_data)

            x_train = x_local_train + x_aug
            y_train = y_local_train + y_aug


            # instantiate a model 
            svm_rbf = svm.SVC(kernel='linear')
            # fit
            svm_rbf.fit(x_train, y_train)
            # predict
            predictions = svm_rbf.predict(x_test)
            
            # measure accuracy
            total_acc = metrics.accuracy_score(y_true=y_test, y_pred=predictions)
            print(f'total acc: {total_acc}')

            fed_aug[i-1].append(total_acc)

            # # class-wise accuracy
            # class_wise = metrics.classification_report(y_true=y_test, y_pred=predictions, digits=4)
            # print(class_wise)

no_aug_mean = compute_aug_mean(no_aug)
nonfed_aug_mean = compute_aug_mean(nonfed_aug)
fed_aug_mean = compute_aug_mean(fed_aug)

no_aug_std = compute_aug_std(no_aug)
nonfed_aug_std = compute_aug_std(nonfed_aug)
fed_aug_std = compute_aug_std(fed_aug)

aug_result = pd.DataFrame({'noaug_avg': no_aug_mean,
                           'nofed_aug_avg': nonfed_aug_mean,
                           'fed_aug_avg': fed_aug_mean,
                           'noaug_std': no_aug_std,
                           'nofed_aug_std': nonfed_aug_std,
                           'fed_aug_std': fed_aug_std},

                           index= ['client1', 'client2', 'client3', 'client4', 'client5'])
            
print(aug_result)