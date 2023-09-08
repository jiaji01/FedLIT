from sklearn import svm, metrics
import torch
import numpy as np
from exp_config import ONESHOT_CONFIG
import random


np.random.seed(35813)
torch.manual_seed(35813)

### Direct to working environment
MODEL_OUTPUT_PATH = "/home/jia/Desktop/MSc_Project/imagetemplate/FedLIT/output"
data_path = "/home/jia/Desktop/MSc_Project/imagetemplate/FedLIT/classifier_data"




def cast_train_data(train_data, sample_size, numbers):
    dataset = {}
    for number in numbers:
        data = []
        for x, y in train_data:
            if y == number:
                data.append(x)
        dataset[number] = data
    n_subjects = min([len(data) for _, data in dataset.items()])
    sampled_subjects_id = [random.randint(1, n_subjects) for _ in numbers]

    sampled_dataset = []
    for i in range(sample_size):
        new_dataset = []
        for number, data in dataset.items():
            new_dataset.append((data[sampled_subjects_id[number-1]], number))
        sampled_dataset.append(new_dataset)   
    
    assert len(sampled_dataset) == sample_size

    return sampled_dataset


def linearize_image_data(dataset):
    new_dataset = []
    for image, label in dataset:
        linear_image = image.flatten()
        new_dataset.append((linear_image, label))
    return new_dataset



if __name__ == '__main__':

    for i in range(1,6):
        print('-----------')
        print(f'Client {i}')
        print('-----------')


        # ------------------------ train data ------------------------------
        train_data = np.load(data_path + '/' + f'client{i}' + '/train_data.npy', allow_pickle=True)
        train_data = linearize_image_data(train_data)
        test_data = np.load(data_path + '/' + f'client{i}' + '/mixed_test_data.npy', allow_pickle=True)
        test_data = linearize_image_data(test_data)
        x_test, y_test = zip(*test_data)

        sampled_oneshot_train_data = cast_train_data(train_data, ONESHOT_CONFIG['train_times'], ONESHOT_CONFIG['numbers'])

        one_shot_accs = []
        for one_shot_train_data in sampled_oneshot_train_data:

            x_train, y_train = zip(*one_shot_train_data)

            # instantiate a model 
            svm_linear = svm.SVC(kernel='linear')
            # fit
            svm_linear.fit(x_train, y_train)
            # predict
            predictions = svm_linear.predict(x_test)

            acc = metrics.accuracy_score(y_true=y_test, y_pred=predictions)
            one_shot_accs.append(acc)
        
        avg_one_shot_acc = sum(one_shot_accs)/len(one_shot_accs)
        std_one_shot_acc = np.std(one_shot_accs)
        print(f'average one-shot accuarcy: {avg_one_shot_acc}, std: {std_one_shot_acc}' )
        # ------------------------ LIT ------------------------------
        LIT_set = []
        for number in ONESHOT_CONFIG['numbers']:
            LIT = np.load( MODEL_OUTPUT_PATH + '/' + f'number{number}' + '/' + 'ablated_fed/image_templates/' f'client{i}' + '_image_template.npy')
            LIT_set.append((LIT, number))
        LIT_set = linearize_image_data(LIT_set)
        x_LIT, y_LIT = zip(*LIT_set)
        # instantiate a model 
        svm_linear = svm.SVC(kernel='linear')
        # fit
        svm_linear.fit(x_LIT, y_LIT)
        # predict
        predictions = svm_linear.predict(x_test)

        LIT_acc = metrics.accuracy_score(y_true=y_test, y_pred=predictions)

        
        print(f'LIT accuarcy: {LIT_acc}' )
     



