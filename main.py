import torch
from preprocess_data import simulate_multiresol_dataset
from data import data_util
import os
from helper import cast_data
from morphomnist.util import plot_grid, plot_digit
import torch.nn.functional as F
from config import CONFIG
from demo import demo

'''
Setting up working environment
'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'{device} is available....')


'''
Setting up working directory
'''
content_path = "/home/jia/Desktop/MSc_Project/imagetemplate/FedLIT"
origin_data_path = "/home/jia/Desktop/MSc_Project/imagetemplate/FedLIT/origin_data"
data_path = "/home/jia/Desktop/MSc_Project/imagetemplate/FedLIT/data"


'''
Import and preprocess data
'''
#To simulate a simtuation that the local data is not only heterogeneous but also at different resolution across clients, we upsample thin MNIST abd thick MNIST at size 56x56; downsample swelling MNIST and fractures MNIST to size 14x14; retain raw MNIST at 28x28.

# if not os.path.exists(data_path):
#     simulate_multiresol_dataset(origin_data_path, data_path)


raw_data_dirs = [os.path.join(data_path, 'raw')]
frac_data_dirs = [os.path.join(data_path, 'frac')]
thin_data_dirs = [os.path.join(data_path, 'thin')]
thic_data_dirs = [os.path.join(data_path, 'thic')]
swel_data_dirs = [os.path.join(data_path, 'swel')]

# Raw MNIST
raw_train_set = data_util.get_dataset(raw_data_dirs, train=True)
raw_test_set = data_util.get_dataset(raw_data_dirs, train=False)

# Frac MNIST
frac_train_set = data_util.get_dataset(frac_data_dirs, train=True)
frac_test_set = data_util.get_dataset(frac_data_dirs, train=False)

# Thin MNIST
thin_train_set = data_util.get_dataset(thin_data_dirs, train=True)
thin_test_set = data_util.get_dataset(thin_data_dirs, train=False)

# Thic MNIST
thic_train_set = data_util.get_dataset(thic_data_dirs, train=True)
thic_test_set = data_util.get_dataset(thic_data_dirs, train=False)

# Swel MNIST
swel_train_set = data_util.get_dataset(swel_data_dirs, train=True)
swel_test_set = data_util.get_dataset(swel_data_dirs, train=False)


#At this stage, label is not required for the training. Therefore we get rid of labels in the dataset. 
# The digit class can be edit in config.py

# raw_train_set = cast_data(raw_train_set, CONFIG['number'])[0:3000]
# raw_test_set = cast_data(raw_test_set, CONFIG['number'])[0:1000]

# frac_train_set = cast_data(frac_train_set, CONFIG['number'])[0:3000]
# frac_test_set = cast_data(frac_test_set, CONFIG['number'])[0:1000]

# thin_train_set = cast_data(thin_train_set, CONFIG['number'])[0:3000]
# thin_test_set = cast_data(thin_test_set, CONFIG['number'])[0:1000]

# thic_train_set = cast_data(thic_train_set, CONFIG['number'])[0:3000]
# thic_test_set = cast_data(thic_test_set, CONFIG['number'])[0:1000]

# swel_train_set = cast_data(swel_train_set, CONFIG['number'])[0:3000]
# swel_test_set = cast_data(swel_test_set, CONFIG['number'])[0:1000]


raw_train_set = cast_data(raw_train_set, CONFIG['number'])[0:1000]
raw_test_set = cast_data(raw_test_set, CONFIG['number'])[0:500]

frac_train_set = cast_data(frac_train_set, CONFIG['number'])[0:1000]
frac_test_set = cast_data(frac_test_set, CONFIG['number'])[0:500]

thin_train_set = cast_data(thin_train_set, CONFIG['number'])[0:1000]
thin_test_set = cast_data(thin_test_set, CONFIG['number'])[0:500]

thic_train_set = cast_data(thic_train_set, CONFIG['number'])[0:1000]
thic_test_set = cast_data(thic_test_set, CONFIG['number'])[0:500]

swel_train_set = cast_data(swel_train_set, CONFIG['number'])[0:1000]
swel_test_set = cast_data(swel_test_set, CONFIG['number'])[0:500]

'''
Generate LIT
'''
recon_results, local_center_results, global_center_results = demo(thin_train_set, thin_test_set, thic_train_set, thic_test_set, raw_train_set, raw_test_set, swel_train_set, swel_test_set, frac_train_set, frac_test_set)


'''
Show evaluation result
'''
print("--------Reconstruction---------")
print(recon_results)
print("--------Local centeredness---------")
print(local_center_results)
print("--------Global centeredness---------")
print(global_center_results)