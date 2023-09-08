import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import torch
import random
import os
import torch.utils.data as data_utils
import torch
from sklearn.decomposition import PCA
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Clears the given directory
def clear_dir(dir_name):
    for file in os.listdir(dir_name):
        os.remove(os.path.join(dir_name, file))


def cast_data(data, number):
    new_data=[]
    for image, label in data:
        if label == torch.tensor(number, dtype=torch.uint8):
            new_data.append(image.unsqueeze(0).float())
    return new_data

def get_loader(feature_vectors, batch_size, num_workers=1):
    """
    Build and return a data loader.
    """
    loader = data_utils.DataLoader(feature_vectors,
                        batch_size=batch_size,
                        shuffle = True, #set to True in case of training and False when testing the model
                        num_workers=num_workers
                        )
    
    return loader

def random_sampling(feature_matrix, sample_size):

    n_subjects =  np.shape(feature_matrix)[0]
    subject_ids = np.arange(1, n_subjects)
    sampled_subjects_id = np.random.choice(subject_ids, size=sample_size, replace=False)

    return sampled_subjects_id.tolist()

def batchnorm_keys(model):
    notshared_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            notshared_layers.append(name)

    notshared_keys = []
    for key in model.state_dict().keys():
        for layer in notshared_layers:
            if layer in key.lower():
                notshared_keys.append(key)
    
    return set(notshared_keys)

def frobenious_distance(input_tensor, output_tensor):
    """Calculate the mean Frobenius distance between two tensors.
        used in the calculation of centerdness loss and reconstruction loss
    """
    assert input_tensor.shape == output_tensor.shape, "Input and output tensors must have the same shape"
    
    # Compute element-wise difference
    diff = input_tensor - output_tensor
    
    # Square the differences
    squared_diff = torch.pow(diff, 2)
    
    # Sum the squared differences along all dimensions
    sum_squared_diff = torch.sum(squared_diff)
    
    # Take the square root of the summed squared differences
    frobenius_distance = torch.sqrt(sum_squared_diff)

    return frobenius_distance


def generate_template_median(model, data_loader):
    model.eval()
    embeddings = []
    for batch_idx, data in enumerate(data_loader):
        data = data.to(device)
        template = model.enc(data)
        embeddings.append(template)
    embeddings = torch.vstack(embeddings)
    final_template = torch.median(embeddings, dim=0)[0]
    return final_template.unsqueeze(0), embeddings.detach()


def generate_image_template(model, generated_template):
    image = model.dec(generated_template)
    return image.detach().cpu().numpy()



def generate_embedding(model, data_loader):
    model.eval()
    embeddings = []
    for batch_idx, data in enumerate(data_loader):
        data = data.to(device)
        z = model.enc(data)
        z = torch.split(z, 1)
        embeddings.extend(z)
    return embeddings



def evaluate_reconstruction(model, data_loader):
    scores = []

    for batch_idx, data in enumerate(data_loader):
            data = data.to(device)

            _,recon = model(data)
            scores.append(torch.mean(torch.abs(data - recon)))
            # scores.append(F.mse_loss(recon, data, reduction="sum"))
    
    recon = (sum(scores)/len(scores)).detach().cpu().clone().numpy()  
    return recon

def evaluate_local_centeredness(model, data_loader, generated_template):
    scores = []

    for batch_idx, data in enumerate(data_loader):
        data = data.to(device)
        z = model.enc(data)

        for i in range(z.shape[0]):
            vec = z[i:(i+1), :]
            scores.append(frobenious_distance(vec, generated_template))

    return (sum(scores)/len(scores)).detach().cpu().clone().numpy()

def evaluate_global_centeredness(template_list):

    total_l2_loss = 0
    for i in range(len(template_list) - 1):
        for j in range(i + 1, len(template_list)):
            total_l2_loss += torch.norm(template_list[i] - template_list[j])
    return total_l2_loss.item()


def plot_TSNE(embeddings_list, template_list, perplexity,type, save_path):
    
    all_embeddings = np.concatenate([z.cpu().detach().numpy() for embeddings in embeddings_list for z in embeddings], axis=0)
    all_templates = np.concatenate([template.cpu().detach().numpy() for template in template_list], axis=0)
    plot_data = np.vstack((all_embeddings, all_templates))
    # Define colors and labels for the embeddings
    colors = ['royalblue',  'tomato', 'limegreen', 'gold', 'mediumorchid', 'navy', 'darkred', 'darkgreen', 'darkorange',  'indigo']
    labels = ['Thin embeddings', 'Thick embeddings', 'raw embeddigs', 'swel embeddings', 'fractures embeddings', 
              'Thin LIT', 'Thick LIT', 'raw LIT', 'swel LIT', 'fractures LIT']
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0)

    # Transform the embeddings to 2D for visualization
    all_embeddings_2d = tsne.fit_transform(plot_data)

    # Plot the transformed embeddings with colors
    plt.figure(figsize=(8, 8))

    # Iterate over embeddings and plot them with corresponding color and label
    start = 0
    for idx, length in enumerate([len(embeddings_list[0]), len(embeddings_list[1]), len(embeddings_list[2]),len(embeddings_list[3]), len(embeddings_list[4]), 1, 1, 1, 1, 1]):
        end = start + length
        if idx < 5:
            plt.scatter(all_embeddings_2d[start:end, 0], all_embeddings_2d[start:end, 1], c=colors[idx], label=labels[idx], alpha= 0.2)
        else:
            plt.scatter(all_embeddings_2d[start:end, 0], all_embeddings_2d[start:end, 1], c=colors[idx], label=labels[idx])
        start = end

    plt.title('t-SNE visualization of all embeddings and client-based LIT')
    plt.legend()  # Add a legend
    plt.savefig(f'{save_path}/{type}_tsne')
    plt.show()


def plot_PCA(embeddings_list, template_list, type, save_path ):
    all_embeddings = np.concatenate([z.cpu().detach().numpy() for embeddings in embeddings_list for z in embeddings], axis=0)
    all_templates = np.concatenate([template.cpu().detach().numpy() for template in template_list], axis=0)
    plot_data = np.vstack((all_embeddings, all_templates))
    # Define colors and labels for the embeddings
    colors = ['royalblue',  'tomato', 'limegreen', 'gold', 'mediumorchid', 'navy', 'darkred', 'darkgreen', 'darkorange',  'indigo']
    labels = ['Thin embeddings', 'Thick embeddings', 'raw embeddigs', 'swel embeddings', 'fractures embeddings', 
              'Thin LIT', 'Thick LIT', 'raw LIT', 'swel LIT', 'fractures LIT']

    pca = PCA(n_components=2)

    # Transform the embeddings to 2D for visualization
    all_embeddings_2d = pca.fit_transform(plot_data)

    # Plot the transformed embeddings with colors
    plt.figure(figsize=(8, 8))
    start = 0
    for idx, length in enumerate([len(embeddings_list[0]), len(embeddings_list[1]), len(embeddings_list[2]),len(embeddings_list[3]), len(embeddings_list[4]), 1, 1, 1, 1, 1]):
        end = start + length
        if idx < 5:
            scatter = plt.scatter(all_embeddings_2d[start:end, 0], all_embeddings_2d[start:end, 1], c=colors[idx], label=labels[idx], alpha= 0.2)
        else:
            scatter = plt.scatter(all_embeddings_2d[start:end, 0], all_embeddings_2d[start:end, 1], c=colors[idx], label=labels[idx])
        start = end
    plt.title('PCA visualization of all embeddings')

    # Create a legend for the scatter plot
    handles, _ = scatter.legend_elements()
    plt.legend(handles, labels, title="Categories")
    plt.savefig(f'{save_path}/{type}_pca')
    plt.show()

# def show_image(img, n, save_path):
#     """The input should be numpy.array
#         i is fold number
#     """
#     # img = np.repeat(np.repeat(img, 10, axis=1), 10, axis=0)
#     plt.imshow(img)
#     plt.colorbar()
#     plt.title(f'Client {n} image template')
#     plt.axis('on')
#     plt.savefig(f'{save_path}/client{n}_ImageTemplate', dpi=300)
#     plt.show()


# def plot_TSNE(embeddings_list, cbt_list, perplexity, i, type, save_path):
#     vectorized_embeddings = [batch_vectorize(embedding) for embedding in embeddings_list]
#     all_embeddings = np.concatenate(vectorized_embeddings , axis=0)
#     vectorized_cbts = [vectorize(cbt).cpu().numpy() for cbt in cbt_list]
#     all_cbts = np.concatenate(vectorized_cbts, axis=0)
#     plot_data = np.concatenate((all_embeddings, all_cbts), axis = 0)
#     # Define colors and labels for the embeddings
#     colors = ['lightcoral', 'lightsteelblue', 'lime', 'red', 'blue', 'green']
#     labels = ['Embeddings of hospital 1', 'Embeddings of hospital 2', 'Embeddings of hospital 3', 'Hospital1-specific CBT', 'Hospital2-specific CBT', 'Hospital3-specific CBT']
    
#     tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0)

#     # Transform the embeddings to 2D for visualization
#     all_embeddings_2d = tsne.fit_transform(plot_data)

#     # Plot the transformed embeddings with colors
#     plt.figure(figsize=(8, 8))

#     # Iterate over embeddings and plot them with corresponding color and label
#     start = 0
#     for idx, length in enumerate([len(embeddings_list[0]), len(embeddings_list[1]), len(embeddings_list[2]), 1, 1, 1]):
#         end = start + length
#         plt.scatter(all_embeddings_2d[start:end, 0], all_embeddings_2d[start:end, 1], c=colors[idx], label=labels[idx])
#         start = end

#     plt.title('t-SNE visualization of all embeddings and client-based CBT')
#     plt.legend()  # Add a legend
#     plt.savefig(f'{save_path}/fold{i}_{type}_tsne')
#     plt.show()


# def plot_PCA(embeddings_list, cbt_list):
#     vectorized_embeddings = [batch_vectorize(embedding) for embedding in embeddings_list]
#     all_embeddings = np.concatenate(vectorized_embeddings , axis=0)
#     vectorized_cbts = [vectorize(cbt).cpu().numpy() for cbt in cbt_list]
#     all_cbts = np.concatenate(vectorized_cbts, axis=0)
#     plot_data = np.concatenate((all_embeddings, all_cbts), axis = 0)

#     # Create color labels for the embeddings
#     colors = ['lightcoral'] * len(embeddings_list[0]) + ['lightsteelblue'] * len(embeddings_list[1]) + ['lime'] * len(embeddings_list[2]) + ["red"] + ["blue"] +["green"]
#     labels = ['Embeddings of hospital 1', 'Embeddings of hospital 2', 'Embeddings of hospital 3', 'Hospital1-specific CBT', 'Hospital2-specific CBT', 'Hospital3-specific CBT']  # Label each category

#     pca = PCA(n_components=2)

#     # Transform the embeddings to 2D for visualization
#     all_embeddings_2d = pca.fit_transform(plot_data)

#     # Plot the transformed embeddings with colors
#     plt.figure(figsize=(8, 8))
#     scatter = plt.scatter(all_embeddings_2d[:, 0], all_embeddings_2d[:, 1], c=colors)
#     plt.title('PCA visualization of all embeddings')

#     # Create a legend for the scatter plot
#     handles, _ = scatter.legend_elements()
#     plt.legend(handles, labels, title="Categories")

#     # plt.savefig(f'{save_path}/client{n}_fold{i}_cbt')
#     plt.show()