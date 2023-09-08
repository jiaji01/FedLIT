import numpy as np
from config import CONFIG
import os
import matplotlib.pyplot as plt
import pickle

def plot_digit(img, ax=None, title=None, **kwargs):
    """Plots a single MNIST digit.

    Parameters
    ----------
    img : (H, W) array_like
        2D array containing the digit image.
    ax : matplotlib.axes.Axes, optional
        Axes onto which to plot. Defaults to current axes.
    title : str, optional
        If given, sets the plot's title.
    **kwargs
        Keyword arguments passed to `plt.imshow(...)`.
    """
    if ax is None:
        ax = plt.gca()
    def_kwargs = dict(cmap='gray_r')
    def_kwargs.update(**kwargs)
    ax.imshow(np.asarray(img).squeeze(), **def_kwargs)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    if title is not None:
        ax.set_title(title)


def plot_grid(imgs, nrow=None, digit_kw=None, **kwargs):
    """Plots a grid of MNIST digits.

    Parameters
    ----------
    imgs : (N, H, W) array_like
        3D array containing the digit images, indexed along the first axis.
    nrow : int, optional
        Number of rows. If `None`, will attempt to make a square grid.
    digit_kw : dict, optional
        Dictionary of keyword arguments to `plot_digit(...)`.
    **kwargs
        Keyword arguments to `plt.subplots(...)`.

    Returns
    -------
    Tuple[matplotlib.figure.Figure, numpy.ndarray[matplotlib.axes.Axes]]
        The created figure and subplot axes.
    """
    imgs = np.asarray(imgs)
    num = imgs.shape[0]
    if nrow is None:
        nrow = int(np.floor(np.sqrt(num)))
    ncol = int(np.ceil(num / nrow))
    fig, axs = plt.subplots(nrow, ncol, **kwargs)
    axs = np.atleast_1d(axs).T
    if digit_kw is None:
        digit_kw = {}
    for i in range(num):
        ax = axs.flat[i]
        plot_digit(imgs[i], ax, **digit_kw)
        ax.axis('off')
    for ax in axs.flat[num:]:
        ax.set_visible(False)
    return fig, axs



def plot_cbt(model_name):
    save_path = f'./output/{model_name}/image_template/ablated_fed'
    image_path = f'./output/{model_name}/ablated_fed/image_templates'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for n in range(5):
        print("*********client {} *********".format(n+1))
        cbt = np.load(f'{image_path}/client{n+1}_image_template.npy')
        cbt = np.squeeze(cbt)
        print(cbt.shape)
        plot_digit(cbt)
        plt.savefig(f'{save_path}/client{n+1}')
        plt.show()


def plot_augmented_images():
    for number in range(10):
        for client in range(1, 6):
            aug = np.load(f'./output/number{number}/ablated_fed/aug_imgs/client{client}_aug_imgs.npy')
            plot_grid(aug[0:16])
            plt.show()


def plot_client_training_log(client_name, loss_data_nofed, loss_data_fed,save_path):
    print("********* {} *********".format(client_name))

    fig, axs = plt.subplots(1, 3, figsize = (20,4))
    for i, key in enumerate(["SNL", "contrastive", "reconstruction"]):
        loss_fed = loss_data_fed[key]
        loss_nofed = loss_data_nofed[key]
        epoch = loss_data_nofed["epoch"]
        axs[i].plot(epoch, loss_fed, 'tab:orange', label='ablated fed')  # Add label for federated loss
        axs[i].plot(epoch, loss_nofed, 'tab:green', label='ablated nofed')  # Add label for non-federated loss
        axs[i].set(xlabel= 'epoch', ylabel= f'{key} loss')
        axs[i].set_title(f'{key} loss')
        axs[i].legend()  # Show the legend
    plt.suptitle(f'{client_name} with random sampling on train set')
    plt.savefig(f'{save_path}/{client_name}_trainloss')
    plt.show()


def plot_training_log(model_name):
    save_path = f'./output/{model_name}/client_loss_plot/nofed'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for client in ['client1', 'client2', 'client3', 'client4', 'client5']:
    
        with open(f'./output/{model_name}/ablated_nofed/loss/client_loss_log.pkl', 'rb') as file:
            log_nofed = pickle.load(file)
        with open(f'./output/{model_name}/ablated_fed/loss/client_loss_log.pkl', 'rb') as file:
            log_fed = pickle.load(file)
        plot_client_training_log(client, log_nofed[client], log_fed[client], save_path)   


def plot_client_eval_log(client_name, eval_data_nofed, eval_data_fed, save_path):
    print("********* {} *********".format(client_name))

    fig, axs = plt.subplots(1, 2, figsize = (16,4))
    for i, key in enumerate(["local_centeredness", 'reconstruction']):
        
        loss_nofed = eval_data_nofed[key]
        loss_fed = eval_data_fed[key]
        epoch = eval_data_fed["epoch"]
        axs[i].plot(epoch, loss_nofed, 'tab:green', label='ablated nofed')  # Add label for non-federated loss
        axs[i].plot(epoch, loss_fed, 'tab:orange', label='ablated fed')  # Add label for federated loss
        axs[i].set(xlabel= 'epoch', ylabel= f'{key} loss')
        axs[i].set_title(f'{key} loss')
        axs[i].legend()  # Show the legend
    plt.suptitle(f'{client_name} with random sampling on train set')
    plt.savefig(f'{save_path}/{client_name}_eval')
    plt.show()


def plot_eval_log(model_name):
    save_path = f'./output/{model_name}/client_eval_plot/nofed'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for client in ['client1', 'client2', 'client3', 'client4', 'client5']:
    
        with open(f'./output/{model_name}/ablated_nofed/eval/client_eval_log.pkl', 'rb') as file:
            log_nofed = pickle.load(file)
        with open(f'./output/{model_name}/ablated_fed/eval/client_eval_log.pkl', 'rb') as file:
            log_fed = pickle.load(file)
        plot_client_eval_log(client, log_nofed[client], log_fed[client], save_path)


def plot_global_centeredness(eval_data_nofed, eval_data_fed, save_path):
  

    fig, axs = plt.subplots(1, 1, figsize = (8,4))
    for i, key in enumerate(["global_centeredness"]):
        
        loss_nofed = eval_data_nofed[key]
        loss_fed = eval_data_fed[key]
        epoch = eval_data_fed["epoch"]
        axs.plot(epoch, loss_nofed, 'tab:green', label='ablated nofed')  # Add label for non-federated loss
        axs.plot(epoch, loss_fed, 'tab:orange', label='ablated fed')  # Add label for federated loss
        axs.set(xlabel= 'epoch', ylabel= f'{key} loss')
        axs.set_title(f'{key} loss')
        axs.legend()  # Show the legend
    plt.suptitle(f'Server evaluation')
    plt.savefig(f'{save_path}/server_eval')
    plt.show()


def compare_global_centeredness(model_name):
    save_path = f'./output/{model_name}/server_eval_plot/nofed'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(f'./output/{model_name}/ablated_nofed/eval/server_eval_log.pkl', 'rb') as file:
        log_nofed = pickle.load(file)
    with open(f'./output/{model_name}/ablated_fed/eval/server_eval_log.pkl', 'rb') as file:
        log_fed = pickle.load(file)
    plot_global_centeredness( log_nofed, log_fed, save_path)



if __name__ == "__main__":
    
    model_name = CONFIG['model_name']

    plot_cbt(model_name)
    plot_training_log(model_name)
    plot_eval_log(model_name)
    compare_global_centeredness(model_name)

    plot_augmented_images()