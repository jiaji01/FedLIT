import torch
import torch.nn.functional as F


def calculate_SNL_loss(input_tensor, output_tensor):
    """Calculate the mean Frobenius distance between two tensors.
    """
    assert input_tensor.shape == output_tensor.shape, "Input and output tensors must have the same shape"
    
    # Compute element-wise difference
    diff = input_tensor - output_tensor  # # i*n*latent_dim
    
    # # Square the differences
    # squared_diff = torch.pow(diff, 2)  # i*n*latent_dim
    
    # # Sum the squared differences along all dimensions
    # sum_squared_diff = torch.sum(squared_diff, dim=(1, 2))  # i
    
    # # Take the square root of the summed squared differences
    # frobenius_distance = torch.sqrt(sum_squared_diff)  # i

    return torch.sum(torch.abs(input_tensor - output_tensor))

# def calculate_reconstruction_loss(input_tensor, output_tensor):
#     """Calculate the mean Frobenius distance between two tensors.
#     """
#     assert input_tensor.shape == output_tensor.shape, "Input and output tensors must have the same shape"
    
#     # Compute element-wise difference
#     diff = input_tensor - output_tensor  # i*1*28*28
    
#     # Square the differences
#     squared_diff = torch.pow(diff, 2)  # i*1*28*28

#     # Sum the squared differences along all dimensions
#     sum_squared_diff = torch.sum(squared_diff, dim = (1, 2, 3))  # i

#     # Take the square root of the summed squared differences
#     frobenius_distance = torch.sqrt(sum_squared_diff) # i

#     return frobenius_distance

def calculate_reconstruction_loss(real_data, fake_data):
   loss = F.binary_cross_entropy(fake_data, (real_data > .5).float(), size_average=False)
   return loss




def calculate_contrastive_loss(embedding1, embedding2, target, margin=1.0):
    assert embedding1.shape == embedding2.shape
    assert len(embedding1.shape) == 3

    losses = []
    for z1, z2 in zip(embedding1, embedding2):

        cosine_sim = F.cosine_similarity(z1, z2)

        # Create a mask for positive (similar) and negative (dissimilar) pairs
        positive_mask = target
        negative_mask = 1 - target

        # Compute the contrastive loss
        loss = (positive_mask * (1 - cosine_sim) + negative_mask * torch.clamp(cosine_sim - margin, min=0.0)).mean()
        losses.append(loss)

    return torch.stack(losses)





# def calculate_contrastive_loss(embedding1, embedding2, target, margin=1.0):
#     assert embedding1.shape == embedding2.shape
#     assert len(embedding1.shape) == 3

#     losses = []
#     for z1, z2 in zip(embedding1, embedding2):

#         # Compute the cosine similarity between the embeddings
#         cosine_sim = F.cosine_similarity(z1, z2)
#         # Compute the contrastive loss
#         loss = torch.mean((1 - target) * torch.pow(cosine_sim, 2) + target * torch.pow(torch.clamp(margin - cosine_sim, min=0.0), 2))
#         losses.append(loss)
#     return torch.stack(losses)