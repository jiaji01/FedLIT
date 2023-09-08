from sklearn.model_selection import KFold
import numpy as np




def k_fold_split_indices(length_of_data, number_of_folds, current_fold_id):
    """split indice of fold n in k-fold

    Args:
        len_of_data (int)
        num_of_folds (_type_)
        current_fold_id (_type_)
    
    Returns:
        list: indices of trainset
        list: indices pf testset
        
    """
    kf = KFold(n_splits=number_of_folds,shuffle=True, random_state= 42)
    split_indices = kf.split(range(length_of_data))
    train_indices, test_indices = [(list(train), list(test)) for train, test in split_indices][current_fold_id]
    return train_indices, test_indices



def mix_dataset(datset_list):
    mixed_dataset = []

    for datset in datset_list:
        mixed_dataset.extend(datset)
    
    np.random.shuffle(mixed_dataset)

    return mixed_dataset