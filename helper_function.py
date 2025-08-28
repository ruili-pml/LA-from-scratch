import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from pathlib import Path

################################### dataset wise #####################################
class toy_dataset(torch.utils.data.Dataset):
    def __init__(self, x_data, y_data):
        self.X = x_data
        self.Y = y_data

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return [self.X[idx], self.Y[idx]]

def banana_dataloaders(data_path = './', batch_size = 32, train_ratio = 0.6, val_ratio = 0.2, test_ratio = 0.2, one_hot = False):
    """
    Generate train, test, and validation dataloaders for the banana dataset
    """
    
    ratio_sum = train_ratio + val_ratio + test_ratio
    assert ratio_sum == 1, "Ratios must sum to 1"
    
    # Load the data
    data = pd.read_csv(Path(data_path)  / 'banana.csv')
    
    x_data = data.iloc[:, :-1].values
    y_data = data.iloc[:, -1].values - 1
    
    n_data = x_data.shape[0]
    num_train_data = int(n_data * train_ratio)
    
    np.random.seed(42)
    indices = np.random.permutation(n_data)
    
    train_indices = indices[:num_train_data]
    val_indices = indices[num_train_data:num_train_data + int(n_data * val_ratio)]
    test_indices = indices[1 + num_train_data + int(n_data * val_ratio):-1]
    
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    x_train = x_data[train_indices]
    y_train = y_data[train_indices]
    
    x_test = x_data[test_indices]
    y_test = y_data[test_indices]
    
    x_val = x_data[val_indices]
    y_val = y_data[val_indices]
    
    # Normalize the data    
    X_scaler = StandardScaler().fit(x_train)
    x_train = X_scaler.transform(x_train)
    x_test = X_scaler.transform(x_test)
    x_val = X_scaler.transform(x_val)
    
    # Create the datasets
    if one_hot:
        train_dataset = toy_dataset(torch.from_numpy(x_train).float(), F.one_hot(torch.from_numpy(y_train)).float())
        test_dataset = toy_dataset(torch.from_numpy(x_test).float(), F.one_hot(torch.from_numpy(y_test)).float())
        val_dataset = toy_dataset(torch.from_numpy(x_val).float(), F.one_hot(torch.from_numpy(y_val)).float())
    else:
        
        train_dataset = toy_dataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long())
        test_dataset = toy_dataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).long())
        val_dataset = toy_dataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).long())
    # Create the dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle=False)

    return train_loader, test_loader, val_loader

