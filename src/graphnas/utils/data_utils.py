import torch
import pandas as pd
import os.path as osp
# PYG
import torch_geometric.transforms as T
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import Planetoid
# Batch execution
from torch_geometric.data import RandomNodeSampler
# Train-test split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
# OGB
from ogb.nodeproppred import PygNodePropPredDataset


ogb_data_name_conv = {
    "Arxiv": 'ogbn-arxiv',
    "MAG": 'ogbn-mag',
    "Papers": 'ogbn-papers100M',
    "Products": 'ogbn-products',
}

small_datasets = [
    "Cora",
    "Citeseer",
    "Pubmed",
]

obg_datasets = [
    "Arxiv",
    "Papers",
    "Products",
    "MAG",
]

batchable_datasets = obg_datasets + [
    "CS",
    "Physics",
    "Computers",
    "Photo",
]

available_datasets = small_datasets + batchable_datasets


def split_and_batch_data(data, batches=40):
    return \
        RandomNodeSampler(data,                   # split
                          num_parts=batches,
                          shuffle=True,
                          num_workers=10)


def split_ogb_data(dataset, dataset_name):
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    if dataset_name == 'MAG':
        split_idx = {k: v['paper']
                     for k, v in split_idx.items()}
        data.num_nodes = data.num_nodes_dict['paper']
        data.edge_index = data.edge_index_dict[('paper', 'cites', 'paper')]
        data.x = data.x_dict['paper']
        data.y = data.y_dict['paper']
        data.node_year = None
        data.num_nodes_dict = None
        data.edge_reltype = None
        data.edge_index_dict = None
        data.x_dict = None
        data.y_dict = None
    # Set split indices to masks.
    for split in ['train', 'valid', 'test']:
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[split_idx[split]] = 1
        data[str(split) + '_mask'] = mask
    data.y = data.y.flatten()
    data.num_classes = data.y.unique().shape[0]
    return data


def split_data(data, seed=10, n_splits=5):
    # Initialize or clean out data train and test masks
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    # Clean out validation mask just for safety
    data.valid_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    # Separate data into train and test, 80% train and 20% test
    # Test data is only used in the very very end
    # Train data is used in k-Fold Cross validation
    y_numpy = data.y.numpy()
    x_df, y_df = pd.DataFrame(data.x.numpy()), pd.DataFrame(y_numpy)
    X_train, X_test, y_train, y_test = \
        train_test_split(x_df, y_df,
                         test_size=0.2,  # 20% test
                         random_state=seed,
                         stratify=y_numpy)
    # If the number of splits is 1, no cross validation is performed
    if n_splits == 1:
        X_train, X_val, y_train, y_val = \
            train_test_split(X_train, y_train,
                             test_size=0.2,  # 20% validation
                             random_state=seed,
                             stratify=y_train)
        # Set the validation mask
        data.valid_mask[X_val.index] = 1
    data.train_mask[X_train.index] = 1  # Mark indices of train data
    if n_splits > 1:
        # Make K-Fold Cross Validation object
        skf = StratifiedKFold(random_state=seed,
                              shuffle=True,
                              n_splits=n_splits)
        # Separate whole training data, in order to split into folds
        t_data = data.x[data.train_mask], data.y[data.train_mask]
        data.train_folds = torch.zeros(data.num_nodes, n_splits,
                                       dtype=torch.bool)
        data.valid_folds = torch.zeros(data.num_nodes, n_splits,
                                       dtype=torch.bool)
        for i, (train_, val_) in enumerate(skf.split(t_data[0], t_data[1])):
            data.train_folds[train_, i] = 1
            data.valid_folds[val_, i] = 1
    data.test_mask[X_test.index] = 1
    print('loaded data: ', data)
    return data


def load_data(dataset_name="Cora", seed=10, n_splits=5):
    # Path in which the data will be stored
    path = osp.join(osp.dirname(osp.realpath(__file__)),
                    '..', 'data', dataset_name)
    if dataset_name in ["CS", "Physics"]:
        dataset = Coauthor(path, dataset_name, T.NormalizeFeatures())
    elif dataset_name in ["Computers", "Photo"]:
        dataset = Amazon(path, dataset_name, T.NormalizeFeatures())
    elif dataset_name in ["Cora", "Citeseer", "Pubmed"]:
        dataset = Planetoid(path, dataset_name, split='public',
                            transform=T.NormalizeFeatures())
    elif dataset_name in ["Arxiv", "Papers", "Products"]:
        dataset = PygNodePropPredDataset(
            name=ogb_data_name_conv[dataset_name],
            root=path,
            transform=T.NormalizeFeatures())
    elif dataset_name == "MAG":
        dataset = PygNodePropPredDataset(
            name=ogb_data_name_conv[dataset_name],
            root=path)
    else:
        raise Exception("[!] Dataset not found: ", str(dataset_name))
    if dataset_name in obg_datasets:
        data = split_ogb_data(dataset, dataset_name)
    else:
        data = dataset[0]  # pyg graph object
        data = split_data(data, seed, n_splits)
        data.num_classes = dataset.num_classes
    return data
