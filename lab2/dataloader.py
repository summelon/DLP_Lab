import os
import torch
import numpy as np


def read_bci_data(dataset_pt):
    S4b_train = np.load(os.path.join(dataset_pt, 'S4b_train.npz'))
    X11b_train = np.load(os.path.join(dataset_pt, 'X11b_train.npz'))
    S4b_test = np.load(os.path.join(dataset_pt, 'S4b_test.npz'))
    X11b_test = np.load(os.path.join(dataset_pt, 'X11b_test.npz'))

    train_data = np.concatenate(
            (S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate(
            (S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate(
            (S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate(
            (S4b_test['label'], X11b_test['label']), axis=0)

    train_label = train_label - 1
    test_label = test_label - 1
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

    train_data = torch.tensor(train_data, dtype=torch.float32).to('cuda:0')
    train_label = torch.tensor(train_label, dtype=torch.long).to('cuda:0')
    test_data = torch.tensor(test_data, dtype=torch.float32).to('cuda:0')
    test_label = torch.tensor(test_label, dtype=torch.long).to('cuda:0')

    return train_data, train_label, test_data, test_label
