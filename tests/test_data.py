import os.path

import pytest
import torch


def corrupt_mnist() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test dataloaders for corrupt MNIST."""
    train_images = torch.load("data/processed/train_images.pt")
    train_target = torch.load("data/processed/train_target.pt")
    test_images = torch.load("data/processed/test_images.pt")
    test_target = torch.load("data/processed/test_target.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)

    return train_set, test_set

@pytest.mark.skipif(not os.path.exists("data/processed/train_images.pt"), reason="Data files not found")
def test_data():
    train, test = corrupt_mnist()
    N_train = 30000
    N_test = 5000
    assert len(train) == N_train, "Dataset did not have the correct number of training samples" 
    assert len(test) == N_test, "Dataset did not have the correct number of test samples" 
    assert sum([t.shape == torch.Size([1,28,28]) for t,_ in train]) == N_train, "Some of the training images have the wrong shape" 
    assert sum([t.shape == torch.Size([1, 28, 28]) for t,_ in test]) == N_test, "Some of the test images have the wrong shape" 
    labels = [int(l) for _, l in train]
    unique_labels = set(labels)
    l = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    assert  unique_labels == l, "Some labels are missing."    
    
    
