import pytest
import torch

from src.s2_cnn_mnist.model import Model


@pytest.mark.parametrize("batch_size", [32, 64])
def test_model(batch_size: int):
    model = Model()
    x = torch.randn(batch_size, 1, 28, 28)
    
    if x.ndim != 4:
        raise ValueError('Expected input to a 4D tensor')
        
    if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
        raise ValueError('Expected each sample to have shape [1, 28, 28]')

    y = model(x)
    assert y.shape == torch.Size([batch_size, 10])



