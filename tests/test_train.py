
from tests import _PROJECT_ROOT
from src.s2_cnn_mnist.train import train
from pathlib import Path

def test_train():
    try:
        train(epochs=1, model_name="models/test_model.pth")
    except:
        print("Training did not finnish correctly")

    model_path = Path(_PROJECT_ROOT + "/models/test_model.pth")
    assert model_path.is_file()