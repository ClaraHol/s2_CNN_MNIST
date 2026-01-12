from src.s2_cnn_mnist.model import Model
from src.s2_cnn_mnist.data import corrupt_mnist
import torch
import matplotlib.pyplot as plt
import wandb
#from sklearn.metrics import RocCurveDisplay

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)




def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10, model_name: str = "models/model.pth") -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    
    # Start a new wandb run to track this script.
    """
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="claho-danmarks-tekniske-universitet-dtu",
        # Set the wandb project where this run will be logged.
        project="cnn_mnist",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": lr,
            "architecture": "CNN",
            "dataset": "MNIST",
            "epochs": epochs,
            "batch_size": batch_size
        },
       
    )
    """
    print("Model")
    model = Model().to(DEVICE)

    print("Load data")
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    print("Optimizer")
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print("Start training")
    #statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()
        preds, targets = [], []
        print(epoch)
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            #statistics["train_loss"].append(loss.item())
        
            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            #statistics["train_accuracy"].append(accuracy)

            preds.append(y_pred.detach().cpu())
            targets.append(target.detach().cpu())

            if i % 100 == 0:
                # Log metrics to wandb.
                #run.log({"Epoch": epoch, "loss": loss.item(), "train_accuracy": accuracy})
                # add a plot of the input images
                #images = wandb.Image(img[:5].detach().cpu(), caption="Input images")
                #wandb.log({"images": images})

                # add a plot of histogram of the gradients
                grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0)
                #wandb.log({"gradients": wandb.Histogram(grads)})

    print("Training complete")
    # add a custom matplotlib plot of the ROC curves
    preds = torch.cat(preds, 0)
    targets = torch.cat(targets, 0)
    """
    for class_id in range(10):
        one_hot = torch.zeros_like(targets)
        one_hot[targets == class_id] = 1
        _ = RocCurveDisplay.from_predictions(
            one_hot,
            preds[:, class_id],
            name=f"ROC curve for {class_id}",
            plot_chance_level=(class_id == 2),
        )

    wandb.log({"roc": wandb.Image(plt)})
    wandb.plot({"roc": plt})
    plt.close()  # close the plot to avoid memory leaks and overlapping figures
    """

    # Save the model
    #art = wandb.Artifact(name="cnn_model", type = "model")
    torch.save(model.state_dict(), model_name)
    
    #art.add_file(local_path=model_name, name="Model")
    #run.log_artifact(art)
    #run.finish()
    
    #print("Training complete")
    #
    #fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    #axs[0].plot(statistics["train_loss"])
    #axs[0].set_title("Train loss")
    #axs[1].plot(statistics["train_accuracy"])
    #axs[1].set_title("Train accuracy")

    


if __name__ == "__main__":
    train()
