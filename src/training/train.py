import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import os
import datetime
from torchsummary import summary

def train_model(config, model, train_loader, test_loader, logger):
    """
    Train a PyTorch model.

    Args:
        config (dict): Configuration dictionary.
        model (nn.Module): PyTorch model to train.
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for test data.
        logger (logging.Logger): Logger object for logging information.

    Returns:
        None
    """
    training_config = config["training"]
    log_dir = config["logging"]["log_dir"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(log_dir, timestamp))

    criterion = nn.CrossEntropyLoss()

    # Initialize optimizer
    if training_config["optimizer"] == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=training_config["learning_rate"],
            weight_decay=training_config["weight_decay"],
        )
    elif training_config["optimizer"] == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=training_config["learning_rate"],
            momentum=training_config["momentum"],
            weight_decay=training_config["weight_decay"],
        )
    else:
        raise ValueError("Unsupported optimizer")

    # Initialize learning rate scheduler
    scheduler = StepLR(
        optimizer,
        step_size=training_config["step_size"],
        gamma=training_config["gamma"],
    )

    # Training loop
    for epoch in range(training_config["epochs"]):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Log training progress
            if batch_idx % config["logging"]["log_interval"] == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{training_config['epochs']}], "
                    f"Step [{batch_idx}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}"
                )
                writer.add_scalar(
                    "training loss",
                    running_loss / (batch_idx + 1),
                    epoch * len(train_loader) + batch_idx,
                )

        scheduler.step()

        # Save model checkpoint
        if epoch % config["logging"]["save_interval"] == 0:
            checkpoint_dir = os.path.join(log_dir, timestamp, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Model checkpoint saved to {checkpoint_path}")

    logger.info("Training finished.")
    writer.close()

def evaluate_model(model, test_loader):
    """
    Evaluate a PyTorch model.

    Args:
        model (nn.Module): PyTorch model to evaluate.
        test_loader (DataLoader): DataLoader for test data.

    Returns:
        float: Accuracy of the model on the test dataset.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    return accuracy