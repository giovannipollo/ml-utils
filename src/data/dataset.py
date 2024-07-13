import torch
from torchvision import datasets, transforms


def get_data_loaders(data_config, training_config):
    transform = transforms.Compose([transforms.ToTensor()])

    if data_config["dataset"] == "MNIST":
        train_dataset = datasets.MNIST(
            data_config["data_dir"], train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            data_config["data_dir"], train=False, download=True, transform=transform
        )
    else:
        raise ValueError("Unsupported dataset")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=data_config["shuffle"],
        num_workers=data_config["num_workers"],
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
    )

    return train_loader, test_loader
