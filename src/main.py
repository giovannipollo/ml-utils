import argparse
from utils.config_parser import parse_config
from utils.logger import get_logger
from data.dataset import get_data_loaders
from models.simple_nn import SimpleNN_MNIST
from training.train import train_model, evaluate_model
import shutil

def main():
    """
    Main function to run the neural network training process.
    """
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="NN Training from YAML Configuration")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    # Parse configuration file
    config = parse_config(config_path=args.config)

    # Set up logger
    logger = get_logger(
        log_dir=config["logging"]["log_dir"], log_name=config["experiment"]["name"]
    )

    # Get data loaders
    train_loader, test_loader = get_data_loaders(
        data_config=config["data"], training_config=config["training"]
    )

    # Initialize model
    if config["model"]["name"] == "SimpleNN_MNIST":
        model = SimpleNN_MNIST(config)
    else:
        raise ValueError("Unsupported model type")

    logger.info("Copying the config file to the experiment directory")
    # Start training process
    logger.info("Starting training process")
    train_model(
        config=config,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        logger=logger,
    )

    # Evaluate model
    accuracy = evaluate_model(model=model, test_loader=test_loader)
    logger.info(f"Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
